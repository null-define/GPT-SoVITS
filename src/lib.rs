use hound::{WavReader, WavSpec, WavWriter};
use ndarray::{s, Array, ArrayView, Axis, IntoDimension, IxDyn};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::Session,
    value::{Tensor, TensorRef},
};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::SystemTime,
};

// Module imports (assuming these are in separate files)
mod conversion;
mod detection;
mod error;
mod phoneme;
mod symbol;

pub use {
    conversion::*,
    detection::*,
    error::GptSoVitsError, // Explicitly import the error type
    phoneme::*,
    symbol::*,
};

fn argmax(tensor: &ArrayView<f32, IxDyn>) -> (usize, usize) {
    let mut max_index = (0, 0);
    let mut max_value = tensor
        .get(IxDyn::zeros(2))
        .copied()
        .unwrap_or(f32::NEG_INFINITY);

    for i in 0..tensor.shape()[0] {
        for j in 0..tensor.shape()[1] {
            if let Some(value) = tensor.get((i, j).into_dimension()) {
                if *value > max_value {
                    max_value = *value;
                    max_index = (i, j);
                }
            }
        }
    }

    max_index
}

const EARLY_STOP_NUM: usize = 2700;
const T2S_DECODER_EOS: usize = 1024;

pub fn run<P: AsRef<Path>>(
    sovits_path: P,
    ssl_path: P,
    t2s_encoder_path: P,
    t2s_fs_decoder_path: P,
    t2s_s_decoder_path: P,
    reference_audio_path: P,
    ref_text: &str,
    text: &str,
) -> Result<(), GptSoVitsError> {
    type DType = f32;
    // Initialize ONNX sessions
    let mut sovits = Session::builder()?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .commit_from_file(sovits_path)?;
    let mut ssl = Session::builder()?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .commit_from_file(ssl_path)?;
    let mut t2s_encoder = Session::builder()?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .commit_from_file(t2s_encoder_path)?;
    let mut t2s_fs_decoder = Session::builder()?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .commit_from_file(t2s_fs_decoder_path)?;
    let mut t2s_s_decoder = Session::builder()?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .commit_from_file(t2s_s_decoder_path)?;

    // Text processing
    let mut extractor = PhonemeExtractor::default();
    extractor.push_str(text);
    let text_seq = extractor.get_phone_ids();
    println!("{:?}", extractor);
    let text_seq = Array::from_shape_vec((1, text_seq.len()), text_seq)?;
    let text_bert = Array::<f32, _>::zeros((text_seq.shape()[1], 1024));

    extractor = PhonemeExtractor::default();
    extractor.push_str(ref_text);
    let ref_seq = extractor.get_phone_ids();
    println!("{:?}", extractor); // Reference sequence

    let ref_seq = Array::from_shape_vec((1, ref_seq.len()), ref_seq)?;
    let ref_bert = Array::<f32, _>::zeros((ref_seq.shape()[1], 1024));

    // Read and resample reference audio to 16kHz
    let mut file = File::open(&reference_audio_path)?;
    let wav_reader = WavReader::new(file)?;
    let spec = wav_reader.spec();
    let audio_samples: Vec<f32> = wav_reader
        .into_samples::<i16>()
        .collect::<Result<Vec<i16>, _>>()?
        .into_iter()
        .map(|s| s as f32 / i16::MAX as f32)
        .collect();
    let ref_audio_16k = if spec.sample_rate != 16000 {
        resample_audio(audio_samples.clone(), spec.sample_rate, 16000)?
    } else {
        audio_samples.clone()
    };
    let ref_audio_16k = Array::from_shape_vec((1, ref_audio_16k.len()), ref_audio_16k)?;

    // SSL processing
    let time = SystemTime::now();
    let ssl_output = ssl.run(ort::inputs![
        "ref_audio_16k" => Tensor::from_array(ref_audio_16k)?
    ])?;
    println!("SSL Cost is {:?}", time.elapsed()?);

    // T2S Encoder
    let time = SystemTime::now();
    let encoder_output = t2s_encoder.run(ort::inputs![
        "ref_seq" => Tensor::from_array(ref_seq.to_owned())?,
        "text_seq" => Tensor::from_array(text_seq.to_owned())?,
        "ref_bert" => Tensor::from_array(ref_bert.to_owned())?,
        "text_bert" => Tensor::from_array(text_bert.to_owned())?,
        "ssl_content" => Tensor::from_array(ssl_output["ssl_content"].try_extract_array::<f32>()?.to_owned())?,
    ])?;
    println!("T2S Encoder Cost is {:?}", time.elapsed()?);
    let x = encoder_output["x"].try_extract_array::<f32>()?;
    let prompts = encoder_output["prompts"].try_extract_array::<i64>()?;
    let prefix_len = prompts.dim()[1];

    // T2S FS Decoder
    let time = SystemTime::now();
    let fs_decoder_output = t2s_fs_decoder.run(ort::inputs![
        "x" => Tensor::from_array(x.to_owned())?,
        "prompts" => Tensor::from_array(prompts.to_owned())?
    ])?;
    println!("T2S FS Decoder Cost is {:?}", time.elapsed()?);
    let (mut y, mut k, mut v, mut y_emb, x_example) = (
        fs_decoder_output["y"]
            .try_extract_array::<i64>()?
            .to_owned(),
        fs_decoder_output["k"]
            .try_extract_array::<DType>()?
            .to_owned(),
        fs_decoder_output["v"]
            .try_extract_array::<DType>()?
            .to_owned(),
        fs_decoder_output["y_emb"]
            .try_extract_array::<DType>()?
            .to_owned(),
        fs_decoder_output["x_example"]
            .try_extract_array::<DType>()?
            .to_owned(),
    );

    // T2S S Decoder
    let time = SystemTime::now();
    let mut idx = 1;
    let pred_semantic = loop {
        let (y_new, k_new, v_new, y_emb_new, samples, logits) = {
            let time = SystemTime::now();

            // New scope for s_decoder_output
            let s_decoder_output = t2s_s_decoder.run(ort::inputs![
                "iy" => Tensor::from_array(y.to_owned())?,
                "ik" => Tensor::from_array(k.to_owned())?,
                "iv" => Tensor::from_array(v.to_owned())?,
                "iy_emb" => Tensor::from_array(y_emb.to_owned())?,
                "ix_example" => Tensor::from_array(x_example.to_owned())?
            ])?;
            println!("T2S S Decoder Forward Once Cost is {:?}", time.elapsed()?);

            (
                s_decoder_output["y"].try_extract_array::<i64>()?.to_owned(),
                s_decoder_output["k"].try_extract_array::<DType>()?.to_owned(),
                s_decoder_output["v"].try_extract_array::<DType>()?.to_owned(),
                s_decoder_output["y_emb"]
                    .try_extract_array::<DType>()?
                    .to_owned(),
                s_decoder_output["samples"]
                    .try_extract_array::<i32>()?
                    .to_owned(),
                s_decoder_output["logits"]
                    .try_extract_array::<f32>()?
                    .to_owned(),
            )
        }; // s_decoder_output is dropped here
        let time = SystemTime::now();

        y = y_new;
        k = k_new;
        v = v_new;
        y_emb = y_emb_new;

        println!("y: {:?}", y.last());
        println!("y shape: {:?}", y.shape());
        println!("idx: {:?}", idx);

        if idx >= 1500
            || (y.dim()[1] - prefix_len) > EARLY_STOP_NUM.try_into().unwrap()
            || argmax(&logits.view()).1 == T2S_DECODER_EOS
            || samples
                .get((0, 0).into_dimension())
                .unwrap_or(&(T2S_DECODER_EOS as i32))
                == &(T2S_DECODER_EOS as i32)
        {
            let mut y = y.to_owned();
            y.last_mut().and_then(|i| {
                *i = 0;
                None::<()>
            });

            break y
                .slice(s![.., y.shape()[1] - idx..; 1])
                .into_owned()
                .insert_axis(Axis(0));
        }

        idx += 1;
        println!("T2S S Decoder Forward Once Other Cost is {:?}", time.elapsed()?);
    };

    println!("T2S S Decoder Cost is {:?}", time.elapsed()?);

    // Read reference audio at 32kHz
    // Final SoVITS processing

    let ref_audio_32k = resample_audio(audio_samples, spec.sample_rate, 32000)?;
    let ref_audio = Array::from_shape_vec((1, ref_audio_32k.len()), ref_audio_32k)?;
    let time = SystemTime::now();
    let outputs = sovits.run(ort::inputs![
        "text_seq" => Tensor::from_array(text_seq.to_owned())?,
        "pred_semantic" => Tensor::from_array(pred_semantic.to_owned())?,
        "ref_audio" =>  Tensor::from_array(ref_audio)?
    ])?;
    let output = outputs["audio"].try_extract_array::<f32>()?;
    println!("SoVITS Cost is {:?}", time.elapsed()?);

    // Save output to WAV file
    let spec = WavSpec {
        channels: 1,
        sample_rate: 32000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = WavWriter::create("output.wav", spec)?;
    for sample in output.as_slice().unwrap() {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;

    Ok(())
}

// Simple resampling function (linear interpolation)
fn resample_audio(
    input: Vec<f32>,
    in_rate: u32,
    out_rate: u32,
) -> Result<Vec<f32>, GptSoVitsError> {
    if in_rate == out_rate {
        return Ok(input);
    }

    let ratio = in_rate as f32 / out_rate as f32;
    let out_len = ((input.len() as f32 / ratio).ceil() as usize).max(1);
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_idx = i as f32 * ratio;
        let idx_floor = src_idx.floor() as usize;
        let frac = src_idx - idx_floor as f32;

        if idx_floor + 1 < input.len() {
            let sample = input[idx_floor] * (1 as f32 - frac) + input[idx_floor + 1] * frac;
            output.push(sample);
        } else if idx_floor < input.len() {
            output.push(input[idx_floor]);
        } else {
            output.push(0.);
        }
    }

    Ok(output)
}
