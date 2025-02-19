mod conversion;
mod detection;
mod error;
mod phoneme;
mod symbol;

use std::{
    fs::File,
    io::{Seek, SeekFrom},
    path::Path,
    time::SystemTime,
};
pub use {conversion::*, detection::*, error::*, phoneme::*, symbol::*};

use ndarray::{s, Array, ArrayView, Axis, IntoDimension, IxDyn};
use ort::{execution_providers::CUDAExecutionProvider, inputs, session::Session};
use rodio::{
    buffer::SamplesBuffer, source::UniformSourceIterator, Decoder, OutputStream, Sink, Source,
};

fn argmax(tensor: &ArrayView<f32, IxDyn>) -> (usize, usize) {
    let mut max_index = (0, 0);
    let mut max_value = tensor.get(IxDyn::zeros(2));

    for i in 0..tensor.shape()[0] {
        for j in 0..tensor.shape()[1] {
            let value = tensor.get((i, j).into_dimension());
            if value > max_value {
                max_value = value;
                max_index = (i, j);
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
    text: &str,
) -> Result<(), GptSoVitsError> {
    let sovits = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(sovits_path)?;
    let ssl = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(ssl_path)?;
    let t2s_encoder = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(t2s_encoder_path)?;
    let t2s_fs_decoder = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(t2s_fs_decoder_path)?;
    let t2s_s_decoder = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(t2s_s_decoder_path)?;

    let mut extractor = PhonemeExtractor::default();
    extractor.push_str(text);
    let text_seq = extractor.get_phone_ids();
    println!("{:?}", extractor);
    let text_seq = Array::from_shape_vec((1, text_seq.len()), text_seq)?;
    let text_bert = Array::<f32, _>::zeros((text_seq.shape()[1], 1024));
    let ref_seq = vec![
        227i64, 167, 158, 119, 97, 1, 316, 232, 251, 169, 320, 169, 227, 146, 318, 257, 318, 196,
        320, 258, 251, 242,
    ];
    let ref_seq = Array::from_shape_vec((1, ref_seq.len()), ref_seq)?;
    let ref_bert = Array::<f32, _>::zeros((ref_seq.shape()[1], 1024));

    let mut file = File::open(reference_audio_path)?;
    let ref_audio_16k = UniformSourceIterator::new(
        Decoder::new(file.try_clone()?)?.convert_samples::<f32>(),
        1,
        16000,
    )
    .collect::<Vec<f32>>();
    let ref_audio_16k = Array::from_shape_vec((1, ref_audio_16k.len()), ref_audio_16k)?;

    let time = SystemTime::now();
    let ssl_output = ssl.run(inputs!["ref_audio_16k" => ref_audio_16k.view()]?)?;
    println!("SSL Cost is {:?}", time.elapsed()?);

    let time = SystemTime::now();
    let encoder_output = t2s_encoder.run(inputs![
        "ref_seq" => ref_seq.view(),
        "text_seq" => text_seq.view(),
        "ref_bert" => ref_bert,
        "text_bert" => text_bert,
        "ssl_content" => ssl_output["ssl_content"].try_extract_tensor::<f32>()?,
    ]?)?;
    println!("T2S Encoder Cost is {:?}", time.elapsed()?);
    let x = encoder_output["x"].try_extract_tensor::<f32>()?;
    let prompts = encoder_output["prompts"].try_extract_tensor::<i64>()?;
    let prefix_len = prompts.shape()[1];

    let time = SystemTime::now();
    let fs_decoder_output = t2s_fs_decoder.run(inputs![
        "x" => x.view(),
        "prompts" => prompts.view()
    ]?)?;
    println!("T2S FS Decoder Cost is {:?}", time.elapsed()?);
    let (mut y, mut k, mut v, mut y_emb, x_example) = (
        fs_decoder_output["y"].try_extract_tensor::<i64>()?,
        fs_decoder_output["k"].try_extract_tensor::<f32>()?,
        fs_decoder_output["v"].try_extract_tensor::<f32>()?,
        fs_decoder_output["y_emb"].try_extract_tensor::<f32>()?,
        fs_decoder_output["x_example"].try_extract_tensor::<f32>()?,
    );

    let time = SystemTime::now();
    let mut s_decoder_output = t2s_s_decoder.run(inputs![
        "iy" => y.view(),
        "ik" => k.view(),
        "iv" => v.view(),
        "iy_emb" => y_emb.view(),
        "ix_example" => x_example.view()
    ]?)?;
    println!("T2S S Decoder Cost is {:?}", time.elapsed()?);
    let mut idx = 1;
    let pred_semantic = loop {
        y = s_decoder_output["y"].try_extract_tensor()?;
        k = s_decoder_output["k"].try_extract_tensor()?;
        v = s_decoder_output["v"].try_extract_tensor()?;
        y_emb = s_decoder_output["y_emb"].try_extract_tensor()?;
        let samples = s_decoder_output["samples"].try_extract_tensor::<i32>()?;
        let logits = s_decoder_output["logits"].try_extract_tensor::<f32>()?;
        if idx >= 1500
            || (y.shape()[1] - prefix_len) > EARLY_STOP_NUM
            || argmax(&logits).1 == T2S_DECODER_EOS
            || samples
                .get((0, 0).into_dimension())
                .unwrap_or(&(T2S_DECODER_EOS as i32))
                == &(T2S_DECODER_EOS as i32)
        {
            let mut y = y.into_owned();
            y.last_mut().and_then(|i| {
                *i = 0;
                None::<()>
            });
            break y
                .slice(s![.., y.shape()[1] - idx..])
                .into_owned()
                .insert_axis(Axis(0));
        }

        s_decoder_output = t2s_s_decoder.run(inputs![
            "iy" => y.view(),
            "ik" => k.view(),
            "iv" => v.view(),
            "iy_emb" => y_emb.view(),
            "ix_example" => x_example.view()
        ]?)?;
        idx += 1;
    };

    file.seek(SeekFrom::Start(0))?;
    let ref_audio_32k =
        UniformSourceIterator::new(Decoder::new(file)?.convert_samples::<f32>(), 1, 32000)
            .collect::<Vec<f32>>();
    let ref_audio = Array::from_shape_vec((1, ref_audio_32k.len()), ref_audio_32k)?;

    let time = SystemTime::now();
    let outputs = sovits.run(inputs![
        "text_seq" => text_seq.view(),
        "pred_semantic" => pred_semantic.view(),
        "ref_audio" => ref_audio.view()
    ]?)?;
    let output = outputs["audio"].try_extract_tensor::<f32>()?;
    println!("{:?}, {:?}", time.elapsed().unwrap(), output);
    play_sound(output.as_slice().unwrap());
    Ok(())
}

fn play_sound(data: &[f32]) {
    let (_stream, handle) = OutputStream::try_default().unwrap();
    let player = Sink::try_new(&handle).unwrap();
    player.append(SamplesBuffer::new(1, 32000, data));
    player.sleep_until_end()
}
