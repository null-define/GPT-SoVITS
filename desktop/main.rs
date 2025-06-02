use gpt_sovits::run;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let assets_dir = Path::new("/home/qiang/projects/GPT-SoVITS/onnxsim-onnx/kaoyu");
    run(
        assets_dir.join("kaoyu_vits.onnx"),
        assets_dir.join("kaoyu_ssl.onnx"),
        assets_dir.join("kaoyu_t2s_encoder.onnx"),
        assets_dir.join("kaoyu_t2s_fs_decoder.onnx"),
        assets_dir.join("kaoyu_t2s_s_decoder.onnx"),
        assets_dir.join("ref.wav"),
        "格式化，可以给自家的奶带来大量的",
        "你好呀，我们是一群追逐梦想的人！"
    )?;
    Ok(())
}
