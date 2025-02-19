use gpt_sovits::run;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let assets_dir = Path::new("assets");
    run(
        assets_dir.join("xxx_vits.onnx"),
        assets_dir.join("xxx_ssl.onnx"),
        assets_dir.join("xxx_t2s_encoder.onnx"),
        assets_dir.join("xxx_t2s_fs_decoder.onnx"),
        assets_dir.join("xxx_t2s_s_decoder.onnx"),
        assets_dir.join("hello_in_cn.mp3"),
        "你好呀，我们是一群追逐梦想的人！你好，我是智能语音助手。现在GPT-SoVits已经使用Rust实现了跨平台推理。"
    )?;
    Ok(())
}
