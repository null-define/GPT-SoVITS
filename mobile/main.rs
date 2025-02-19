use std::path::Path;
use gpt_sovits::run;

#[mobile_entry_point::mobile_entry_point]
fn main() {
    let assets_dir = Path::new("/data/local/tmp");
    run(
        assets_dir.join("xxx_vits.onnx"),
        assets_dir.join("xxx_ssl.onnx"),
        assets_dir.join("xxx_t2s_encoder.onnx"),
        assets_dir.join("xxx_t2s_fs_decoder.onnx"),
        assets_dir.join("xxx_t2s_s_decoder.onnx"),
        assets_dir.join("hello_in_cn.mp3"),
        "你好呀，我们是一群追逐梦想的人！你好，我是智能语音助手。"
    ).unwrap();
}