# ANDROID 二进制构建

为了保证onnx runtime的构建最新，不提供二进制分发，建议自行构建onnx runtime

1. 保证自己的 CMake  >= 3.28 ,如果OS内部的包管理器不支持，建议使用conda环境安装
2. 自行下载ndk和sdk，并设置环境变量
3. 使用onnx/build_android构建onnxruntime
4. 使用build_for_android.sh构建Android可执行文件和库