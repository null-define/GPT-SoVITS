# GPT-SoVITS的rust推理实现，专注于CPU部署

该项目是 https://github.com/mzdk100/GPT-SoVITS 的fork项目，旨在通过ONNX Runtime，在较新的中高端X86/ARM CPU设备上，达到可以接受的延迟速度。

虽然作者在原readme中写道："在移动端推理的时间很慢，基本上达到无法忍受的程度，但在桌面平台上推理还算不错。"，但是由于我觉得这个结论有些不符合预期，所以打算继续探索全平台的部署方案。

目前的计划包括：

1. 升级onnxruntime，并始终保持最新 (done)
2. 在较新的x86 CPU上优化模型，在保证效果的前提下，使得其具有可以接受的推理延迟 (doing)
    1. 导出原始onnx模型(done)
    2. 使用常见的onnx优化/裁剪工具优化模型(done)
    3. 量化模型到int4(doing)
    4. 尝试使用更新的op set(done)
    5. 尝试使用除了onnxruntime之外的后端，比如candle
3. 在armv9设备上优化模型，在保证效果的前提下，使得其具有可以接受的推理延迟 (to do)
    1. 导出onnx fp16模型
    2. 量化模型到int4
    3. 尝试使用除了onnxruntime之外的后端，比如mnn/ncnn


当前x86性能请参考 perf.md，不同设备上可能有不同的优化策略