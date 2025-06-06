# GPT-SoVITS的rust推理实现，专注于CPU部署

该项目是 <https://github.com/mzdk100/GPT-SoVITS> 的fork项目，旨在通过ONNX Runtime，在较新的中高端X86/ARM CPU设备上，达到可以接受的延迟速度。

虽然作者在原readme中写道："在移动端推理的时间很慢，基本上达到无法忍受的程度，但在桌面平台上推理还算不错。"，但是由于我觉得这个结论有些不符合预期，所以打算继续探索全平台的部署方案。

目前的计划包括：

1. 升级onnxruntime，并始终保持最新 (done)
2. 在较新的x86 CPU上优化模型，在保证效果的前提下，使得其具有可以接受的推理延迟 (Current Done)
    1. 导出原始onnx模型(done)
    2. 使用常见的onnx优化/裁剪工具优化模型(done)
    3. 量化模型到int4( FAILED， 有一定的耗时提升，但是quant后的精度不对)
    4. 尝试使用更新的op set(done)
    5. 尝试使用除了onnxruntime之外的后端，比如(paused, 后续尝试openVINO)
    6. 给出一键优化脚本
3. 在armv9设备上优化模型，在保证效果的前提下，使得其具有可以接受的推理延迟 (doing)
    1. 导出onnx fp16模型(done)
    2. 部署到高通骁龙8 elite上并测试性能
    3. 尝试使用除了onnxruntime之外的后端，比如mnn/ncnn(放弃了，没有好用的rust绑定)

当前x86性能请参考 perf.md，不同设备上可能有不同的优化策略

## 为什么不使用gpt-sovits-rs 的 libtorch方案

如果您的目标是在x86（+GPU）设备上使用静态语言部署，建议您使用 <https://github.com/second-state/gpt_sovits_rs> ， libtorch能够保证最佳的质量和性能。
libtorch方案在x86上确实有相当优秀的性能和多样的设备后端可选。但目前pytorch官方已经停止了pytorch mobile的支持，转而建议使用executorch。这带来了几个问题：

1. executorch不支持torchscript，现有的模型转换脚本需要重写，但是我对executorch的特性并不熟悉（主要原因）
2. 移动端（executorch）和x86（libtorch）的模型不一致，导致需要更多的开发和对齐工作
3. 个人并不了解executorch的部署优化方法（除了部署到高通qnn和mtk的np8，但是这有又需要额外的量化工作和量化数据集）
