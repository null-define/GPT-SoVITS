# GPT-SoVITS的rust推理实现

本项目是使用rust实现的一个GPT-SoVITS模型的推理，GPT-SoVITS是一个语音合成和语音克隆模型，他有出色的性能被人们所熟知，但因为推理受限于pytorch环境，在部署时会很不方便，因此本项目孕育而生。
本项目使用的是onnx格式的模型，他需要将GPT-SoVits的pytorch模型转换成onnx格式后才能进行推理。
项目中提供一个用于桌面平台的推理示例(desktop)，可以用于windows和MacOS，还包含一个安卓示例(mobile)，理论上ios也同样受支持，但没有充分测试。
在移动端推理的时间很慢，基本上达到无法忍受的程度，但在桌面平台上推理还算不错。
