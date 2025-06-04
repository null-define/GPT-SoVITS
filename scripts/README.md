# 使用说明

1. 拷贝export_onnx_v2.py到GPT-Sovits python项目目录，和原来的onnx export脚本同级
2. 执行脚本，导出所有需要的onnx模型
3. 修改脚本中的路径为自己的模型路径（别问我为啥叫kaoyu）
4. 按照onnxsim -> onnxslim(但是VITS报错，还在看为啥)  ->quant + update op version 的顺序执行脚本