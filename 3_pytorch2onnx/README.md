# PyTorch 转 ONNX 详解

[原教程](https://mmdeploy.readthedocs.io/zh-cn/latest/tutorial/03_pytorch2onnx.html)

---

<br>
<br>

## `torch.onnx.export` 详解

    torch.onnx.export 可以通过两种方法得到 onnx 模型：
    - torch.jit.trace (无控制流, 默认)
    - torch.jit.script (有控制流)

    无论是 trace 还是 script 都会转为中间格式 torch.jit.ScriptModule, 再通过 torch.onnx.export 转化为 onnx 模型



## Pytorch 对 ONNX 的算子支持
