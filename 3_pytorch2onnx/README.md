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


**记录下，我运行代码倒出来的模型无论是哪种都是完整的图，不像教程里面说的 script 是用loop代替了所有节点**

<br>
<br>

 **由于推理引擎对静态图的支持更好，通常我们在模型部署时不需要显式地把 PyTorch 模型转成 TorchScript 模型，直接把 PyTorch 模型用 torch.onnx.export 跟踪导出即可**。了解这部分的知识主要是为了在模型转换报错时能够更好地定位问题是否发生在 PyTorch 转 TorchScript 阶段。


<br>
<br>
<br>


## Pytorch 对 ONNX 的算子支持
