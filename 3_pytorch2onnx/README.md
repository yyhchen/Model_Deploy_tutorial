# PyTorch 转 ONNX 详解

[原教程](https://mmdeploy.readthedocs.io/zh-cn/latest/tutorial/03_pytorch2onnx.html)


.py 使用顺序:
1. trace_script.py
2. param_dynamic_axes.py
3. test_dynamic_axes.py


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

使用 `torch.onnx.export` 可能会出现算子不兼容等问题，可能有一下几种情况：

- 该算子可以一对一地翻译成一个 ONNX 算子。

- 该算子在 ONNX 中没有直接对应的算子，会翻译成一至多个 ONNX 算子。

- 该算子没有定义翻译成 ONNX 的规则，报错。


<br>


### onnx 算子文档
[onnx 官方算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

表格的第一列是算子名，第二列是该算子发生变动的算子集版本号，也就是我们之前在 `torch.onnx.export` 中提到的 `opset_version` 表示的算子集版本号。

<br>


### pytorch对onnx算子映射
[pytorch 算子映射](https://github.com/pytorch/pytorch/tree/main/torch/onnx)

`symbloic_opset{n}.py`（符号表文件）即表示 PyTorch 在支持第 n 版 ONNX 算子集时新加入的内容。

