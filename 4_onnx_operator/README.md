# 在 PYTORCH 中支持更多 ONNX 算子

[原教程](https://mmdeploy.readthedocs.io/zh-cn/latest/tutorial/04_onnx_custom_op.html#id2)

在开始之前，我们有必要解释一些相关的概念：

## Aten 是什么？
ATen 本质上是一个张量库，PyTorch 中几乎所有其他 Python 和 C++ 接口都是在其之上构建的。它提供了一个核心 `Tensor` 类，在该类上定义了数百个操作。大多数这些操作都有 CPU 和 GPU 实现， `Tensor` 类将根据其类型动态调度到这些实现

<br>

## 什么是符号函数？
符号函数，可以看成是 PyTorch 算子类的一个静态方法。在把 PyTorch 模型转换成 ONNX 模型时，各个 PyTorch 算子的符号函数会被依次调用，以完成 PyTorch 算子到 ONNX 算子的转换。


<br>
<br>

正文
---

在实际的部署过程中，难免碰到模型无法用原生 PyTorch 算子表示的情况。这个时候，我们就得考虑扩充 PyTorch，即在 PyTorch 中支持更多 ONNX 算子。

但是我们可能会遇到一些问题，比如：
1. 算子在pytorch中没有实现
2. 缺少算子间的映射关系
3. onnx中没有相应的算子

<br>

解决办法如下：
- PyTorch 算子
    - 组合现有算子
    - 添加 TorchScript 算子
    - 添加普通 C++ 拓展算子

- 映射方法
    - 为 ATen 算子添加符号函数
    - 为 TorchScript 算子添加符号函数
    - 封装成 torch.autograd.Function 并添加符号函数

- ONNX 算子
    - 使用现有 ONNX 算子
    - 定义新 ONNX 算子

<br>
<br>
