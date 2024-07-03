# 在 PYTORCH 中支持更多 ONNX 算子

**本章节是在 MAC 上进行的，所以一些编译的内容在其他平台可能无法使用，需要删掉重新编译！！**

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


## register_op.py 
实际部署过程中，遇到算子缺失问题的一种：
在pytorch中有算子定义及相关实现，在onnx中也有定义和实现，但是两者缺乏映射关系
<br>

**解决办法：**

在pytorch中为算子添加符号函数, 并且注册到 onnx 对应的算子集中


<br>
<br>


## register_torchscripts_op.py
在pytorch原生算子无法实现一些运算时候，就需要自己定义一个 pytorch 算子，官方推荐是用 `TorchScript` 来[实现](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)。

<br>

在本次案例中，跳过了自定义 `TorchScript` 算子的步骤（很复杂，写完还要编译什么的），直接在定义好的 `TorchScript` 算子上添加了符号函数，并注册到 onnx 对应的算子集中。


<br>
<br>


## setup.py + my_add.cpp

自定义 C++ 算子

需要注意：
- [libtorch](https://pytorch.org/get-started/locally/) 按自己的系统环境安装，需要将自定义 C++ 算子编译为动态库，并拷贝到目标机器上
- 需要将动态库路径添加到环境变量中
比如下面配置文件:
```json
{
    "configurations": [
        {
            "name": "Mac",
            "includePath": [
                "${workspaceFolder}/**",
                "${workspaceFolder}/libtorch/include/torch/csrc/api/include",
                "/Users/solochan/anaconda3/envs/deploy_onnx/include/python3.9"
            ],
            "defines": [],
            "macFrameworkPath": [
                "/Library/Developer/CommandLineTools/SDKs/MacOSX12.sdk/System/Library/Frameworks"
            ],
            "compilerPath": "/opt/homebrew/opt/llvm/bin/clang++",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "macos-clang-arm64"
        }
    ],
    "version": 4
}
```
- 配置的 python 环境路径一定要装有 torch ！！

- 运行一定要用 `python setup.py develop`
