# TensorRT 模型构建与推理

[原教程](https://mmdeploy.readthedocs.io/zh-cn/latest/tutorial/06_introduction_to_tensorrt.html)

---

## 前置条件 CUDA+cuDNN+TensorRT
(以下是我的实验环境，按需自己配置)
- CUDA 11.8
- cuDNN 8700
- TensorRT 10.2

## 补充一些TensorRT的安装配置（Win11）
[TensorRT 下载](https://developer.nvidia.com/tensorrt/download/10x)

[参考安装教程](https://blog.csdn.net/weixin_43134049/article/details/124752259)

安装步骤：
1. 下载后解压到自己想要存放的路径，建议和CUDA Toolkits在同一个并行路径，这样方便找
2. 配置TensorRT的环境变量(有可能因为手法原因导致无效)
3. 我是直接将 TensorRT 的 `include` 文件夹下的所有 `.h` 文件拷贝到 CUDA 的 `include` 文件夹下，TensorRT 的 `lib` 文件夹下的所有 `.dll` 文件拷贝到 CUDA 的 `bin` 文件夹下，所有 `.lib` 文件 拷贝到 `bin/x64` 文件下
4. 然后在对用的 anaconda 环境 下，使用 `pip install "你的TensorRT路径\python\tensorrt-10.2.0-cp38-none-win_amd64.whl"` 安装 TensorRT 的 python 包
5. 最后安装 `pip install cuda-python`，没有意外的话就成功了

<br>

### 测试TensorRT是否成功
方法一：

```python
import tensorrt as trt
if __name__ == "__main__":
    print(trt.__version__)
    print("hello trt!!")
```
方法二：
在 TensorRT 的路径下 `TensorRT-10.2.0.19\samples\python\network_api_pytorch_mnist` 使用 `python sample.py` ，成功了会有推理的 epoch显示


<br>
<br>
<br>


## 1. TensorRT 简介

TensorRT 是 NVIDIA 提供的一个高性能深度学习推理优化器和运行时库，用于在 NVIDIA GPU 上部署深度学习模型。TensorRT 通过对模型进行优化，可以显著提高模型的推理速度和效率。


<br>
<br>
<br>


## 2. 模型构建

**使用 TensorRT 生成模型主要有两种方式：**

1. 直接通过 TensorRT 的 API 逐层搭建网络；

2. 将中间表示的模型转换成 TensorRT 的模型，比如将 ONNX 模型转换成 TensorRT 模型。

<br>
<br>

### 2.1 直接构建

Python API 与 [原教程](https://mmdeploy.readthedocs.io/zh-cn/latest/tutorial/06_introduction_to_tensorrt.html) 区别:
- 修改了 `network.add_pooling()` --> `network.add_pooling_nd()`
- 修改了 `pool.stride` --> `pool.stride_nd`
- 修改了 `config.max_workspace_size` --> `config.set_memory_pool_limit()`
- 修改了 `build_engine()` --> `build_serialized_network()`
- 以上在 TensorRT 10.2 版本均报错 no Attribute


<br>
<br>

C++ API 构建 与 原教程区别 自己看代码吧（心累）
- 用 cmake 构建的，因为用 g++ 编译一直正在运行 (并且没找到问题在哪，但是用cmake编译时，发现.lib库有路径问题，linux 是 .so)
- cmake 详情请看我的 `CMakeLists.txt` 文件
```cmake
cmake_minimum_required(VERSION 3.10)

# 项目名称和语言
project(MyTensorRTProject LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含路径
include_directories(
    ${CMAKE_SOURCE_DIR}
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include"
    "E:/Program toolkit/TensorRT-10.2.0.19/include"
    "E:/Program toolkit/TensorRT-10.2.0.19/samples/common"
)

# 查找 CUDA 包
find_package(CUDA REQUIRED)

# 查找 TensorRT 包
find_library(NVINFER nvinfer_10.lib HINTS "E:/Program toolkit/TensorRT-10.2.0.19/lib")
find_library(NVONNXPARSER nvonnxparser_10.lib HINTS "E:/Program toolkit/TensorRT-10.2.0.19/lib")

# 如果没有找到，打印错误消息
if(NOT NVINFER)
    message(FATAL_ERROR "Could not find the NVINFER library!")
endif()
if(NOT NVONNXPARSER)
    message(FATAL_ERROR "Could not find the NVONNXPARSER library!")
endif()

# 添加可执行文件
add_executable(MyTensorRTExecutable cpp_api.cpp)

# 链接库
target_link_libraries(MyTensorRTExecutable ${CUDA_LIBRARIES} ${NVINFER} ${NVONNXPARSER})

# 设置编译选项
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(MyTensorRTExecutable PRIVATE /W3 /WX)
    target_compile_definitions(MyTensorRTExecutable PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(MyTensorRTExecutable PRIVATE -Wall -Wextra -Werror)
endif()
```


<br>
<br>


### 2.2 IR 转换模型

python API转换
- 跟直接构建一样，需要改一些东西，具体看代码吧，这里就不一一列举了

<br>
<br>

C++ API 转换
- 也是跟直接构建一些，需要改一些东西(CMakeLists.txt 也需要改!)

