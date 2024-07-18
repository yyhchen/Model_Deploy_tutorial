# Text-Generation Inference

使用 Huggingface 提供的推理框架Text-Generation-Inference（TGI）进行推理部署

[官方文档](https://huggingface.co/docs/text-generation-inference/index)

---

<br>
<br>

## 环境准备

**本地安装(Ubuntu 22.04 + cuda12.1)**

1. 先下载安装rust
```bash
#Installing Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. 下载protobuf协议安装包，当前目录直接下载代码并解压（做镜像装的，没用conda）
```bash
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

3. 下载 huggingface 官方提供的 text-generation inference 代码
```bash
git clone https://github.com/huggingface/text-generation-inference.git

## 进入目录
cd text-generation-inference
```

4. 编译安装 下载回来的源码（没意外这一步都会出问题）
  - 先装 `ninja`  `pip install ninja`（加速flash-atten 包的编译，因为要涉及到cuda文件编译。看编译过程是这样）

  - `BUILD_EXTENSIONS=False make install` (因为Makefile文件用都是一些 install，不懂的自行了解CMake )

  - 接下来没意外都是一堆错误

  - 安装过程耐心等待，**切勿**中断操作(crtl + c!)




<br>
<br>



## 1. 启动服务端

```bash
text-generation-launcher --model-id 自定义模型 --num-shard 1 --port 8080
```

1. [huggingface官方模型列表](https://huggingface.co/models) 理论上都支持，实际上以 TGI模块列出为准
2. [TGI部分列出支持的模型](https://huggingface.co/docs/text-generation-inference/supported_models)


<br>
<br>


## 2. 测试


### 2.1 `curl` 测试（简单测试模型是否连通）

```bash
!curl 0.0.0.0:8080/generate -X POST -d '{"inputs": "You are a helpful assistant.","stream": true,"max_tokens": 20}' -H 'Content-Type: application/json'
```

### 2.2 python 代码测试

需要先安装 `pip install text-generation` （huggingface提供的库, 本环境已经安装好）

```python
from text_generation import Client
client = Client("http://0.0.0.0:8080")
client.generate(prompt="hello. who are you")
```

<br>
<br>


## 3. gradio搭建 webui 启动推理服务

本环境路径下的 `web_ui.py` 文件

在**新建**终端运行 `python web_ui.py --model-url http://0.0.0.0:8080/generate --host 0.0.0.0 --port 7860` 即可





<br>
<br>
<br>


## 安装编译出现常见错误

1. **gcc版本问题（cuda 12.1 不支持 12 以后的gcc/g++）**

出现错误： #error -- unsupported GNU version! gcc versions later than 12 are not supported!
  
列几种解决办法（按号入座即可）：

- 更换gcc版本
    ```bash
    # 解决办法 （ubuntu安装软件请自觉运行 sudo apt update）
    sudo apt install gcc-12 g++-12 

    # 然后替换系统的编译器
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 10
    ```

- 好像强行用 13版本也行 (需要懂CMake这个可能忽略警告吧～)
    ```bash
    # 这个要在 CMakeLists.txt文件中强制加 -allow-unsupported-compiler
    # 例如
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    ```

- 用 llvm 编译也可以（不同编译器罢了，个人更喜欢用llvm）
    ```bash
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang)
    ```
  
2. Openssl 问题
  
出现错误：error: failed to run custom build command for `openssl-sys v0.9.102`
  
官方有提到 `sudo apt-get install libssl-dev` 但是还是会报相关错误
重新再安装一次就好了
  
这一步还可能是 缺少 `pkg-config` 包（是一个编译工具来的，看有没有相关错误吧～）
`sudo apt install pkg-config`