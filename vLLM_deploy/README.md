# vLLM 

---

## 官方推荐安装方式(Linux)

```bash
# (Recommended) Create a new conda environment.
conda create -n myenv python=3.10 -y
conda activate myenv

# Install vLLM with CUDA 12.1.
pip install vllm
```

如果 CUDA 版本是 11.8
```bash
# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.4.0
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

<br>
<br>

## 模型下载

使用 modelscope 中的 snapshot_download 函数下载模型，第一个参数为模型名称，参数 cache_dir 为模型的下载路径。

先安装依赖
```bash
pip install modelscope
```
下载模型
```bash
from modelscope import snapshot_download

# 自动下载模型时，指定使用modelscope。不设置的话，会从 huggingface 下载
os.environ['VLLM_USE_MODELSCOPE']='True'
snapshot_download('qwen/Qwen-1_8B-Chat', cache_dir='自己取一个位置')
```



<br>
<br>


## 官方推荐简单部署 (offline)
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json


def get_completion(prompts, model, tokenizer=None, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len)
    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":    
    # 初始化 vLLM 推理引擎
    model='刚刚下载哪里指定的缓存位置' # 指定模型路径 or 模型名称
    # model = 'qwen/Qwen-1_8B-Chat'

    tokenizer = None
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False) # 加载分词器后传入vLLM 模型，但不是必要的。
    
    text = ["你是谁？",
           "你能做什么。"]
   

    outputs = get_completion(text, model, tokenizer=tokenizer, max_tokens=512, temperature=1, top_p=1, max_model_len=2048)

    # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        
```


<br>
<br>



## 参考 Qwen 官方的vLLM部署 (offline)

先准备一个 `vllm_wrapper.py` 文件，定一个文件 `prompt_utils.py` 自定义 `Qwen` 的 Prompt 格式, 再运行`vllm_deploy.py` 进行连续对话


<br>
<br>

### vllm_wrapper.py 要点

- `__init__` 方法中，封装了 vLLM 部署的一些步骤：
    - 模型下载 `snapshot_download()`
    - `tokenizer` 和 `generation_config` 是为了找到 推理停止的 token_id
    - **核心** `LLM` 类的初始化，`LLM` 类是 vLLM 的核心，封装了推理的细节（具体怎么配置可以看源码，非常清晰）。
<br>

- `chat` 方法，是能够连续对话的关键：
    - `history` 变量必须是 深拷贝的，防止篡改上下文信息
    - `_build_prompt` 方法 是构造 `Qwen` 的 Prompt 格式 (后面讲)
    - `SamplingParams` 是另一个**核心**, 封装了生成文本的参数，比如 `temperature` 和 `top_p`，具体可以看源码。
    - `llm.generate` 有非常多个，具体情况具体使用
    - 后面的都比较简单了


<br>
<br>


### prompt_utils.py 要点


- `——build_prompt` 方法，是构造 `Qwen` 的 Prompt 格式用于对话。
    - `im_start` 和 `im_end`，用于包裹对话内容。
    - `nl_token` 用于对话换行
    - `_tokenize_str` 函数，用于将角色和内容编码为模型可以理解的格式

<br>

- `remove_stop_words` 函数负责移除生成文本末尾的停用词



<br>
<br>
<br>



## API方式部署 （Online / Offline）

- [Fastchat](https://github.com/lm-sys/FastChat#other-models)
- [vLLM官方实现 (FastAPI + OpenAI规范)](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)


<br>
<br>


### Fastchat （Online）

对 `transformers` 库有版本要求，源码跟模型配置和加载相关用的是 `transformers`. 关于 Fastchat 离线推理这里不涉及，请自己查看官方文档。

若部署的是自己的模型，请参考 [model_support 部分](https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md#supported-models)

<br>

**安装：**
```bash
pip3 install "fschat[model_worker,webui]"
```

<br>

**启动步骤：**

先启动一个controller

```bash
python3 -m fastchat.serve.controller
```


然后启动model worker读取模型

```bash
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
```


(可选择步骤) 发送一个请求测试API,会得到一个输出
```bash
python3 -m fastchat.serve.test_message --model-name vicuna-7b-v1.5
```

启动 Gradio 界面
```bash
python3 -m fastchat.serve.gradio_web_server
```


<br>
<br>


## vLLM官方实现 (FastAPI + OpenAI规范) （Online）

使用 类OpenAI 的API，注意要加上 `api-key`

<br>

**启动步骤：**

首先启动API服务

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen-7B-Chat --dtype auto --api-key token-abc123
```

<br>

(可选择步骤)发送请求测试API

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/completions -H "Authorization: Bearer token-abc123" -H "Content-Type: application/json" -d '{"model":"qwen/Qwen-7B-Chat","prompt": "你好！","max_tokens": 7,"temperature": 0}'
```

<br>

启动 Gradio 部分得根据情况修改

[官方demo参考](https://docs.vllm.ai/en/latest/getting_started/examples/gradio_webserver.html)



<br>
<br>
<br>


## 部署过程中可能会遇到的问题
- 不同模型可能会遇到不同的 `prompt` 模板，详情需要参考具体模型（说明并不是一次部署并不是可以完全适用）

- ...