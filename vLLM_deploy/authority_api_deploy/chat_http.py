import gradio as gr
import json
import requests
import vllm.entrypoints
import re


"""

本例使用 request 方法向 API 发送请求获取数据实现对话


下面以 qwen/Qwen-7B-Chat 模型为例，运行模型对话的步骤：

    先运行：（新建终端运行）
        python -m vllm.entrypoints.openai.api_server --model qwen/Qwen-7B-Chat --dtype auto --api-key token-abc123
    
    在运行：（另新建终端）
        python chat_http.py
    
"""


special_chars = ['<|im_start|>', '<|im_end|>', '<|endoftext|>']

def remove_special_characters(s, special_chars):
    pattern = '|'.join(re.escape(char) for char in special_chars)
    return re.sub(pattern, '', s)


def inference(prompt, history):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token-abc123"
    }

    data = {
        "model": "qwen/Qwen-7B-Chat",
        "messages": [{"role": "user", "content": prompt}]
    }

    response=requests.post("http://localhost:8000/v1/chat/completions",json=data,stream=True, headers=headers)

    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            print("====data:", data)
            output = data["choices"][0]["message"]["content"].rstrip('\r\n')
            output = remove_special_characters(output, special_chars)
            yield output

gr.ChatInterface(
    inference,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Chat with me!", container=False, scale=7),
    description="This is the demo for Gradio UI consuming TGI endpoint with huggingface🤗 model.",
    title="Qwen 🇨🇳 vLLM 🚀",
    examples=["你是谁?", "你能干什么？", "请你介绍下北京"],
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
).queue().launch(server_name="0.0.0.0", share=False)