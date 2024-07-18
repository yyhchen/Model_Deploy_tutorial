import argparse
import json
import gradio as gr
import requests
import vllm.entrypoints

"""
    wenb_ui.py:
        启动一个 gradio界面 访问 text-generation-inference 中的API

    打开两个不同的终端窗口，分别运行如下：
    
    先运行：(服务端)
        text-generation-launcher --model-id microsoft/phi-1_5 --num-shard 1 --port 8080
    
    在运行：（客户端）

        1. 通用格式模型
        python web_ui.py --model-url http://0.0.0.0:8080/generate --host 0.0.0.0 --port 7860

        or

        2. 带Instruct格式模型
        python web_ui.py --model-url http://0.0.0.0:8080/v1/chat/completions --host 0.0.0.0 --port 7860
    
"""

def vllm_bot(prompt):
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "inputs":prompt
    }
    
    response = requests.post(f"{args.model_url}", headers=headers, json=data, stream=True)

    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            print("====data:", data)
            output = data['generated_text']
            yield output

def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# vLLM Chat Completion Demo")
        inputbox = gr.Textbox(label="Input", placeholder="Enter your message")
        outputbox = gr.Textbox(label="Output", placeholder="Model's response")
        inputbox.submit(vllm_bot, [inputbox], [outputbox])
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model-url", type=str, default="http://0.0.0.0:8080/generate")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=True)