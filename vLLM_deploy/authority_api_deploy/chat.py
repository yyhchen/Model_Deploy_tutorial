from openai import OpenAI
import gradio as gr
import re

"""
    运行前，请先启动服务端：
    
        python -m vllm.entrypoints.openai.api_server --model qwen/Qwen-7B-Chat --dtype auto --api-key token-abc123 --trust-remote-code
"""

# init the client
client = OpenAI(
    base_url="http://0.0.0.0:8000/v1",
    api_key="token-abc123"
)


special_chars = ['<|im_start|>', '<|im_end|>', '<|endoftext|>']

def remove_special_characters(s, special_chars):
    pattern = '|'.join(re.escape(char) for char in special_chars)
    return re.sub(pattern, '', s)


def inference(prompt, history):
    chat_completion = client.chat.completions.create(
        model="qwen/Qwen-7B-Chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant." },
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    partial_message = ""

    for chunk in chat_completion:
        for choice in chunk.choices:
            content = choice.delta.content
            if content:
                content = remove_special_characters(content, special_chars)
                partial_message += content
                yield partial_message


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