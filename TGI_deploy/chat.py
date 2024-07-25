import gradio as gr
from huggingface_hub import InferenceClient
from openai import OpenAI

"""
    流式对话：
    1. 同样先在终端启动 API:
        text-generation-launcher --model-id 自定义模型 --num-shard 1 --port 8080
    
    2. 直接运行 chat.py

"""

# init the client but point it to TGI
client = OpenAI(
    base_url="http://0.0.0.0:8080/v1",
    api_key="-"
)


def inference(prompt, history):
    chat_completion = client.chat.completions.create(
    model="tgi",
    max_tokens=384,
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": prompt}
    ],
    stream=True)
    
    partial_message = ""
    for token in chat_completion:
        token = token.choices[0].delta.content
        partial_message += token
        yield partial_message

gr.ChatInterface(
    inference,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Chat with me!", container=False, scale=7),
    description="This is the demo for Gradio UI consuming TGI endpoint with 🤗 model.",
    title="Chat 🇨🇳 TGI 🚀",
    examples=["who are you?", "what can you do?"],
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
).queue().launch(server_name='0.0.0.0',share=False)