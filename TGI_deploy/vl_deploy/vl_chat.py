import gradio as gr
from huggingface_hub import InferenceClient
import base64
import io
import spaces
import re

client = InferenceClient(base_url="http://0.0.0.0:8081")

# 去除一些特殊token，如停止词等
special_chars = ['<end_of_utterance>']
def remove_special_characters(s, special_chars):
    pattern = '|'.join(re.escape(char) for char in special_chars)
    return re.sub(pattern, '', s)

@spaces.GPU(duration=180)
def model_inference(
    image, text, decoding_strategy, temperature,
    max_new_tokens, repetition_penalty, top_p
):
    if text == "" and not image:
        gr.Error("Please input a query and optionally image(s).")

    if text == "" and image:
        gr.Error("Please input a text query along the image(s).")

    # 将图像编码为base64(关键)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_data_uri = f"data:image/png;base64,{image_base64}"

    # 构建prompt
    prompt = f"![]({image_data_uri}){text}\n\n"

    # 设置生成参数
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }

    assert decoding_strategy in [
        "Greedy",
        "Top P Sampling",
    ]
    if decoding_strategy == "Greedy":
        generation_args["do_sample"] = False
    elif decoding_strategy == "Top P Sampling":
        generation_args["temperature"] = temperature
        generation_args["do_sample"] = True
        generation_args["top_p"] = top_p

    try:
        # 发送文本生成请求
        response = client.text_generation(prompt, **generation_args, stream=True)
        
        # 处理流式响应
        generated_text = ""
        for token in response:
            generated_text += remove_special_characters(token, special_chars)
            print(token, end="", flush=True)
        
        return generated_text
    except Exception as e:
        print(f"错误: {e}")
        return "生成文本时发生错误，请重试。"


with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("## IDEFICS2 Instruction 🐶")
    gr.Markdown("Play with [IDEFICS2-8B](https://huggingface.co/HuggingFaceM4/idefics2-8b) in this demo. To get started, upload an image and text or try one of the examples.")
    gr.Markdown("**Important note**: This model is not made for chatting, the chatty IDEFICS2 will be released in the upcoming days. **This model is very strong on various tasks, including visual question answering, document retrieval and more, you can see it through the examples.**")
    gr.Markdown("Learn more about IDEFICS2 in this [blog post](https://huggingface.co/blog/idefics2).")

    with gr.Column():
        image_input = gr.Image(label="Upload your Image", type="pil")
        query_input = gr.Textbox(label="Prompt")
        submit_btn = gr.Button("Submit")
        output = gr.Textbox(label="Output")

    with gr.Accordion(label="Example Inputs and Advanced Generation Parameters"):
        examples=[["./example_images/docvqa_example.png", "How many items are sold?", "Greedy", 0.4, 512, 1.2, 0.8],
                    ["./example_images/example_images_travel_tips.jpg", "I want to go somewhere similar to the one in the photo. Give me destinations and travel tips.", "Greedy", 0.4, 512, 1.2, 0.8],
                    ["./example_images/baklava.png", "Where is this pastry from?", "Greedy", 0.4, 512, 1.2, 0.8],
                    ["./example_images/dummy_pdf.png", "How much percent is the order status?", "Greedy", 0.4, 512, 1.2, 0.8],
                    ["./example_images/art_critic.png", "As an art critic AI assistant, could you describe this painting in details and make a thorough critic?.", "Greedy", 0.4, 512, 1.2, 0.8],
                    ["./example_images/s2w_example.png", "What is this UI about?", "Greedy", 0.4, 512, 1.2, 0.8]]

        # Hyper-parameters for generation
        max_new_tokens = gr.Slider(
              minimum=8,
              maximum=1024,
              value=512,
              step=1,
              interactive=True,
              label="Maximum number of new tokens to generate",
          )
        repetition_penalty = gr.Slider(
              minimum=0.01,
              maximum=5.0,
              value=1.2,
              step=0.01,
              interactive=True,
              label="Repetition penalty",
              info="1.0 is equivalent to no penalty",
          )
        temperature = gr.Slider(
              minimum=0.0,
              maximum=5.0,
              value=0.4,
              step=0.1,
              interactive=True,
              label="Sampling temperature",
              info="Higher values will produce more diverse outputs.",
          )
        top_p = gr.Slider(
              minimum=0.01,
              maximum=0.99,
              value=0.8,
              step=0.01,
              interactive=True,
              label="Top P",
              info="Higher values is equivalent to sampling more low-probability tokens.",
          )
        decoding_strategy = gr.Radio(
              [
                  "Greedy",
                  "Top P Sampling",
              ],
              value="Greedy",
              label="Decoding strategy",
              interactive=True,
              info="Higher values is equivalent to sampling more low-probability tokens.",
          )
        decoding_strategy.change(
              fn=lambda selection: gr.Slider(
                  visible=(
                      selection in ["contrastive_sampling", "beam_sampling", "Top P Sampling", "sampling_top_k"]
                  )
              ),
              inputs=decoding_strategy,
              outputs=temperature,
          )

        decoding_strategy.change(
              fn=lambda selection: gr.Slider(
                  visible=(
                      selection in ["contrastive_sampling", "beam_sampling", "Top P Sampling", "sampling_top_k"]
                  )
              ),
              inputs=decoding_strategy,
              outputs=repetition_penalty,
          )
        decoding_strategy.change(
              fn=lambda selection: gr.Slider(visible=(selection in ["Top P Sampling"])),
              inputs=decoding_strategy,
              outputs=top_p,
          )
        gr.Examples(
                        examples = examples,
                        inputs=[image_input, query_input, decoding_strategy, temperature,
                                                              max_new_tokens, repetition_penalty, top_p],
                        outputs=output,
                        fn=model_inference
                    )

        submit_btn.click(model_inference, inputs = [image_input, query_input, decoding_strategy, temperature,
                                                      max_new_tokens, repetition_penalty, top_p], outputs=output)


demo.launch(server_name="0.0.0.0", debug=True)