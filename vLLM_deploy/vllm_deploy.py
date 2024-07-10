from vllm_wrapper import vLLMWrapper

model = "qwen/Qwen-1_8B-Chat"

vllm_model = vLLMWrapper(model,
                            quantization = 'fp8',
                            dtype="float16",
                            tensor_parallel_size=1,
                            gpu_memory_utilization=0.8)

history=None 
while True:
    Q=input('提问:')
    response, history = vllm_model.chat(query=Q,
                                        history=history)
    print(response)
    history=history[:20]