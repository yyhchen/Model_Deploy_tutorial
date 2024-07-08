# 使用 TensorRT模型 做推理

前置条件：
TensorRT 10.2 版本！！

---


## python_inference.py

[TensorRT Python API 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Engine.html), 想知道具体的函数内容，可以参考官方文档。

代码部分详解：

- 序列化读取模型，然后通过 `self.engine = runtime.deserialize_cuda_engine(engine_bytes)` 模型反序列化后 是一个 `ICudaEngine` 对象

- 接下来的 `self.context = self.engine.create_execution_context()` 是一个 `IExecutionContext` 对象，这个对象是用于执行推理的上下文，可以理解为模型执行的上下文环境，通过这个对象可以执行推理操作。

- `self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]` 这一步将 `names` 变为类对象，是为了后续 绑定 输入输出的 index.

<br>
<br>



使用python API 编写代码，根据之前构建的 `model.engine` 进行推理，总结出一些问题:

1. 使用 `execute_async_v3` 如果没有绑定输入输出的 地址，大概率会出现如下错误：
```shell
[TRT] [E] IExecutionContext::enqueueV3: Error Code 3: API Usage Error (Parameter check failed, condition: mContext.profileObliviousBindings.at(profileObliviousIndex) != nullptr. Address is not set for input tensor input. Call setInputTensorAddress or setTensorAddress before enqueue/execute.)
```


2. 并且使用 `execute_async_v3` 不能用 `orch.cuda.current_stream().cuda_stream`，会报错：
```shell
[TRT] [W] Using default stream in enqueueV3() may lead to performance issues due to additional calls to cudaStreamSynchronize() by TensorRT to ensure correct synchronization. Please use non-default stream instead.
```

<br>

解决办法：

1. 使用 `execute_v2()`, 输入 `bindings` 即可，这里的 `bindings` 是 输入输出的地址，前面需要用 `.data_ptr()` 进行绑定


2. 使用 `execute_async_v3()`，需要使用 `set_tensor_address()` 绑定输入输出的地址,并且使用 `torch.cuda.Stream().cuda_stream`，这样就不会报错了。（execute_async_v2已经删除）


<br>
<br>
<br>



## cpp_inference.cpp

[TensorRT C++ API 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_runtime.html#a3e67c882843e066ab9087cef845dfed4), 想知道具体的函数内容，可以参考官方文档。

使用 C++ API 进行推理，和python API 的使用方式基本一致，只是语言不同，具体可以参考官方文档。


因为本次实验的版本是 TensorRT 10.2 , 移除了 `getBindingIndex` 方法, 这样找 index 就变得很麻烦.

下面是我重写的一个方法：

```cpp
int inputIndex = -1;
        int outputIndex = -1;
        for (int i = 0; i < engine.getNbIOTensors(); ++i)
        {
            const char* tensorName = engine.getIOTensorName(i);
            if (strcmp(tensorName, IN_NAME) == 0)
            {
                inputIndex = i;
            }
            else if (strcmp(tensorName, OUT_NAME) == 0)
            {
                outputIndex = i;
            }
        }
        assert(inputIndex != -1 && outputIndex != -1);
```


平替之前的:

```cpp
const int inputIndex = engine.getBindingIndex(IN_NAME);
const int outputIndex = engine.getBindingIndex(OUT_NAME);
```


<br>
<br>


还遇到了一个跟之前 python api 一样的问题：
```shell
[E] [TRT] IExecutionContext::enqueueV3: Error Code 3: API Usage Error (Parameter check failed, condition: mContext.profileObliviousBindings.at(profileObliviousIndex) != nullptr. Address is not set for input tensor input. Call setInputTensorAddress or setTensorAddress before enqueue/execute.)
```


<br>


老规矩，还是因为 在调用 `enqueueV3` 进行推理之前，**输入张量的内存地址尚未被正确设置**。在之前的代码中，创建了 GPU 缓冲区，但却没有完成绑定

<br>

解决办法：

```cpp
// 将输入和输出张量的地址绑定到缓冲区
context->setTensorAddress(inputIndex, inputTensor.data_ptr());
context->setTensorAddress(outputIndex, outputTensor.data_ptr());
```


<br>

代码补充位置如下所示：

```cpp
void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    // ... 其他代码保持不变 ...

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float)));

    // Bind the buffers to the execution context
    context.setTensorAddress(engine.getIOTensorName(inputIndex), buffers[inputIndex]);
    context.setTensorAddress(engine.getIOTensorName(outputIndex), buffers[outputIndex]);

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // ... 其余代码保持不变 ...
}
```