from typing import Union, Optional, Sequence, Dict, Any

import torch
import tensorrt as trt

"""
    使用 execute_v2(bindings=bindings) 是没有问题的，使用 execute_async_v3(torch.cuda.current_stream().cuda_stream)) 还是不行
"""

class TRTWrapper(torch.nn.Module):
    def __init__(self, engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        print('slef.context: ', self.context)
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        print("self.name: ", self.names)
        input_names = [name for i, name in enumerate(self.names) if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        print('input_names: ', input_names)
        self._input_names = input_names
        print('out_put_names: ', output_names)
        self._output_names = output_names

        if self._output_names is None:
            output_names = [name for name in self.names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_tensor_profile_shape(input_name, profile_id)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.names.index(input_name)
            print('idx: ', idx, type(idx))
            print('input_name: ', input_name, type(input_name))
            print('profile_id: ', profile_id, type(profile_id))
            print('profile: ', profile, type(profile))

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_input_shape(input_name, tuple(input_tensor.shape))
            self.context.set_tensor_address(input_name, input_tensor.data_ptr())
            bindings[idx] = input_tensor.contiguous().data_ptr()
            print('bindings[idx]: ', bindings[idx])

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.names.index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_tensor_shape(output_name))

            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            self.context.set_tensor_address(output_name, output.data_ptr()) # 让类的函数绑定地址，execute_async_v3() 需要, 输入是cuda数据流
            bindings[idx] = output.data_ptr()   # 手动绑定地址， execute_v2() 需要，因为输入是 bindings
            print('bindings[idx]: ', bindings[idx], type(bindings[idx]))
        print(bindings, type(bindings))
        print(torch.cuda.current_stream().cuda_stream, type(torch.cuda.current_stream().cuda_stream))
        # self.context.execute_v2(bindings=bindings)
        # self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        self.context.execute_async_v3(torch.cuda.Stream().cuda_stream)
        return outputs

model = TRTWrapper(r'..\IR_convert\model.engine', ['output'])
output = model(dict(input=torch.randn(1, 3, 224, 224).cuda()))
print(output)


"""
    下面这个是 bindings 没有绑定好的，output上有问题，留下是为了后续学习，对比区别
"""

# from typing import Union, Optional, Sequence, Dict, Any

# import torch
# import tensorrt as trt

# class TRTWrapper(torch.nn.Module):
#     def __init__(self, engine: Union[str, trt.ICudaEngine],
#                  output_names: Optional[Sequence[str]] = None) -> None:
#         super().__init__()
#         self.engine = engine
#         if isinstance(self.engine, str):
#             with trt.Logger() as logger, trt.Runtime(logger) as runtime:
#                 with open(self.engine, mode='rb') as f:
#                     engine_bytes = f.read()
#                 self.engine = runtime.deserialize_cuda_engine(engine_bytes)
#         self.context = self.engine.create_execution_context()
#         names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
#         print("name: ", names)
#         input_names = [name for i, name in enumerate(names) if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
#         print('input_names: ', input_names)
#         self._input_names = input_names
#         print('out_put_names: ', output_names)
#         self._output_names = output_names

#         if self._output_names is None:
#             output_names = [name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
#             self._output_names = output_names

#     def forward(self, inputs: Dict[str, torch.Tensor]):
#         assert self._input_names is not None
#         assert self._output_names is not None
#         bindings = [None] * (len(self._input_names) + len(self._output_names))
#         profile_id = 0
#         for input_name, input_tensor in inputs.items():
#             # check if input shape is valid
#             profile = self.engine.get_tensor_profile_shape(input_name, profile_id)
#             assert input_tensor.dim() == len(
#                 profile[0]), 'Input dim is different from engine profile.'
#             for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
#                                              profile[2]):
#                 assert s_min <= s_input <= s_max, \
#                     'Input shape should be between ' \
#                     + f'{profile[0]} and {profile[2]}' \
#                     + f' but get {tuple(input_tensor.shape)}.'
#             idx = self.engine.get_tensor_name(profile_id)
#             print('idx: ', idx, type(idx))
#             print('input_name: ', input_name, type(input_name))
#             print('profile_id: ', profile_id, type(profile_id))
#             print('profile: ', profile, type(profile))

#             # All input tensors must be gpu variables
#             assert 'cuda' in input_tensor.device.type
#             input_tensor = input_tensor.contiguous()
#             if input_tensor.dtype == torch.long:
#                 input_tensor = input_tensor.int()
#             self.context.set_input_shape(idx, tuple(input_tensor.shape))
#             # print(input_tensor, type(input_tensor))
#             bindings[profile_id] = input_tensor.contiguous().data_ptr()

#         # create output tensors
#         outputs = {}
#         for output_name in self._output_names:
#             idx = self.engine.get_tensor_name(profile_id)
#             dtype = torch.float32
#             shape = tuple(self.context.get_tensor_shape(idx))

#             device = torch.device('cuda')
#             output = torch.empty(size=shape, dtype=dtype, device=device)
#             outputs[output_name] = output
#             bindings[profile_id] = output.data_ptr()
#         print(bindings, type(bindings))
#         print(torch.cuda.current_stream().cuda_stream, type(torch.cuda.current_stream().cuda_stream))
#         # self.context.execute_v2(bindings)
#         self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
#         return outputs

# model = TRTWrapper(r'../model_construct/model.engine', ['output'])
# output = model(dict(input=torch.randn(1, 3, 224, 224).cuda()))
# # print(output)