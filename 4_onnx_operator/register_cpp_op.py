import torch
import my_lib

# 封装自定义的C++函数（my_lib.my_add）
class MyAddFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        return my_lib.my_add(a, b)

    @staticmethod
    def symbolic(g, a, b):
        two = g.op("Constant", value_t=torch.tensor([2]))
        a = g.op('Mul', a, two)
        return g.op('Add', a, b)


## 调用算子
my_add = MyAddFunction.apply

class MyAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return my_add(a, b)


## 测试算子
model = MyAdd()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, (input, input), '4_onnx_operator/my_add.onnx')
torch_output = model(input, input).detach().numpy()
# print(torch_output)

import onnxruntime
import numpy as np
sess = onnxruntime.InferenceSession('4_onnx_operator/my_add.onnx')
inputname1 = sess.get_inputs()[0].name  # 获取输入名
print(inputname1)
inputname2 = sess.get_inputs()[1].name
print(inputname2)


# ort_output = sess.run(None, {inputname1: input.numpy(), inputname2: input.numpy()})[0]    # 等价
ort_output = sess.run(None, {'a.1': input.numpy(), 'b.1': input.numpy()})[0]
# print(ort_output)

assert np.allclose(torch_output, ort_output)
