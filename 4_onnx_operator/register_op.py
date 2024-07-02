import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)

from torch.onnx import register_custom_op_symbolic


# 自己定义一个 符号函数 (symbolic function)
# 符号函数用于将 PyTorch 的 op 转换为 ONNX 的 op
def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)

# 自定义的符号函数需要 注册
register_custom_op_symbolic('aten::asinh', asinh_symbolic, 9)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, input, '4_onnx_operator/asinh.onnx')

