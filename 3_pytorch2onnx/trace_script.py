"""
    torch.onnx.export 可以通过两种方法得到 onnx 模型：
    - torch.jit.trace (无控制流, 默认)
    - torch.jit.script (有控制流)

    无论是 trace 还是 script 都会转为中间格式 torch.jit.ScriptModule, 再通过 torch.onnx.export 转化为 onnx 模型
"""

import torch

class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        for i in range(self.n):
            x = self.conv(x)
        return x



models = [Model(2), Model(3)]
model_names = ['model_2', 'model_3']

for model, model_name in zip(models, model_names):
    dummy_input = torch.rand(1, 3, 10, 10)
    dummy_output = model(dummy_input)
    model_trace = torch.jit.trace(model, dummy_input)
    model_script = torch.jit.script(model)

    # 跟踪法与直接 torch.onnx.export(model, ...)等价
    torch.onnx.export(model_trace, dummy_input, f'./3_pytorch2onnx/{model_name}_trace.onnx')
    # 脚本化必须先调用 torch.jit.sciprt
    torch.onnx.export(model_script, dummy_input, f'./3_pytorch2onnx/{model_name}_script.onnx')