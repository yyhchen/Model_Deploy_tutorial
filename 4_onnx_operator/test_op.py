import onnxruntime
import torch
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch_output = model(input).detach().numpy()
print(torch_output)

sess = onnxruntime.InferenceSession('4_onnx_operator/asinh.onnx')
input_name = sess.get_inputs()[0].name
print(sess.get_inputs()[0])
print(input_name)
ort_output = sess.run(None, {input_name: input.numpy()})[0]
print(ort_output)

assert np.allclose(torch_output, ort_output)