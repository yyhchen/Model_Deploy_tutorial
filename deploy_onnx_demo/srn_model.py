import os
import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn
import onnxruntime

"""
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    运行本文件产生了5个文件: face.png、srcnn.pth、face_torch.png、srcnn.onnx、face_ort.png
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    face.png: 下载的原图
    srcnn.pt: 下载的超分辨模型 (为了方便跳过了自己训练，直接下载模型参数)
    
    face_torch.png: 将 face.png 超分辨后得到的图
    srcnn.onnx: 将model转化为的中间表示
    
    face_ort.png: 通过 onnxruntime部署的模型推理face.png得到
    
"""


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4)
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


# Download checkpoint and test image
urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
        'https://raw.githubusercontent.com/open-mmlab/mmagic/master/tests/data/face/000001.png']
names = ['srcnn.pth', 'face.png']
for url, name in zip(urls, names):
    if not os.path.exists(name):
        open(name, 'wb').write(requests.get(url).content)


def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)

    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint(由于 MMagic 中 SRCNN 的权重结构与这里定义的不一样，故修改权重字典的key来适配这里的模型)
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


model = init_torch_model()
input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img)).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("face_torch.png", torch_output)


"""
    ##########################################
    以下代码把 PyTorch 的模型转换成 ONNX 格式的模型 
    ##########################################
    
    这一步产生 srcnn.onnx
    
    torch.onnx.export 是 PyTorch 自带的把模型转换成 ONNX 格式的函数
    
    !!!前三个必选参数!!!：前三个参数分别是要转换的模型、模型的任意一组输入、导出的 ONNX 文件的文件名
    
    ##########################################
    为什么要指定一组输入？
    ##########################################
    
    ONNX 记录不考虑控制流的静态图。
    因此，PyTorch 提供了一种叫做追踪（trace）的模型转换方法：给定一组输入，再实际执行一遍模型，
    即把这组输入对应的计算图记录下来，保存为 ONNX 格式
"""
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "srcnn.onnx",
        opset_version=11,   # 算子集版本
        input_names=['input'],  # 指定输入 tensor 名称
        output_names=['output'])


"""
    ##################################################
    通过 .onnx 文件 完成模型部署, 然后通过部署后的模型进行推理
    ##################################################
"""

ort_session = onnxruntime.InferenceSession("srcnn.onnx")    # 这一步其实就已经部署完了
ort_inputs = {'input': input_img}
ort_output = ort_session.run(['output'], ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_ort.png", ort_output)
