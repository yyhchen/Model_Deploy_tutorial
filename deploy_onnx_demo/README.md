# 用 PyTorch 实现一个超分辨率模型，并把模型部署到 ONNX Runtime 这个推理引擎上


[OpenMMLab 提供的教程](https://mmdeploy.readthedocs.io/zh-cn/latest/tutorial/01_introduction_to_model_deployment.html)

---

- 创建 pytorch 环境 (二选一,看自己情况)
```bash
# cpu only
conda install pytorch torchvision cpuonly -c pytorch

# gpu cuda 11.8
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

- 三方库 (onnxruntime, onnx, opencv)
```bash
pip install onnxruntime onnx opencv-python
```

<br>
<br>


- srn_model.py: 实现模型加载，onnx格式转换，最后使用 onnxruntime 进行部署推理

<br>

- check_onnx_model.py: 检查onnx格式是否转换正确