import onnx

"""
    onnx.load 函数用于读取一个 ONNX 模型。
    
    onnx.checker.check_model 用于检查模型格式是否正确，如果有错误的话该函数会直接报错
"""

onnx_model = onnx.load("srcnn.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")
