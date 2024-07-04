import onnx
model = onnx.load('linear_func.onnx')

node = model.graph.node
node[1].op_type = 'Sub'

onnx.checker.check_model(model)
onnx.save(model, 'linear_func_2.onnx')