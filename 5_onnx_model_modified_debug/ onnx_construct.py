import onnx
from onnx import helper
from onnx import TensorProto

"""
    构造描述张量信息的 ValueInfoProto 对象
    ValueInfoProto 对象包含两个属性：
        name: 字符串，表示张量的名字
        type: TensorProto，表示张量的类型信息
    构造方法：
        helper.make_value_info(name, type, shape)
"""

a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 10])


"""
    构造节点信息 NodeProto 对象
    NodeProto 对象包含以下属性：
        op_type: 字符串，表示节点的类型
        inputs: 字符串列表，表示节点的输入张量的名字
        outputs: 字符串列表，表示节点的输出张量的名字
        name: 字符串，表示节点的名字
        attribute: 键值对，表示节点的属性
    构造方法：
        helper.make_node(op_type, inputs, outputs, name, **attributes)
"""
# mul 输出名 ['c'] 与 add 输入名 ['c'] 同名，故两个点相连
mul = helper.make_node('Mul', ['a', 'x'], ['c'])
add = helper.make_node('Add', ['c', 'b'], ['output'])



"""
    构造描述计算图的 GraphProto 对象
    GraphProto 对象包含以下属性：
        name: 字符串，计算图的名字 (必须是拓扑结构！！)
        inputs: ValueInfoProto 列表，计算图中输入张量的信息
        outputs: ValueInfoProto 列表，计算图中输出张量的信息
        initializer: TensorProto 列表，计算图中常量张量的信息
        node: NodeProto 列表，计算图中节点的信息
    构造方法：
        helper.make_graph(nodes, name, inputs, outputs)
"""
graph = helper.make_graph([mul, add], 'linear_func', [a, x, b], [output])


# 把计算图 GraphProto 封装进 模型 ModelProto 对象中
model = helper.make_model(graph)


# 检查模型是否满足 onnx 标准
onnx.checker.check_model(model)
print(model)
onnx.save(model, '5_onnx_model_modified_debug/linear_func.onnx')