# onnx 模型的修改与调试

[原教程](https://mmdeploy.readthedocs.io/zh-cn/latest/tutorial/05_onnx_model_editing.html)

.py/.ipynb 顺序

1. **onnx_construct.py**: 构建一个简单的 onnx 模型
2. **test_model.py**:  测试之前构建的 onnx 模型 是否正确
3. **onnx_modified.py**:  简单修改 onnx 模型中某个节点的功能
4. **define_model_debug.ipynb**: 从模型本身的结构进行修改

---

围绕 ONNX 这一套神经网络定义标准本身，探究 ONNX 模型的构造、读取、子模型提取、调试。

<br>
<br>

## onnx底层实现
- onnx 存储格式
- onnx 的结构定义

<br>

### onnx 存储格式
onnx 底层是 Protobuf（Protocol Buffer）实现, 是一套表示和序列化数据的机制。分两步实现，先定义数据类型，再根据定义将数据存进一份二进制文件。如：

```protobuf
message Person {
  required string name = 1;
  required int32 id = 2;
  optional string email = 3;
}
```

`required` 字段表示必须的参数，`optional` 字段表示可选择。

**直接用 Protobuf 实现很复杂，但是 onnx 提供了API用与构造和读取 onnx 模型**


<br>
<br>


### onnx 的结构定义

可能类似如下结构：

- ModelProto
    - GraphProto
        - NodeProto
        - ValueInfoProto


<br>
<br>




## 读取 onnx 模型
- 构造 onnx 模型
- 修改 onnx 模型

<br>

### 构造 onnx 模型

用 `helper.make_tensor_value_info` 构造出一个描述张量信息的 `ValueInfoProto` 对象。 

**在 ONNX 中，不管是输入张量还是输出张量，它们的表示方式都是一样的。**

根绝之前的[结构](#onnx-的结构定义) 由底向上构建即可～ 

详细可看 [onnx_construct.py](5_onnx_model_modified_debug/onnx_construct.py) 代码。

<br>
<br>


### 修改 onnx 模型
可以通过访问节点修改它们的值来达到 修改 onnx 模型, 改完之后可以通过 推理过程验证是否正确～

<br>
<br>
<br>

## 调试 onnx 模型
- 子模型提取
- 输出 onnx 中间节点的值

<br>


### 子模型提取
模型部署出了问题，可以通过 `onnx.utils.extract_model` 提取子模型进行调试。

具体用法：

```python
onnx.utils.extract_model('input_path', 'output_path', 'input_node_name', 'output_node_name')
```

核心方法就是, 先通过 `onnx.load()` 加载模型, 然后 `print(model.graph)` 查看模型结构, 然后根据结构提取子模型。

**节点顺序遵循从上到下，从左到右这样的序号。** 提取的时候就明白这句话的意思了。

<br>

### 输出 onnx 中间节点的值

就是添加一些中间节点的输出，方便我们调试模型的子模块。

