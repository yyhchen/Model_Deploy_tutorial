import tensorrt as trt

verbose = True
IN_NAME = 'input'
OUT_NAME = 'output'
IN_H = 224
IN_W = 224
BATCH_SIZE = 1

EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config(
) as config, builder.create_network(EXPLICIT_BATCH) as network:
    # define network
    input_tensor = network.add_input(
        name=IN_NAME, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H, IN_W))
    pool = network.add_pooling_nd(
        input=input_tensor, type=trt.PoolingType.MAX, window_size=(2, 2))
    pool.stride_nd = (2, 2)
    pool.get_output(0).name = OUT_NAME
    network.mark_output(pool.get_output(0))

    # serialize the model to engine file
    profile = builder.create_optimization_profile()
    profile.set_shape(IN_NAME, (BATCH_SIZE, 3, IN_H, IN_W), (BATCH_SIZE, 3, IN_H, IN_W), (BATCH_SIZE, 3, IN_H, IN_W))
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    serialized_engine = builder.build_serialized_network(network, config)
    with open('model_python_trt.engine', mode='wb') as f:
        f.write(serialized_engine)
        print("generating file done!")