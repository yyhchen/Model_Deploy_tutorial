#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <logger.h>

using namespace nvinfer1;
using namespace sample;

const char* IN_NAME = "input";
const char* OUT_NAME = "output";
static const int IN_H = 224;
static const int IN_W = 224;
static const int BATCH_SIZE = 1;
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

int main(int argc, char** argv)
{
    // Create builder
    Logger m_logger;
    IBuilder* builder = createInferBuilder(m_logger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network
    INetworkDefinition* network = builder->createNetworkV2(EXPLICIT_BATCH);
    ITensor* input_tensor = network->addInput(IN_NAME, DataType::kFLOAT, Dims4{ BATCH_SIZE, 3, IN_H, IN_W });
    IPoolingLayer* pool = network->addPoolingNd(*input_tensor, PoolingType::kMAX, DimsHW{ 2, 2 });
    pool->setStrideNd(DimsHW{ 2, 2 });
    pool->getOutput(0)->setName(OUT_NAME);
    network->markOutput(*pool->getOutput(0));

    // Build engine
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(IN_NAME, OptProfileSelector::kMIN, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
    profile->setDimensions(IN_NAME, OptProfileSelector::kOPT, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
    profile->setDimensions(IN_NAME, OptProfileSelector::kMAX, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 20);
    IHostMemory* serialized_engine = builder->buildSerializedNetwork(*network, *config);

    // Serialize the model to engine file
    if (!serialized_engine) {
        std::cerr << "could not build serialized engine" << std::endl;
        return -1;
    }

    std::ofstream p("model.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open output file to save model" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    std::cout << "generating file done!" << std::endl;

    // Release resources
    // serialized_engine->destroy();
    // network->destroy();
    // builder->destroy();
    // config->destroy();

    delete serialized_engine;
    delete network;
    delete builder;
    delete config;

    return 0;
}