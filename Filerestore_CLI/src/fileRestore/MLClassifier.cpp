/**
 * @file MLClassifier.cpp
 * @brief 基于机器学习的文件类型分类器实现
 */

#include "MLClassifier.h"
#include "Logger.h"

#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <mutex>

// 条件编译包含 ONNX Runtime
#ifdef USE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace ML {

// ============================================================================
// OrtResources - PIMPL 实现（指针实现惯用法）
// ============================================================================

struct MLClassifier::OrtResources {
#ifdef USE_ONNX_RUNTIME
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<std::string> inputNameStrings;
    std::vector<std::string> outputNameStrings;
#endif
};

// ============================================================================
// MLClassifier 类实现
// ============================================================================

MLClassifier::MLClassifier() : m_ort(std::make_unique<OrtResources>()) {
    // 初始化默认归一化参数（将被元数据覆盖）
    m_normParams.mean.fill(0.0f);
    m_normParams.std.fill(1.0f);
}

MLClassifier::~MLClassifier() = default;

MLClassifier::MLClassifier(MLClassifier&&) noexcept = default;
MLClassifier& MLClassifier::operator=(MLClassifier&&) noexcept = default;

bool MLClassifier::isOnnxRuntimeAvailable() {
#ifdef USE_ONNX_RUNTIME
    return true;
#else
    return false;
#endif
}

bool MLClassifier::loadModel(const std::wstring& modelPath) {
#ifdef USE_ONNX_RUNTIME
    try {
        // 检查模型文件是否存在
        if (!std::filesystem::exists(modelPath)) {
            LOG_ERROR(L"ONNX model file not found: " + modelPath);
            return false;
        }

        // 初始化 ONNX Runtime 环境
        m_ort->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MLClassifier");

        // 会话选项
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // 尝试使用 GPU（如果可用）
        try {
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id = 0;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            LOG_INFO("ONNX Runtime: Using CUDA execution provider");
        } catch (...) {
            LOG_INFO("ONNX Runtime: Using CPU execution provider");
        }

        // 创建会话
        m_ort->session = std::make_unique<Ort::Session>(*m_ort->env, modelPath.c_str(), sessionOptions);

        // 获取输入/输出名称
        Ort::AllocatorWithDefaultOptions allocator;

        size_t numInputs = m_ort->session->GetInputCount();
        for (size_t i = 0; i < numInputs; i++) {
            auto name = m_ort->session->GetInputNameAllocated(i, allocator);
            m_ort->inputNameStrings.push_back(name.get());
        }

        size_t numOutputs = m_ort->session->GetOutputCount();
        for (size_t i = 0; i < numOutputs; i++) {
            auto name = m_ort->session->GetOutputNameAllocated(i, allocator);
            m_ort->outputNameStrings.push_back(name.get());
        }

        // 转换为 const char* 指针
        m_ort->inputNames.clear();
        m_ort->outputNames.clear();
        for (const auto& name : m_ort->inputNameStrings) {
            m_ort->inputNames.push_back(name.c_str());
        }
        for (const auto& name : m_ort->outputNameStrings) {
            m_ort->outputNames.push_back(name.c_str());
        }

        // 创建内存信息
        m_ort->memoryInfo = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
        );

        // 加载元数据 JSON 文件
        std::wstring metadataPath = modelPath;
        size_t dotPos = metadataPath.rfind(L'.');
        if (dotPos != std::wstring::npos) {
            metadataPath = metadataPath.substr(0, dotPos) + L".json";
        }

        if (!loadMetadata(metadataPath)) {
            LOG_WARNING("Failed to load model metadata, using defaults");
        }

        m_loaded = true;
        LOG_INFO(L"ONNX model loaded successfully: " + modelPath);
        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX Runtime error: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Error loading ONNX model: " + std::string(e.what()));
        return false;
    }
#else
    LOG_WARNING("ONNX Runtime not available (USE_ONNX_RUNTIME not defined)");
    return false;
#endif
}

bool MLClassifier::loadMetadata(const std::wstring& jsonPath) {
    try {
        std::ifstream file(jsonPath);
        if (!file.is_open()) {
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();

        // 简单 JSON 解析（避免外部依赖）
        // 解析 label_map（标签映射）
        size_t labelMapPos = content.find("\"label_map\"");
        if (labelMapPos != std::string::npos) {
            size_t braceStart = content.find('{', labelMapPos);
            size_t braceEnd = content.find('}', braceStart);
            if (braceStart != std::string::npos && braceEnd != std::string::npos) {
                std::string labelMapStr = content.substr(braceStart + 1, braceEnd - braceStart - 1);

                // 解析键值对
                size_t pos = 0;
                while (pos < labelMapStr.size()) {
                    // 查找键
                    size_t keyStart = labelMapStr.find('"', pos);
                    if (keyStart == std::string::npos) break;
                    size_t keyEnd = labelMapStr.find('"', keyStart + 1);
                    if (keyEnd == std::string::npos) break;

                    std::string keyStr = labelMapStr.substr(keyStart + 1, keyEnd - keyStart - 1);
                    int key = std::stoi(keyStr);

                    // 查找值
                    size_t valStart = labelMapStr.find('"', keyEnd + 1);
                    if (valStart == std::string::npos) break;
                    size_t valEnd = labelMapStr.find('"', valStart + 1);
                    if (valEnd == std::string::npos) break;

                    std::string value = labelMapStr.substr(valStart + 1, valEnd - valStart - 1);
                    m_labelMap[key] = value;

                    pos = valEnd + 1;
                }
            }
        }

        // 解析归一化参数
        auto parseArray = [&content](const std::string& key) -> std::vector<float> {
            std::vector<float> result;
            size_t keyPos = content.find("\"" + key + "\"");
            if (keyPos != std::string::npos) {
                size_t arrayStart = content.find('[', keyPos);
                size_t arrayEnd = content.find(']', arrayStart);
                if (arrayStart != std::string::npos && arrayEnd != std::string::npos) {
                    std::string arrayStr = content.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
                    std::stringstream ss(arrayStr);
                    std::string token;
                    while (std::getline(ss, token, ',')) {
                        try {
                            result.push_back(std::stof(token));
                        } catch (...) {}
                    }
                }
            }
            return result;
        };

        auto meanVec = parseArray("mean");
        auto stdVec = parseArray("std");

        if (meanVec.size() == FileFeatures::FEATURE_DIM) {
            std::copy(meanVec.begin(), meanVec.end(), m_normParams.mean.begin());
        }
        if (stdVec.size() == FileFeatures::FEATURE_DIM) {
            std::copy(stdVec.begin(), stdVec.end(), m_normParams.std.begin());
        }

        return !m_labelMap.empty();

    } catch (const std::exception&) {
        return false;
    }
}

bool MLClassifier::isLoaded() const {
    return m_loaded;
}

FileFeatures MLClassifier::extractFeatures(const uint8_t* data, size_t size) {
    FileFeatures features;

    if (data == nullptr || size == 0) {
        return features;
    }

    // 1. 字节频率（256 维）
    std::array<size_t, 256> byteCounts{};
    for (size_t i = 0; i < size; i++) {
        byteCounts[data[i]]++;
    }

    for (int i = 0; i < 256; i++) {
        features.data[i] = static_cast<float>(byteCounts[i]) / static_cast<float>(size);
    }

    // 2. 香农熵（归一化到 0-1）
    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (byteCounts[i] > 0) {
            double p = static_cast<double>(byteCounts[i]) / static_cast<double>(size);
            entropy -= p * std::log2(p);
        }
    }
    features.data[256] = static_cast<float>(entropy / 8.0);  // 最大熵为 8

    // 3. 平均字节值（归一化到 0-1）
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        sum += data[i];
    }
    double mean = sum / static_cast<double>(size);
    features.data[257] = static_cast<float>(mean / 255.0);

    // 4. 标准差（归一化到 0-1）
    double sqSum = 0.0;
    for (size_t i = 0; i < size; i++) {
        double diff = static_cast<double>(data[i]) - mean;
        sqSum += diff * diff;
    }
    double stdDev = std::sqrt(sqSum / static_cast<double>(size));
    features.data[258] = static_cast<float>(stdDev / 255.0);

    // 5. 唯一字节比率
    size_t uniqueCount = 0;
    for (int i = 0; i < 256; i++) {
        if (byteCounts[i] > 0) uniqueCount++;
    }
    features.data[259] = static_cast<float>(uniqueCount) / 256.0f;

    // 6. ASCII 可打印字符比率（32-126）
    size_t asciiCount = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] >= 32 && data[i] <= 126) {
            asciiCount++;
        }
    }
    features.data[260] = static_cast<float>(asciiCount) / static_cast<float>(size);

    return features;
}

void MLClassifier::normalizeFeatures(FileFeatures& features) const {
    for (size_t i = 0; i < FileFeatures::FEATURE_DIM; i++) {
        float std = m_normParams.std[i];
        if (std < 1e-8f) std = 1e-8f;  // 防止除零
        features.data[i] = (features.data[i] - m_normParams.mean[i]) / std;
    }
}

std::vector<float> MLClassifier::runInference(const std::vector<float>& input) const {
#ifdef USE_ONNX_RUNTIME
    if (!m_loaded || !m_ort->session) {
        return {};
    }

    try {
        // 创建输入张量
        std::array<int64_t, 2> inputShape = {1, static_cast<int64_t>(FileFeatures::FEATURE_DIM)};

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            *m_ort->memoryInfo,
            const_cast<float*>(input.data()),
            input.size(),
            inputShape.data(),
            inputShape.size()
        );

        // 运行推理
        auto outputTensors = m_ort->session->Run(
            Ort::RunOptions{nullptr},
            m_ort->inputNames.data(),
            &inputTensor,
            1,
            m_ort->outputNames.data(),
            m_ort->outputNames.size()
        );

        // 获取输出
        if (outputTensors.empty()) {
            return {};
        }

        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        size_t outputSize = outputInfo.GetElementCount();

        return std::vector<float>(outputData, outputData + outputSize);

    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX inference error: " + std::string(e.what()));
        return {};
    }
#else
    return {};
#endif
}

std::optional<ClassificationResult> MLClassifier::classify(
    const uint8_t* data,
    size_t size,
    size_t fragmentSize
) const {
    if (!m_loaded) {
        return std::nullopt;
    }

    // 使用指定的片段大小，如果数据更小则使用实际大小
    size_t analyzeSize = (std::min)(size, fragmentSize);

    // 提取特征
    FileFeatures features = extractFeatures(data, analyzeSize);

    return classifyFeatures(features);
}

std::optional<ClassificationResult> MLClassifier::classifyFeatures(
    const FileFeatures& features
) const {
    if (!m_loaded) {
        return std::nullopt;
    }

    // 归一化特征
    FileFeatures normalizedFeatures = features;
    normalizeFeatures(normalizedFeatures);

    // 转换为向量用于推理
    std::vector<float> input(normalizedFeatures.data.begin(), normalizedFeatures.data.end());

    // 运行推理
    std::vector<float> logits = runInference(input);
    if (logits.empty()) {
        return std::nullopt;
    }

    // 应用 softmax 获取概率
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sumExp = 0.0f;

    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sumExp += probs[i];
    }

    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sumExp;
    }

    // 查找预测类别
    auto maxIt = std::max_element(probs.begin(), probs.end());
    int predictedIdx = static_cast<int>(std::distance(probs.begin(), maxIt));
    float confidence = *maxIt;

    // 构建结果
    ClassificationResult result;

    auto labelIt = m_labelMap.find(predictedIdx);
    if (labelIt != m_labelMap.end()) {
        result.predictedType = labelIt->second;
    } else {
        result.predictedType = "unknown";
        result.isUnknown = true;
    }

    result.confidence = confidence;

    // 如果置信度低于阈值则标记为未知
    if (confidence < m_confidenceThreshold) {
        result.isUnknown = true;
        // 保留预测类型但标记为不确定
    }

    // 填充所有概率
    for (size_t i = 0; i < probs.size(); i++) {
        auto it = m_labelMap.find(static_cast<int>(i));
        if (it != m_labelMap.end()) {
            result.probabilities[it->second] = probs[i];
        }
    }

    return result;
}

ClassificationResult MLClassifier::interpretOutput(const std::vector<float>& logits) const {
    ClassificationResult result;

    if (logits.empty()) {
        result.isUnknown = true;
        result.predictedType = "unknown";
        result.confidence = 0.0f;
        return result;
    }

    // 应用 softmax 获取概率
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sumExp = 0.0f;

    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sumExp += probs[i];
    }

    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sumExp;
    }

    // 查找预测类别
    auto maxIt = std::max_element(probs.begin(), probs.end());
    int predictedIdx = static_cast<int>(std::distance(probs.begin(), maxIt));
    float confidence = *maxIt;

    auto labelIt = m_labelMap.find(predictedIdx);
    if (labelIt != m_labelMap.end()) {
        result.predictedType = labelIt->second;
    } else {
        result.predictedType = "unknown";
        result.isUnknown = true;
    }

    result.confidence = confidence;

    if (confidence < m_confidenceThreshold) {
        result.isUnknown = true;
    }

    // 填充所有概率
    for (size_t i = 0; i < probs.size(); i++) {
        auto it = m_labelMap.find(static_cast<int>(i));
        if (it != m_labelMap.end()) {
            result.probabilities[it->second] = probs[i];
        }
    }

    return result;
}

std::vector<std::vector<float>> MLClassifier::runBatchInference(
    const std::vector<std::vector<float>>& inputs
) const {
#ifdef USE_ONNX_RUNTIME
    if (!m_loaded || !m_ort->session || inputs.empty()) {
        return {};
    }

    try {
        size_t batchSize = inputs.size();

        // 将输入展平到单个缓冲区
        std::vector<float> flatInput;
        flatInput.reserve(batchSize * FileFeatures::FEATURE_DIM);
        for (const auto& input : inputs) {
            flatInput.insert(flatInput.end(), input.begin(), input.end());
        }

        // 创建带批次维度的输入张量
        std::array<int64_t, 2> inputShape = {
            static_cast<int64_t>(batchSize),
            static_cast<int64_t>(FileFeatures::FEATURE_DIM)
        };

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            *m_ort->memoryInfo,
            flatInput.data(),
            flatInput.size(),
            inputShape.data(),
            inputShape.size()
        );

        // 运行推理
        auto outputTensors = m_ort->session->Run(
            Ort::RunOptions{nullptr},
            m_ort->inputNames.data(),
            &inputTensor,
            1,
            m_ort->outputNames.data(),
            m_ort->outputNames.size()
        );

        if (outputTensors.empty()) {
            return {};
        }

        // 获取输出
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        auto outputShape = outputInfo.GetShape();

        // 输出形状应为 [batchSize, numClasses]
        size_t numClasses = (outputShape.size() > 1) ? outputShape[1] : outputInfo.GetElementCount() / batchSize;

        // 将输出拆分为单独的结果
        std::vector<std::vector<float>> results;
        results.reserve(batchSize);

        for (size_t i = 0; i < batchSize; i++) {
            std::vector<float> singleOutput(outputData + i * numClasses, outputData + (i + 1) * numClasses);
            results.push_back(std::move(singleOutput));
        }

        return results;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX batch inference error: " + std::string(e.what()));
        return {};
    }
#else
    return {};
#endif
}

std::vector<std::optional<ClassificationResult>> MLClassifier::classifyBatch(
    const std::vector<BatchClassificationInput>& inputs,
    size_t fragmentSize
) const {
    std::vector<std::optional<ClassificationResult>> results;
    results.reserve(inputs.size());

    if (!m_loaded || inputs.empty()) {
        results.resize(inputs.size(), std::nullopt);
        return results;
    }

    // 为所有输入提取并归一化特征
    std::vector<std::vector<float>> normalizedInputs;
    normalizedInputs.reserve(inputs.size());

    for (const auto& input : inputs) {
        size_t analyzeSize = (std::min)(input.size, fragmentSize);
        FileFeatures features = extractFeatures(input.data, analyzeSize);
        normalizeFeatures(features);

        std::vector<float> featureVec(features.data.begin(), features.data.end());
        normalizedInputs.push_back(std::move(featureVec));
    }

    // 运行批量推理
    std::vector<std::vector<float>> batchOutputs = runBatchInference(normalizedInputs);

    if (batchOutputs.size() != inputs.size()) {
        // 如果批量推理失败则回退到单个推理
        for (const auto& input : inputs) {
            results.push_back(classify(input.data, input.size, fragmentSize));
        }
        return results;
    }

    // 解释每个输出
    for (const auto& output : batchOutputs) {
        results.push_back(interpretOutput(output));
    }

    return results;
}

std::vector<std::string> MLClassifier::getSupportedTypes() const {
    std::vector<std::string> types;
    for (const auto& [idx, name] : m_labelMap) {
        types.push_back(name);
    }
    std::sort(types.begin(), types.end());
    return types;
}

bool MLClassifier::isTypeSupported(const std::string& type) const {
    for (const auto& [idx, name] : m_labelMap) {
        if (name == type) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// 全局实例
// ============================================================================

static std::unique_ptr<MLClassifier> g_mlClassifier;
static std::once_flag g_mlClassifierInitFlag;

MLClassifier& getMLClassifier() {
    std::call_once(g_mlClassifierInitFlag, []() {
        g_mlClassifier = std::make_unique<MLClassifier>();
    });
    return *g_mlClassifier;
}

bool initMLClassifier(const std::wstring& modelPath) {
    return getMLClassifier().loadModel(modelPath);
}

} // namespace ML
