#pragma once
/**
 * @file MLClassifier.h
 * @brief 基于 ONNX Runtime 的 ML 文件类型分类器
 *
 * 本模块提供基于神经网络的文件类型分类功能，
 * 用于补充 FileCarver 中的签名检测。
 */

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <unordered_map>
#include <optional>
#include <cstdint>

// ONNX Runtime 前向声明
namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
    struct MemoryInfo;
}

namespace ML {

/**
 * @brief 文件分类特征向量（261 维）
 *
 * 特征组成：
 * - 256 维：字节频率分布 (0-255)
 * - 1 维：香农熵（归一化 0-1）
 * - 1 维：平均字节值（归一化 0-1）
 * - 1 维：标准差（归一化 0-1）
 * - 1 维：唯一字节比例
 * - 1 维：ASCII 可打印字符比例
 */
struct FileFeatures {
    static constexpr size_t FEATURE_DIM = 261;
    std::array<float, FEATURE_DIM> data;

    FileFeatures() { data.fill(0.0f); }
};

/**
 * @brief ML 模型分类结果
 */
struct ClassificationResult {
    std::string predictedType;      // 预测的文件类型（如 "pdf", "jpg", "unknown"）
    float confidence;               // 置信度分数 (0-1)
    std::unordered_map<std::string, float> probabilities;  // 所有类别的概率
    bool isUnknown = false;         // 如果无法可靠分类则为 true

    bool isValid() const { return !predictedType.empty() && confidence > 0; }
    bool isKnownType() const { return isValid() && !isUnknown; }
};

/**
 * @brief 批量分类输入
 */
struct BatchClassificationInput {
    const uint8_t* data;    // 原始文件数据
    size_t size;            // 数据大小
    uint64_t lcn;           // 逻辑簇号（用于追踪）
    size_t offset;          // 缓冲区内偏移（用于追踪）
};

/**
 * @brief 特征标准化参数
 */
struct NormalizationParams {
    std::array<float, FileFeatures::FEATURE_DIM> mean;
    std::array<float, FileFeatures::FEATURE_DIM> std;
};

/**
 * @brief 基于 ONNX 的文件类型分类器
 *
 * 使用示例：
 * @code
 * MLClassifier classifier;
 * if (classifier.loadModel("file_classifier.onnx")) {
 *     auto result = classifier.classify(data, dataSize);
 *     if (result && result->confidence > 0.8) {
 *         std::cout << "检测到: " << result->predictedType << std::endl;
 *     }
 * }
 * @endcode
 */
class MLClassifier {
public:
    MLClassifier();
    ~MLClassifier();

    // 禁用拷贝
    MLClassifier(const MLClassifier&) = delete;
    MLClassifier& operator=(const MLClassifier&) = delete;

    // 启用移动
    MLClassifier(MLClassifier&&) noexcept;
    MLClassifier& operator=(MLClassifier&&) noexcept;

    /**
     * @brief 加载 ONNX 模型和元数据
     * @param modelPath .onnx 文件路径
     * @return 加载成功返回 true
     *
     * 同时加载同路径下的 .json 元数据文件
     */
    bool loadModel(const std::wstring& modelPath);

    /**
     * @brief 检查模型是否已加载就绪
     */
    bool isLoaded() const;

    /**
     * @brief 分类文件数据
     * @param data 原始文件数据
     * @param size 数据大小（字节）
     * @param fragmentSize 分析的片段大小（默认 4096）
     * @return 分类结果，错误时返回 nullopt
     */
    std::optional<ClassificationResult> classify(
        const uint8_t* data,
        size_t size,
        size_t fragmentSize = 4096
    ) const;

    /**
     * @brief 使用预提取的特征进行分类
     */
    std::optional<ClassificationResult> classifyFeatures(
        const FileFeatures& features
    ) const;

    /**
     * @brief 批量分类多个样本
     * @param inputs 输入数据向量
     * @param fragmentSize 每个样本的分析片段大小
     * @return 分类结果向量（与输入大小相同）
     *
     * 比多次调用 classify() 更高效。
     * 当 ONNX Runtime 支持时使用批量推理。
     */
    std::vector<std::optional<ClassificationResult>> classifyBatch(
        const std::vector<BatchClassificationInput>& inputs,
        size_t fragmentSize = 4096
    ) const;

    /**
     * @brief 从原始数据提取特征
     * @param data 原始文件数据
     * @param size 数据大小
     * @return 特征向量
     */
    static FileFeatures extractFeatures(const uint8_t* data, size_t size);

    /**
     * @brief 获取支持的文件类型列表
     */
    std::vector<std::string> getSupportedTypes() const;

    /**
     * @brief 检查 ML 模型是否支持某文件类型
     * @param type 文件扩展名（如 "pdf", "jpg"）
     */
    bool isTypeSupported(const std::string& type) const;

    /**
     * @brief 获取有效分类的置信度阈值
     */
    float getConfidenceThreshold() const { return m_confidenceThreshold; }

    /**
     * @brief 设置置信度阈值
     */
    void setConfidenceThreshold(float threshold) { m_confidenceThreshold = threshold; }

    /**
     * @brief 检查 ONNX Runtime 是否可用
     */
    static bool isOnnxRuntimeAvailable();

private:
    // ONNX Runtime 对象（PIMPL 模式，隐藏 Ort 头文件）
    struct OrtResources;
    std::unique_ptr<OrtResources> m_ort;

    // 模型元数据
    std::unordered_map<int, std::string> m_labelMap;
    NormalizationParams m_normParams;
    bool m_loaded = false;
    float m_confidenceThreshold = 0.5f;

    // 内部方法
    bool loadMetadata(const std::wstring& jsonPath);
    std::vector<float> runInference(const std::vector<float>& input) const;
    std::vector<std::vector<float>> runBatchInference(const std::vector<std::vector<float>>& inputs) const;
    void normalizeFeatures(FileFeatures& features) const;
    ClassificationResult interpretOutput(const std::vector<float>& output) const;
};

/**
 * @brief 全局 ML 分类器实例（懒加载）
 */
MLClassifier& getMLClassifier();

/**
 * @brief 使用模型路径初始化 ML 分类器
 * @param modelPath ONNX 模型路径
 * @return 初始化成功返回 true
 */
bool initMLClassifier(const std::wstring& modelPath);

} // namespace ML
