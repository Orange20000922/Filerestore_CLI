#pragma once
#include <Windows.h>
#include <string>
#include <array>
#include <vector>
#include <memory>
#include <optional>

namespace Continuity {

// 连续性特征向量（64维）
struct ContinuityFeatures {
    static constexpr size_t FEATURE_DIM = 64;
    static constexpr size_t BLOCK_SIZE = 8192;  // 8KB 采样块

    std::array<float, FEATURE_DIM> data;

    ContinuityFeatures() { data.fill(0.0f); }

    // 转换为浮点数组（用于 ONNX 推理）
    std::vector<float> ToVector() const {
        return std::vector<float>(data.begin(), data.end());
    }

    // 特征索引定义（便于调试和理解）
    // === 块 1 特征 (0-15, 16维) ===
    static constexpr size_t B1_ENTROPY = 0;           // 熵值 [0-1]
    static constexpr size_t B1_MEAN = 1;              // 均值 [0-1]
    static constexpr size_t B1_STDDEV = 2;            // 标准差 [0-1]
    static constexpr size_t B1_ZERO_RATIO = 3;        // 零字节比例
    static constexpr size_t B1_HIGH_BYTE_RATIO = 4;   // 高字节(>0x80)比例
    static constexpr size_t B1_PRINTABLE_RATIO = 5;   // 可打印字符比例
    static constexpr size_t B1_HISTOGRAM_START = 6;   // 8区间直方图 (6-13)
    static constexpr size_t B1_PK_SIGNATURE = 14;     // PK签名检测分数
    static constexpr size_t B1_COMPRESSION_SCORE = 15;// 压缩数据特征分数

    // === 块 2 特征 (16-31, 16维) ===
    static constexpr size_t B2_ENTROPY = 16;
    static constexpr size_t B2_MEAN = 17;
    static constexpr size_t B2_STDDEV = 18;
    static constexpr size_t B2_ZERO_RATIO = 19;
    static constexpr size_t B2_HIGH_BYTE_RATIO = 20;
    static constexpr size_t B2_PRINTABLE_RATIO = 21;
    static constexpr size_t B2_HISTOGRAM_START = 22;  // 8区间直方图 (22-29)
    static constexpr size_t B2_PK_SIGNATURE = 30;
    static constexpr size_t B2_COMPRESSION_SCORE = 31;

    // === 边界特征 (32-47, 16维) ===
    static constexpr size_t ENTROPY_DIFF = 32;            // 熵值差异
    static constexpr size_t ENTROPY_GRADIENT = 33;        // 熵值变化梯度
    static constexpr size_t MEAN_DIFF = 34;               // 均值差异
    static constexpr size_t DISTRIBUTION_SIMILARITY = 35; // 字节分布相似度 (余弦)
    static constexpr size_t BOUNDARY_SMOOTHNESS = 36;     // 边界平滑度
    static constexpr size_t CROSS_CORRELATION = 37;       // 跨边界相关性
    static constexpr size_t TRANSITION_HIST_START = 38;   // 边界字节转移直方图 (38-45)
    static constexpr size_t LOCAL_HEADER_AT_BOUNDARY = 46;// 边界处是否有本地文件头
    static constexpr size_t EOCD_PROXIMITY = 47;          // 接近EOCD的可能性

    // === 格式特定特征 (48-63, 16维) ===
    // ZIP/DOCX/XLSX 格式:
    static constexpr size_t DEFLATE_CONTINUITY = 48;      // DEFLATE流连续性
    static constexpr size_t BLOCK_ALIGNMENT = 49;         // 块边界对齐分数
    static constexpr size_t COMPRESSION_RATIO = 50;       // 估算压缩率
    static constexpr size_t ZIP_STRUCTURE_SCORE = 51;     // ZIP结构完整性分数
    static constexpr size_t CENTRAL_DIR_SIGNATURE = 52;   // 中央目录签名检测
    static constexpr size_t DATA_DESCRIPTOR_SCORE = 53;   // 数据描述符检测分数
    static constexpr size_t FILE_HEADER_CHAIN = 54;       // 文件头链完整性
    static constexpr size_t COMPRESSED_SIZE_VALID = 55;   // 压缩大小字段有效性
    static constexpr size_t CRC_PATTERN = 56;             // CRC模式检测
    static constexpr size_t EXTRA_FIELD_VALID = 57;       // 扩展字段有效性
    static constexpr size_t FILENAME_VALID = 58;          // 文件名有效性分数
    static constexpr size_t VERSION_NEEDED = 59;          // 解压版本合理性
    static constexpr size_t GENERAL_FLAGS = 60;           // 通用标志合理性
    static constexpr size_t COMPRESSION_METHOD = 61;      // 压缩方法检测
    static constexpr size_t LAST_MOD_TIME = 62;           // 最后修改时间合理性
    static constexpr size_t FORMAT_RESERVED = 63;         // 保留特征

    // MP3 格式特征 (复用索引 48-63):
    // 48: 帧同步检测分数 (0xFF 0xFB/0xFA/0xF3)
    // 49: 帧头有效性分数
    // 50: 比特率一致性
    // 51: 采样率一致性
    // 52: 帧长度一致性
    // 53: ID3标签检测
    // 54: Xing/VBRI头检测
    // 55-63: 保留

    // MP4/MOV 格式特征 (复用索引 48-63):
    // 48: Box头检测分数
    // 49: Box大小有效性
    // 50: mdat连续性分数
    // 51: moov结构检测
    // 52: ftyp检测
    // 53: 原子边界对齐
    // 54-63: 保留
};

// 连续性检测结果
struct ContinuityResult {
    float score;            // 连续性分数 [0-1]
    float confidence;       // 置信度 [0-1]
    bool isContinuous;      // 是否连续 (score > threshold)
    std::string reason;     // 判断原因（调试用）

    ContinuityResult()
        : score(0.0f), confidence(0.0f), isContinuous(false) {}

    ContinuityResult(float s, float c, bool cont, const std::string& r = "")
        : score(s), confidence(c), isContinuous(cont), reason(r) {}
};

// 标准化参数
struct NormalizationParams {
    std::array<float, ContinuityFeatures::FEATURE_DIM> mean;
    std::array<float, ContinuityFeatures::FEATURE_DIM> std;
    bool loaded = false;

    NormalizationParams() {
        mean.fill(0.0f);
        std.fill(1.0f);
    }
};

// 块连续性检测器
class BlockContinuityDetector {
public:
    BlockContinuityDetector();
    ~BlockContinuityDetector();

    // 禁用拷贝
    BlockContinuityDetector(const BlockContinuityDetector&) = delete;
    BlockContinuityDetector& operator=(const BlockContinuityDetector&) = delete;

    // 加载 ONNX 模型
    bool LoadModel(const std::wstring& modelPath);

    // 检查模型是否已加载
    bool IsModelLoaded() const;

    // 获取默认模型搜索路径
    static std::vector<std::wstring> GetDefaultModelPaths();

    // 自动加载模型（搜索默认路径）
    bool AutoLoadModel();

    // ==================== 特征提取 ====================

    // 提取特征（静态方法，可独立使用）
    static ContinuityFeatures ExtractFeatures(
        const BYTE* block1_tail,        // 块1末尾数据
        size_t tail_size,               // 8192 字节 (8KB)
        const BYTE* block2_head,        // 块2开头数据
        size_t head_size,               // 8192 字节 (8KB)
        const std::string& fileType     // "zip", "mp4" 等
    );

    // 提取单块特征（内部使用）
    static void ExtractBlockFeatures(
        const BYTE* data,
        size_t size,
        float* features,                // 输出: 16维特征
        const std::string& fileType
    );

    // 提取边界特征
    static void ExtractBoundaryFeatures(
        const BYTE* block1_tail,
        size_t tail_size,
        const BYTE* block2_head,
        size_t head_size,
        float* features,                // 输出: 16维特征
        const std::string& fileType
    );

    // 提取 ZIP 特定特征
    static void ExtractZIPFeatures(
        const BYTE* block1_tail,
        size_t tail_size,
        const BYTE* block2_head,
        size_t head_size,
        float* features                 // 输出: 16维特征
    );

    // 提取 MP3 特定特征
    static void ExtractMP3Features(
        const BYTE* block1_tail,
        size_t tail_size,
        const BYTE* block2_head,
        size_t head_size,
        float* features                 // 输出: 16维特征
    );

    // 提取 MP4/MOV 特定特征
    static void ExtractMP4Features(
        const BYTE* block1_tail,
        size_t tail_size,
        const BYTE* block2_head,
        size_t head_size,
        float* features                 // 输出: 16维特征
    );

    // 提取通用媒体特征 (用于未特化的格式)
    static void ExtractGenericMediaFeatures(
        const BYTE* block1_tail,
        size_t tail_size,
        const BYTE* block2_head,
        size_t head_size,
        float* features                 // 输出: 16维特征
    );

    // ==================== 推理 ====================

    // 预测连续性（使用预提取的特征）
    ContinuityResult Predict(const ContinuityFeatures& features);

    // 便捷方法：直接从原始数据预测
    ContinuityResult PredictContinuity(
        const BYTE* block1_tail,
        size_t tail_size,
        const BYTE* block2_head,
        size_t head_size,
        const std::string& fileType,
        float threshold = 0.5f
    );

    // 批量预测
    std::vector<ContinuityResult> PredictBatch(
        const std::vector<ContinuityFeatures>& featuresBatch,
        float threshold = 0.5f
    );

    // ==================== 工具函数 ====================

    // 计算熵值
    static float CalculateEntropy(const BYTE* data, size_t size);

    // 计算字节分布直方图（8区间）
    static void CalculateHistogram8(
        const BYTE* data,
        size_t size,
        float* histogram               // 输出: 8维直方图
    );

    // 计算字节分布相似度（余弦相似度）
    static float CalculateDistributionSimilarity(
        const BYTE* data1, size_t size1,
        const BYTE* data2, size_t size2
    );

    // 计算边界平滑度（跨边界字节变化）
    static float CalculateBoundarySmoothness(
        const BYTE* block1_tail,
        size_t tail_size,
        const BYTE* block2_head,
        size_t head_size
    );

    // 检测 ZIP 签名
    static float DetectPKSignature(const BYTE* data, size_t size);

    // 检测压缩数据特征
    static float DetectCompressionScore(const BYTE* data, size_t size);

    // 检测 EOCD 接近度
    static float DetectEOCDProximity(const BYTE* data, size_t size);

    // 设置/获取阈值
    void SetThreshold(float threshold) { defaultThreshold = threshold; }
    float GetThreshold() const { return defaultThreshold; }

private:
    // PIMPL 隐藏 ONNX Runtime 实现
    class Impl;
    std::unique_ptr<Impl> pImpl;

    // 标准化参数
    NormalizationParams normParams;

    // 默认阈值
    float defaultThreshold = 0.5f;

    // 标准化特征
    void NormalizeFeatures(ContinuityFeatures& features) const;

    // 运行 ONNX 推理
    std::vector<float> RunInference(const std::vector<float>& input);

    // 加载标准化参数（从 JSON 元数据）
    bool LoadNormalizationParams(const std::wstring& metadataPath);
};

} // namespace Continuity
