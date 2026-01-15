#pragma once
#include "MLFileRepair.h"
#include <cstdint>
#include <array>
#include <memory>

// ONNX Runtime 前向声明
namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
}

// ============================================================================
// 图像头部修复器 - 修复 JPEG/PNG 文件头部损坏
// ============================================================================

// 标准化参数（从训练时保存的 metadata.json 加载）
struct NormalizationParams {
    vector<float> mean;   // 31维均值
    vector<float> std;    // 31维标准差
    bool loaded = false;
};

// ML 特征向量（用于模型推理）
struct ImageFeatureVector {
    // 基础统计特征 (5维)
    float mean;                 // 字节均值 [0, 1]
    float stddev;               // 标准差 [0, 1]
    float entropy;              // 熵 [0, 1] (归一化到8)
    float zeroRatio;            // 零字节比例
    float ffRatio;              // 0xFF 字节比例

    // 直方图特征 (16维)
    array<float, 16> histogram; // 16区间分布

    // JPEG 特定特征 (4维)
    float ffMarkerDensity;      // 0xFF 标记密度（JPEG 特征）
    float dctBlockRegularity;   // DCT 块规律性
    float quantTablePresence;   // 量化表存在指示
    float huffmanTablePresence; // 霍夫曼表存在指示

    // PNG 特定特征 (4维)
    float deflateSignature;     // DEFLATE 压缩签名强度
    float filterBytePattern;    // 滤波字节模式
    float chunkBoundaryScore;   // Chunk 边界得分
    float colorTypeIndicator;   // 颜色类型指示

    // 图像尺寸预测特征 (2维)
    float predictedWidth;       // 预测宽度（归一化）
    float predictedHeight;      // 预测高度（归一化）

    // 总特征维度: 5 + 16 + 4 + 4 + 2 = 31
    static constexpr size_t FEATURE_DIM = 31;

    // 转换为浮点数组（用于 ONNX 推理）
    vector<float> ToVector() const;

    // 默认构造
    ImageFeatureVector();
};

class ImageHeaderRepairer : public FileTypeRepairer {
private:
    // ==================== ML 模型成员 ====================

    // ONNX Runtime 环境（静态共享）
    static std::shared_ptr<Ort::Env> s_ortEnv;

    // 图像类型分类器 (JPEG vs PNG)
    std::unique_ptr<Ort::Session> m_typeClassifier;

    // 可修复性预测器
    std::unique_ptr<Ort::Session> m_repairabilityPredictor;

    // 标准化参数
    NormalizationParams m_normParams;

    // ML 模型是否已加载
    bool m_mlModelsLoaded = false;

    // JPEG 标准头部
    struct JPEGHeader {
        // SOI (Start of Image)
        static constexpr BYTE SOI[2] = {0xFF, 0xD8};

        // JFIF APP0 标记
        static constexpr BYTE JFIF_APP0[18] = {
            0xFF, 0xE0,             // APP0 marker
            0x00, 0x10,             // Length (16 bytes)
            0x4A, 0x46, 0x49, 0x46, // "JFIF"
            0x00,                   // Null terminator
            0x01, 0x01,             // Version 1.1
            0x00,                   // Density units (no units)
            0x00, 0x01,             // X density (1)
            0x00, 0x01,             // Y density (1)
            0x00, 0x00              // Thumbnail (0x0)
        };

        // Exif APP1 标记（可选）
        static constexpr BYTE EXIF_APP1[10] = {
            0xFF, 0xE1,             // APP1 marker
            0x00, 0x08,             // Length (8 bytes minimum)
            0x45, 0x78, 0x69, 0x66, // "Exif"
            0x00, 0x00              // Padding
        };
    };

    // PNG 标准头部
    struct PNGHeader {
        // PNG 文件签名
        static constexpr BYTE SIGNATURE[8] = {
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A
        };

        // IHDR chunk (需要根据实际图像数据推断)
        struct IHDR {
            uint32_t width;
            uint32_t height;
            uint8_t bitDepth;
            uint8_t colorType;
            uint8_t compressionMethod;
            uint8_t filterMethod;
            uint8_t interlaceMethod;
        };
    };

    // ==================== JPEG 修复 ====================

    // 分析 JPEG 损坏
    DamageAnalysis AnalyzeJPEGDamage(const vector<BYTE>& data);

    // 查找 JPEG 标记（如 SOF, DQT 等）
    static size_t FindJPEGMarker(const vector<BYTE>& data, BYTE marker, size_t startOffset = 0);

    // 重建 JPEG 头部
    bool RebuildJPEGHeader(vector<BYTE>& data);

    // 提取 JPEG 图像尺寸（从 SOF 段）
    bool ExtractJPEGDimensions(const vector<BYTE>& data, uint16_t& width, uint16_t& height);

    // ==================== PNG 修复 ====================

    // 分析 PNG 损坏
    DamageAnalysis AnalyzePNGDamage(const vector<BYTE>& data);

    // 查找 PNG chunk
    static size_t FindPNGChunk(const vector<BYTE>& data, const char* chunkType, size_t startOffset = 0);

    // 重建 PNG 头部（包括 IHDR）
    bool RebuildPNGHeader(vector<BYTE>& data);

    // 重建 PNG 头部（指定尺寸）
    bool RebuildPNGHeaderWithSize(vector<BYTE>& data, uint32_t width, uint32_t height);

    // 提取 PNG 图像尺寸（从 IHDR chunk）
    bool ExtractPNGDimensions(const vector<BYTE>& data, uint32_t& width, uint32_t& height);

    // 计算 CRC32（PNG chunk 校验）
    uint32_t CalculateCRC32(const BYTE* data, size_t length);

    // ==================== ML 特征提取 ====================

    // 提取基础统计特征
    void ExtractBasicFeatures(const vector<BYTE>& data, size_t offset, size_t length,
                              ImageFeatureVector& features);

    // 提取 JPEG 特定特征
    void ExtractJPEGFeatures(const vector<BYTE>& data, ImageFeatureVector& features);

    // 提取 PNG 特定特征
    void ExtractPNGFeatures(const vector<BYTE>& data, ImageFeatureVector& features);

    // 从 IDAT 数据推断 PNG 尺寸（启发式方法）
    bool InferPNGDimensionsFromIDAT(const vector<BYTE>& data, uint32_t& width, uint32_t& height);

    // ==================== 旧接口（兼容） ====================

    // 从图像主体数据提取统计特征
    vector<float> ExtractImageFeatures(const vector<BYTE>& data, size_t offset, size_t length);

    // 检测图像数据块的熵（判断是否为压缩数据）
    double CalculateEntropy(const vector<BYTE>& data, size_t offset, size_t length);

public:
    ImageHeaderRepairer();
    ~ImageHeaderRepairer() override;

    // 实现基类接口
    DamageAnalysis AnalyzeDamage(const vector<BYTE>& data) override;
    RepairReport TryRepair(vector<BYTE>& data, RepairType type) override;
    vector<string> GetSupportedTypes() const override;

    // ==================== ML 模型管理 ====================

    // 加载修复用 ML 模型
    // 搜索路径: exe目录, models/repair/, ml/models/repair/
    bool LoadRepairModels();

    // 检查 ML 模型是否已加载
    bool IsMLModelsLoaded() const { return m_mlModelsLoaded; }

    // 获取模型搜索路径
    static vector<wstring> GetRepairModelSearchPaths();

    // ==================== ML 增强接口 ====================

    // 提取完整特征向量（用于 ML 推理）
    ImageFeatureVector ExtractFullFeatures(const vector<BYTE>& data);

    // 预测图像类型（基于特征）- 使用 ML 模型或规则回退
    // 返回: "jpeg", "png", 或 "unknown"
    string PredictImageType(const ImageFeatureVector& features);

    // 使用 ML 模型预测图像类型（需要先加载模型）
    // 返回: "jpeg"=0, "png"=1, 以及置信度
    bool PredictImageTypeML(const ImageFeatureVector& features,
                            string& predictedType, float& confidence);

    // 预测文件是否可修复（使用 ML 模型）
    // 返回: 可修复=true, 不可修复=false, 以及置信度
    bool PredictRepairability(const ImageFeatureVector& features, float& confidence);

    // 预测 PNG 尺寸（基于特征）
    bool PredictPNGDimensions(const ImageFeatureVector& features,
                              uint32_t& width, uint32_t& height);

    // 使用 ML 辅助修复（结合规则和特征）
    RepairReport TryRepairML(vector<BYTE>& data, const string& expectedType = "");

    // 工具方法
    static bool IsJPEG(const vector<BYTE>& data);
    static bool IsPNG(const vector<BYTE>& data);
    static bool IsLikelyJPEG(const vector<BYTE>& data);  // 宽松检测
    static bool IsLikelyPNG(const vector<BYTE>& data);   // 宽松检测

private:
    // ==================== ML 内部方法 ====================

    // 加载标准化参数（从 metadata.json）
    bool LoadNormalizationParams(const wstring& metadataPath);

    // 对特征向量进行标准化
    vector<float> NormalizeFeatures(const ImageFeatureVector& features);

    // 运行 ONNX 推理
    vector<float> RunInference(Ort::Session* session, const vector<float>& input);
};
