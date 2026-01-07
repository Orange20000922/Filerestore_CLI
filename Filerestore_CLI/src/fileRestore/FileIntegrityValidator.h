#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

// ============================================================================
// 文件完整性验证模块
//
// 用途: 在基于签名恢复过程中检测损坏或受损的文件
// 方法: 熵分析、结构验证、统计分析
// ============================================================================

// 单个文件的验证结果
struct FileIntegrityScore {
    double entropyScore;        // 熵分析得分 (0-1)
    double structureScore;      // 结构验证得分 (0-1)
    double statisticalScore;    // 统计分析得分 (0-1)
    double footerScore;         // 文件尾验证得分 (0-1)
    double overallScore;        // 综合总分 (0-1)

    string diagnosis;           // 可读的诊断信息
    bool isLikelyCorrupted;     // 快速检查结果

    // 详细信息
    double entropy;             // 原始熵值 (0-8)
    double zeroRatio;           // 零字节比率 (0-1)
    double chiSquare;           // 卡方值
    bool hasValidHeader;        // 文件头验证结果
    bool hasValidFooter;        // 文件尾验证结果
    bool hasEntropyAnomaly;     // 是否检测到熵异常
    size_t anomalyOffset;       // 检测到异常的偏移量（如果有）

    FileIntegrityScore() : entropyScore(0), structureScore(0), statisticalScore(0),
                           footerScore(0), overallScore(0), isLikelyCorrupted(false),
                           entropy(0), zeroRatio(0), chiSquare(0),
                           hasValidHeader(false), hasValidFooter(false),
                           hasEntropyAnomaly(false), anomalyOffset(0) {}
};

// 不同文件类型的熵特征
struct EntropyProfile {
    string extension;
    double expectedMin;         // 最小预期熵值
    double expectedMax;         // 最大预期熵值
    double anomalyThreshold;    // 异常检测的熵方差阈值
    bool isCompressed;          // 文件是否通常是压缩的
};

// JPEG结构验证结果
struct JPEGValidation {
    bool hasSOI;                // 图像起始标记 (FFD8)
    bool hasEOI;                // 图像结束标记 (FFD9)
    bool hasValidMarkers;       // 有效的段标记
    bool hasValidDHT;           // 霍夫曼表
    bool hasValidDQT;           // 量化表
    bool hasSOF;                // 帧起始标记
    bool hasSOS;                // 扫描起始标记
    int markerCount;            // 标记总数
    double confidence;          // 整体置信度
};

// PNG结构验证结果
struct PNGValidation {
    bool hasValidSignature;     // 8字节PNG签名
    bool hasIHDR;               // 图像头部块
    bool hasIEND;               // 图像结束块
    bool hasValidCRC;           // CRC校验通过
    int chunkCount;             // 块总数
    double confidence;          // 整体置信度
};

// ZIP结构验证结果
struct ZIPValidation {
    bool hasValidLocalHeader;   // 本地文件头 (PK..)
    bool hasValidCentralDir;    // 中央目录
    bool hasEndOfCentralDir;    // 中央目录结束记录
    DWORD declaredFileCount;    // 声明的文件数量
    DWORD actualFileCount;      // 实际找到的文件数量
    double confidence;          // 整体置信度
};

// PDF结构验证结果
struct PDFValidation {
    bool hasValidHeader;        // %PDF-x.x 头部
    bool hasEOF;                // %%EOF 标记
    bool hasXRef;               // 交叉引用表
    bool hasTrailer;            // 尾部字典
    int objectCount;            // 找到的对象数量
    double confidence;          // 整体置信度
};

class FileIntegrityValidator {
private:
    // 不同文件类型的熵特征配置
    static const EntropyProfile entropyProfiles[];
    static const int entropyProfileCount;

    // 获取文件类型的熵特征配置
    static const EntropyProfile* GetEntropyProfile(const string& extension);

    // 熵计算辅助函数
    static double CalculateEntropy(const BYTE* data, size_t size);
    static vector<double> CalculateEntropyBlocks(const BYTE* data, size_t size, size_t blockSize);
    static bool DetectEntropyAnomaly(const vector<double>& entropies, double threshold, size_t& anomalyOffset);

    // 统计分析辅助函数
    static double CalculateChiSquare(const BYTE* data, size_t size);
    static double CalculateZeroRatio(const BYTE* data, size_t size);
    static bool IsLikelyRandom(const BYTE* data, size_t size);

    // 特定格式的结构验证
    static JPEGValidation ValidateJPEG(const BYTE* data, size_t size);
    static PNGValidation ValidatePNG(const BYTE* data, size_t size);
    static ZIPValidation ValidateZIP(const BYTE* data, size_t size);
    static PDFValidation ValidatePDF(const BYTE* data, size_t size);

    // 得分计算辅助函数
    static double EvaluateEntropyForType(double entropy, const string& extension);
    static double EvaluateStatistics(double zeroRatio, double chiSquare, const string& extension);
    static double ValidateFooter(const BYTE* data, size_t size, const string& extension);
    static double ValidateFileStructure(const BYTE* data, size_t size, const string& extension);

    // PNG验证用的CRC32计算
    static DWORD CalculateCRC32(const BYTE* data, size_t size);
    static DWORD ReadBigEndian32(const BYTE* data);

public:
    // 主验证函数
    static FileIntegrityScore Validate(const BYTE* data, size_t size, const string& extension);

    // 快速损坏检查
    static bool IsLikelyCorrupted(const BYTE* data, size_t size, const string& extension);

    // 单项得分函数
    static double GetEntropyScore(const BYTE* data, size_t size, const string& extension);
    static double GetStructureScore(const BYTE* data, size_t size, const string& extension);
    static double GetStatisticalScore(const BYTE* data, size_t size, const string& extension);

    // 阈值常量
    static constexpr double MIN_INTEGRITY_SCORE = 0.5;      // 最低可接受得分
    static constexpr double HIGH_CONFIDENCE_SCORE = 0.8;    // 高置信度阈值
    static constexpr double ENTROPY_VARIANCE_THRESHOLD = 1.5; // 熵异常阈值
    static constexpr double MAX_ZERO_RATIO_COMPRESSED = 0.05; // 压缩文件最大零字节比率
    static constexpr double MAX_ZERO_RATIO_GENERAL = 0.15;    // 一般文件最大零字节比率
};
