#pragma once
/**
 * @file DatasetGenerator.h
 * @brief ML Dataset Generator - 从本地文件系统提取特征生成训练数据集
 *
 * 用于扫描指定目录/卷，提取文件特征并导出为CSV或二进制格式，
 * 供Python端训练ML模型使用。
 *
 * 支持两种模式：
 * 1. 分类模式 (Classification): 261维特征，用于文件类型分类
 * 2. 修复模式 (Repair): 31维图像特征，用于训练文件修复模型
 */

#include <Windows.h>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <functional>
#include <random>
#include "MLClassifier.h"
#include "ImageHeaderRepairer.h"
#include "BlockContinuityDetector.h"

namespace ML {

// ============================================================================
// 数据集生成模式
// ============================================================================
enum class DatasetMode {
    CLASSIFICATION,     // 分类模式: 261维特征，用于文件类型分类
    REPAIR,             // 修复模式: 31维图像特征 + 损坏模拟
    CONTINUITY          // 连续性模式: 64维特征，用于块连续性检测
};

// ============================================================================
// 损坏类型（用于训练数据生成）
// ============================================================================
enum class SimulatedDamageType {
    NONE,                   // 无损坏（正常样本）
    HEADER_ZEROED,          // 头部清零
    HEADER_RANDOM,          // 头部随机字节
    PARTIAL_OVERWRITE,      // 部分数据覆盖
    TRUNCATED,              // 文件截断
    RANDOM_CORRUPTION       // 随机位置损坏
};

// ============================================================================
// 数据集样本信息（分类模式）
// ============================================================================
struct SampleInfo {
    std::wstring filePath;              // 文件完整路径
    std::string extension;              // 文件扩展名（标签）
    FileFeatures features;              // 261维特征向量
    ULONGLONG fileSize;                 // 文件大小
    bool valid;                         // 是否有效

    SampleInfo() : fileSize(0), valid(false) {}
};

// ============================================================================
// 修复训练样本信息（修复模式）
// ============================================================================
struct RepairSampleInfo {
    std::wstring filePath;              // 原始文件路径
    std::string imageType;              // 图像类型: "jpeg" 或 "png"
    ImageFeatureVector features;        // 31维图像特征向量
    SimulatedDamageType damageType;     // 损坏类型
    float damageSeverity;               // 损坏严重程度 (0.0-1.0)
    size_t damageOffset;                // 损坏起始位置
    size_t damageSize;                  // 损坏区域大小
    bool isRepairable;                  // 是否可修复（标签）
    ULONGLONG fileSize;                 // 文件大小
    bool valid;                         // 是否有效

    RepairSampleInfo()
        : damageType(SimulatedDamageType::NONE)
        , damageSeverity(0.0f)
        , damageOffset(0)
        , damageSize(0)
        , isRepairable(true)
        , fileSize(0)
        , valid(false) {}
};

// ============================================================================
// 连续性样本类型（用于训练数据标注）
// ============================================================================
enum class ContinuitySampleType {
    SAME_FILE,              // 同一文件的连续块（正样本）
    DIFFERENT_FILES,        // 不同文件的块拼接（负样本）
    FILE_BOUNDARY,          // 文件边界处的块（负样本）
    RANDOM_DATA,            // ZIP块 + 随机数据（负样本）
    DIFFERENT_TYPE,         // ZIP块 + 其他类型文件数据（负样本）
    // ========== 损坏样本类型（负样本）==========
    CORRUPTED_TRUNCATION,   // 截断损坏：块2被截断并填充零
    CORRUPTED_BITFLIP,      // 比特翻转：块2中随机比特被翻转
    CORRUPTED_ZERO_FILL,    // 零填充：块2部分区域被清零
    CORRUPTED_RANDOM_FILL,  // 随机填充：块2部分区域被随机数据覆盖
    CORRUPTED_HEADER_DAMAGE,// 头部损坏：块2的格式头部被破坏
    CORRUPTED_PARTIAL       // 部分覆盖：块2部分被其他文件数据覆盖
};

// ============================================================================
// 连续性训练样本信息（连续性模式）
// ============================================================================
struct ContinuitySampleInfo {
    std::wstring file1Path;             // 块1来源文件路径
    std::wstring file2Path;             // 块2来源文件路径（负样本时不同）
    std::string fileType;               // 文件类型（如 "zip"）
    Continuity::ContinuityFeatures features;  // 64维连续性特征向量
    ContinuitySampleType sampleType;    // 样本类型
    bool isContinuous;                  // 是否连续（标签: 1=连续, 0=不连续）
    ULONGLONG block1Offset;             // 块1在文件中的偏移
    ULONGLONG block2Offset;             // 块2在文件中的偏移
    // ========== 损坏样本附加信息 ==========
    float corruptionSeverity;           // 损坏严重程度 (0.0-1.0)
    size_t corruptionOffset;            // 损坏在块2中的起始位置
    size_t corruptionSize;              // 损坏区域大小
    bool valid;                         // 是否有效

    ContinuitySampleInfo()
        : sampleType(ContinuitySampleType::SAME_FILE)
        , isContinuous(true)
        , block1Offset(0)
        , block2Offset(0)
        , corruptionSeverity(0.0f)
        , corruptionOffset(0)
        , corruptionSize(0)
        , valid(false) {}
};

// ============================================================================
// 数据集统计信息
// ============================================================================
struct DatasetStats {
    std::unordered_map<std::string, size_t> typeCounts;  // 每类样本数
    size_t totalSamples;                                  // 总样本数
    size_t totalFilesScanned;                            // 扫描文件总数
    size_t skippedTooSmall;                              // 因太小跳过
    size_t skippedReadError;                             // 因读取错误跳过
    size_t skippedQuotaReached;                          // 因配额满跳过
    double elapsedSeconds;                               // 耗时秒数

    // 修复模式统计
    std::unordered_map<int, size_t> damageTypeCounts;    // 每种损坏类型数量
    size_t normalSamples;                                // 正常（未损坏）样本数
    size_t damagedSamples;                               // 损坏样本数
    size_t repairableSamples;                            // 可修复样本数
    size_t unrepairableSamples;                          // 不可修复样本数

    // 连续性模式统计
    size_t positiveSamples;                              // 正样本数（同一文件连续块）
    size_t negativeSamples;                              // 负样本数（不连续块）
    std::unordered_map<int, size_t> sampleTypeCounts;    // 每种样本类型数量

    DatasetStats() : totalSamples(0), totalFilesScanned(0),
                     skippedTooSmall(0), skippedReadError(0),
                     skippedQuotaReached(0), elapsedSeconds(0),
                     normalSamples(0), damagedSamples(0),
                     repairableSamples(0), unrepairableSamples(0),
                     positiveSamples(0), negativeSamples(0) {}
};

// ============================================================================
// 数据集生成器配置
// ============================================================================
struct DatasetGeneratorConfig {
    // ==================== 通用配置 ====================
    DatasetMode mode;                       // 数据集模式
    std::set<std::string> targetTypes;      // 目标文件类型
    size_t maxSamplesPerType;               // 每类最大样本数
    size_t minFileSize;                     // 最小文件大小
    size_t fragmentSize;                    // 特征提取片段大小
    int workerThreads;                      // 工作线程数
    bool includeFilePath;                   // CSV中是否包含文件路径
    bool verbose;                           // 详细输出

    // ==================== 修复模式配置 ====================
    bool generateDamagedSamples;            // 是否生成损坏样本
    float damageRatio;                      // 损坏样本比例 (0.0-1.0)
    size_t maxDamageSize;                   // 最大损坏区域大小
    float minDamageSeverity;                // 最小损坏严重程度
    float maxDamageSeverity;                // 最大损坏严重程度
    std::set<SimulatedDamageType> enabledDamageTypes;  // 启用的损坏类型

    // ==================== 连续性模式配置 ====================
    size_t continuityBlockSize;             // 连续性检测块大小（默认8KB）
    size_t samplesPerFile;                  // 每个文件生成的样本数（固定模式时使用）
    float posNegRatio;                      // 正负样本比例（1.0 = 相等）

    // ==================== 自适应采样配置 ====================
    bool useAdaptiveSampling;               // 是否启用自适应采样
    float adaptiveSamplingRate;             // 自适应采样率（可用块数的百分比，默认1%）
    size_t minSamplesPerFile;               // 每个文件最少样本数
    size_t maxSamplesPerFile;               // 每个文件最多样本数
    bool useLocalFiles;                     // 使用本地文件
    bool useGovdocs;                        // 使用 Govdocs 数据集
    std::vector<std::wstring> localPaths;   // 本地文件搜索路径
    std::set<ContinuitySampleType> enabledSampleTypes;  // 启用的负样本类型

    // ==================== 连续性模式 - 损坏样本配置 ====================
    bool generateCorruptedSamples;          // 是否生成损坏样本
    float corruptedSampleRatio;             // 损坏样本占负样本的比例 (0.0-1.0)
    float minCorruptionSeverity;            // 最小损坏严重程度 (0.0-1.0)
    float maxCorruptionSeverity;            // 最大损坏严重程度 (0.0-1.0)
    float bitFlipRate;                      // 比特翻转率 (0.001 = 0.1%)
    std::set<ContinuitySampleType> enabledCorruptionTypes;  // 启用的损坏类型

    // ==================== 增量处理配置 ====================
    bool incrementalMode;                   // 是否启用增量模式
    std::string progressFilePath;           // 进度文件路径（JSON）
    bool appendToExistingCSV;               // 是否追加到现有CSV
    size_t saveProgressInterval;            // 保存进度的间隔（文件数）

    DatasetGeneratorConfig()
        : mode(DatasetMode::CLASSIFICATION)
        , maxSamplesPerType(2000)
        , minFileSize(4096)
        , fragmentSize(4096)
        , workerThreads(8)
        , includeFilePath(false)
        , verbose(true)
        , generateDamagedSamples(true)
        , damageRatio(0.7f)
        , maxDamageSize(1024)
        , minDamageSeverity(0.1f)
        , maxDamageSeverity(0.9f)
        , continuityBlockSize(8192)
        , samplesPerFile(10)
        , posNegRatio(1.0f)
        , useAdaptiveSampling(true)
        , adaptiveSamplingRate(0.01f)
        , minSamplesPerFile(10)
        , maxSamplesPerFile(1000)
        , useLocalFiles(true)
        , useGovdocs(true)
        , generateCorruptedSamples(true)
        , corruptedSampleRatio(0.3f)
        , minCorruptionSeverity(0.1f)
        , maxCorruptionSeverity(0.8f)
        , bitFlipRate(0.005f)
        , incrementalMode(false)
        , progressFilePath("")
        , appendToExistingCSV(false)
        , saveProgressInterval(100)
    {
        // 默认目标类型（分类模式）
        targetTypes = {
            "pdf", "doc", "xls", "ppt",
            "html", "txt", "xml",
            "jpg", "gif", "png"
        };

        // 默认启用的损坏类型
        enabledDamageTypes = {
            SimulatedDamageType::HEADER_ZEROED,
            SimulatedDamageType::HEADER_RANDOM,
            SimulatedDamageType::PARTIAL_OVERWRITE,
            SimulatedDamageType::RANDOM_CORRUPTION
        };

        // 默认启用的连续性负样本类型
        enabledSampleTypes = {
            ContinuitySampleType::DIFFERENT_FILES,
            ContinuitySampleType::FILE_BOUNDARY,
            ContinuitySampleType::RANDOM_DATA,
            ContinuitySampleType::DIFFERENT_TYPE
        };

        // 默认启用的损坏样本类型（连续性模式）
        enabledCorruptionTypes = {
            ContinuitySampleType::CORRUPTED_TRUNCATION,
            ContinuitySampleType::CORRUPTED_BITFLIP,
            ContinuitySampleType::CORRUPTED_ZERO_FILL,
            ContinuitySampleType::CORRUPTED_RANDOM_FILL,
            ContinuitySampleType::CORRUPTED_PARTIAL
        };
    }

    // 修复模式预设配置
    static DatasetGeneratorConfig RepairModePreset() {
        DatasetGeneratorConfig config;
        config.mode = DatasetMode::REPAIR;
        config.targetTypes = { "jpg", "jpeg", "png" };
        config.maxSamplesPerType = 5000;
        config.minFileSize = 8192;
        config.fragmentSize = 8192;
        config.generateDamagedSamples = true;
        config.damageRatio = 0.7f;
        return config;
    }

    // 连续性模式预设配置
    static DatasetGeneratorConfig ContinuityModePreset() {
        DatasetGeneratorConfig config;
        config.mode = DatasetMode::CONTINUITY;
        // 支持的文件类型:
        // - ZIP-based: zip, docx, xlsx, pptx, jar, apk
        // - Audio: mp3, wav, flac, ogg, m4a
        // - Video: mp4, mov, avi, mkv, webm
        // - Image: jpg, png, gif, bmp
        config.targetTypes = {
            // ZIP-based formats
            "zip", "docx", "xlsx", "pptx", "jar", "apk",
            // Audio formats
            "mp3", "wav", "flac", "ogg", "m4a",
            // Video formats
            "mp4", "mov", "avi", "mkv", "webm", "3gp",
            // Image formats
            "jpg", "jpeg", "png", "gif", "bmp"
        };
        config.maxSamplesPerType = 10000;
        config.minFileSize = 32768;           // 32KB 最小文件大小（至少4个块）
        config.continuityBlockSize = 8192;    // 8KB 块大小
        config.samplesPerFile = 10;           // 每个文件生成10个样本
        config.posNegRatio = 1.0f;            // 正负样本1:1
        config.useLocalFiles = true;
        config.useGovdocs = true;
        return config;
    }
};

// ============================================================================
// 二进制格式头部
// ============================================================================
#pragma pack(push, 1)
struct BinaryDatasetHeader {
    char magic[4];              // "MLFD"
    uint32_t version;           // 版本号 (1)
    uint32_t sampleCount;       // 样本数
    uint32_t featureDim;        // 特征维度 (261)
    uint32_t reserved[2];       // 保留字段

    BinaryDatasetHeader() {
        magic[0] = 'M'; magic[1] = 'L';
        magic[2] = 'F'; magic[3] = 'D';
        version = 1;
        sampleCount = 0;
        featureDim = FileFeatures::FEATURE_DIM;
        reserved[0] = reserved[1] = 0;
    }
};
#pragma pack(pop)

// ============================================================================
// 增量处理进度信息
// ============================================================================
struct IncrementalProgress {
    std::string sessionId;                      // 会话ID（用于标识同一批处理）
    std::string outputPath;                     // 输出CSV路径
    DatasetMode mode;                           // 数据集模式
    std::set<std::string> processedFiles;       // 已处理的文件路径（UTF-8）
    std::unordered_map<std::string, size_t> typeCounts;  // 每类样本计数
    size_t totalSamplesWritten;                 // 已写入的样本总数
    size_t totalFilesProcessed;                 // 已处理的文件总数
    size_t positiveSamples;                     // 正样本数（连续性模式）
    size_t negativeSamples;                     // 负样本数（连续性模式）
    std::string lastUpdateTime;                 // 最后更新时间
    bool csvHeaderWritten;                      // CSV头部是否已写入

    IncrementalProgress()
        : mode(DatasetMode::CONTINUITY)
        , totalSamplesWritten(0)
        , totalFilesProcessed(0)
        , positiveSamples(0)
        , negativeSamples(0)
        , csvHeaderWritten(false) {}
};

// ============================================================================
// 进度回调类型
// ============================================================================
using ProgressCallback = std::function<void(size_t current, size_t total, const std::string& status)>;

// ============================================================================
// 数据集生成器类
// ============================================================================
class DatasetGenerator {
public:
    DatasetGenerator(const DatasetGeneratorConfig& config = DatasetGeneratorConfig());
    ~DatasetGenerator();

    // 禁用拷贝构造和拷贝赋值
    DatasetGenerator(const DatasetGenerator&) = delete;
    DatasetGenerator& operator=(const DatasetGenerator&) = delete;

    // ==================== 扫描方法 ====================

    // 扫描指定目录
    bool ScanDirectory(const std::wstring& path);

    // 扫描多个目录
    bool ScanDirectories(const std::vector<std::wstring>& paths);

    // 扫描整个卷（使用文件系统遍历）
    bool ScanVolume(char driveLetter);

    // 停止扫描
    void Stop();

    // ==================== 导出方法 ====================

    // 导出为CSV格式（分类模式: 261维特征，修复模式: 31维+损坏信息）
    bool ExportCSV(const std::string& outputPath);

    // 导出为二进制格式
    bool ExportBinary(const std::string& outputPath);

    // 导出修复训练数据集（仅修复模式）
    bool ExportRepairCSV(const std::string& outputPath);

    // ==================== 状态查询 ====================

    // 获取统计信息
    DatasetStats GetStats() const;

    // 获取进度 (0-100)
    double GetProgress() const;

    // 检查是否正在运行
    bool IsRunning() const { return m_running.load(); }

    // 清空已收集的样本
    void Clear();

    // 设置进度回调
    void SetProgressCallback(ProgressCallback callback) { m_progressCallback = callback; }

    // ==================== 配置访问 ====================

    const DatasetGeneratorConfig& GetConfig() const { return m_config; }
    void SetConfig(const DatasetGeneratorConfig& config) { m_config = config; }

    // ==================== 修复模式方法 ====================

    // 获取收集的修复样本数量
    size_t GetRepairSampleCount() const;

    // 获取损坏类型名称
    static std::string GetDamageTypeName(SimulatedDamageType type);

    // ==================== 连续性模式方法 ====================

    // 生成连续性训练数据集
    bool GenerateContinuityDataset(const std::string& outputPath);

    // 扫描本地 ZIP 文件
    bool ScanLocalZipFiles(const std::vector<std::wstring>& directories);

    // 导出连续性训练数据集
    bool ExportContinuityCSV(const std::string& outputPath);

    // 获取收集的连续性样本数量
    size_t GetContinuitySampleCount() const;

    // 获取样本类型名称
    static std::string GetSampleTypeName(ContinuitySampleType type);

    // 计算自适应样本数
    size_t CalculateAdaptiveSamplesPerFile(size_t fileSize) const;

    // ==================== 增量处理方法 ====================

    // 加载进度文件（如果存在）
    bool LoadProgress(const std::string& progressPath);

    // 保存当前进度到文件
    bool SaveProgress(const std::string& progressPath);

    // 检查文件是否已处理
    bool IsFileProcessed(const std::wstring& filePath) const;

    // 标记文件为已处理
    void MarkFileProcessed(const std::wstring& filePath);

    // 获取当前进度信息
    const IncrementalProgress& GetIncrementalProgress() const { return m_progress; }

    // 追加写入CSV（增量模式使用）
    bool AppendToContinuityCSV(const std::string& outputPath);

    // 获取进度文件的默认路径
    static std::string GetDefaultProgressPath(const std::string& outputCsvPath);

private:
    // ==================== 内部方法（通用）====================

    // 处理单个文件（根据模式分派）
    void ProcessFile(const std::wstring& filePath, const std::string& extension);

    // 提取文件特征（分类模式）
    bool ExtractFileFeatures(const std::wstring& filePath, FileFeatures& outFeatures);

    // 工作线程函数
    void WorkerFunction();

    // 检查是否应该处理该文件类型
    bool ShouldProcessType(const std::string& ext) const;

    // 获取文件扩展名（小写）
    static std::string GetLowerExtension(const std::wstring& path);

    // 报告进度
    void ReportProgress(const std::string& status);

    // ==================== 内部方法（修复模式）====================

    // 处理单个图像文件（修复模式）
    void ProcessImageFile(const std::wstring& filePath, const std::string& extension);

    // 提取图像修复特征
    bool ExtractRepairFeatures(const std::vector<uint8_t>& data, ImageFeatureVector& outFeatures);

    // 模拟文件损坏
    void SimulateDamage(std::vector<uint8_t>& data, SimulatedDamageType type,
                        float severity, size_t& damageOffset, size_t& damageSize);

    // 生成随机损坏类型
    SimulatedDamageType GetRandomDamageType();

    // 生成随机严重程度
    float GetRandomSeverity();

    // 判断损坏后是否可修复
    bool EvaluateRepairability(const std::vector<uint8_t>& originalData,
                               const std::vector<uint8_t>& damagedData,
                               const std::string& imageType);

    // ==================== 内部方法（连续性模式）====================

    // 处理单个 ZIP 文件（连续性模式）
    void ProcessZipFileForContinuity(const std::wstring& filePath, const std::string& extension);

    // 从文件生成正样本（同一文件的连续块）
    void GeneratePositiveSamples(const std::wstring& filePath,
                                 const std::vector<uint8_t>& fileData,
                                 const std::string& fileType);

    // 从两个文件生成负样本（不同文件的块拼接）
    void GenerateNegativeSamples(const std::wstring& file1Path,
                                 const std::vector<uint8_t>& file1Data,
                                 const std::wstring& file2Path,
                                 const std::vector<uint8_t>& file2Data,
                                 const std::string& fileType);

    // 生成随机数据负样本
    void GenerateRandomNegativeSamples(const std::wstring& filePath,
                                       const std::vector<uint8_t>& fileData,
                                       const std::string& fileType);

    // 生成不同类型文件负样本
    void GenerateDifferentTypeNegativeSamples(const std::wstring& zipFilePath,
                                              const std::vector<uint8_t>& zipData,
                                              const std::wstring& otherFilePath,
                                              const std::vector<uint8_t>& otherData);

    // 获取随机负样本类型
    ContinuitySampleType GetRandomNegativeSampleType();

    // ==================== 内部方法（损坏样本生成）====================

    // 生成损坏样本（从正常连续块创建损坏的负样本）
    void GenerateCorruptedSamples(const std::wstring& filePath,
                                  const std::vector<uint8_t>& fileData,
                                  const std::string& fileType);

    // 应用截断损坏：块的后半部分被清零
    void ApplyTruncationCorruption(std::vector<uint8_t>& block, float severity,
                                   size_t& corruptionOffset, size_t& corruptionSize);

    // 应用比特翻转损坏：随机翻转一些比特
    void ApplyBitFlipCorruption(std::vector<uint8_t>& block, float severity,
                                size_t& corruptionOffset, size_t& corruptionSize);

    // 应用零填充损坏：随机区域被清零
    void ApplyZeroFillCorruption(std::vector<uint8_t>& block, float severity,
                                 size_t& corruptionOffset, size_t& corruptionSize);

    // 应用随机填充损坏：随机区域被随机数据覆盖
    void ApplyRandomFillCorruption(std::vector<uint8_t>& block, float severity,
                                   size_t& corruptionOffset, size_t& corruptionSize);

    // 应用部分覆盖损坏：随机区域被其他文件数据覆盖
    void ApplyPartialOverwriteCorruption(std::vector<uint8_t>& block,
                                         const std::vector<uint8_t>& otherData,
                                         float severity,
                                         size_t& corruptionOffset, size_t& corruptionSize);

    // 获取随机损坏类型
    ContinuitySampleType GetRandomCorruptionType();

    // 获取随机损坏严重程度
    float GetRandomCorruptionSeverity();

    // ==================== 成员变量 ====================

    DatasetGeneratorConfig m_config;

    // 样本存储（分类模式）
    std::vector<SampleInfo> m_samples;
    std::mutex m_samplesMutex;

    // 样本存储（修复模式）
    std::vector<RepairSampleInfo> m_repairSamples;
    std::mutex m_repairSamplesMutex;

    // 每类计数
    std::unordered_map<std::string, std::atomic<size_t>> m_typeCounts;

    // 统计
    std::atomic<size_t> m_totalScanned;
    std::atomic<size_t> m_skippedSmall;
    std::atomic<size_t> m_skippedError;
    std::atomic<size_t> m_skippedQuota;

    // 修复模式统计
    std::atomic<size_t> m_normalSamples;
    std::atomic<size_t> m_damagedSamples;
    std::atomic<size_t> m_repairableSamples;
    std::atomic<size_t> m_unrepairableSamples;
    std::unordered_map<int, std::atomic<size_t>> m_damageTypeCounts;

    // 样本存储（连续性模式）
    std::vector<ContinuitySampleInfo> m_continuitySamples;
    std::mutex m_continuitySamplesMutex;

    // 连续性模式统计
    std::atomic<size_t> m_positiveSamples;
    std::atomic<size_t> m_negativeSamples;
    std::unordered_map<int, std::atomic<size_t>> m_sampleTypeCounts;

    // 连续性模式缓存（用于生成负样本）
    std::vector<std::pair<std::wstring, std::vector<uint8_t>>> m_zipFileCache;
    std::vector<std::pair<std::wstring, std::vector<uint8_t>>> m_otherFileCache;
    std::mutex m_fileCacheMutex;

    // 线程池
    std::vector<std::thread> m_workers;
    std::queue<std::pair<std::wstring, std::string>> m_taskQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_taskAvailable;

    // 控制标志
    std::atomic<bool> m_running;
    std::atomic<bool> m_stopFlag;

    // 进度
    std::atomic<size_t> m_totalFiles;
    std::atomic<size_t> m_processedFiles;
    ProgressCallback m_progressCallback;

    // 计时
    LARGE_INTEGER m_startTime;
    LARGE_INTEGER m_frequency;

    // 随机数生成器（用于损坏模拟）
    std::mt19937 m_rng;
    std::mutex m_rngMutex;

    // 增量处理进度
    IncrementalProgress m_progress;
    std::mutex m_progressMutex;
};

} // namespace ML
