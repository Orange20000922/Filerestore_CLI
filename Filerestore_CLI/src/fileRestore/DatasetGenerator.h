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

namespace ML {

// ============================================================================
// 数据集生成模式
// ============================================================================
enum class DatasetMode {
    CLASSIFICATION,     // 分类模式: 261维特征，用于文件类型分类
    REPAIR              // 修复模式: 31维图像特征 + 损坏模拟
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


    DatasetStats() : totalSamples(0), totalFilesScanned(0),
                     skippedTooSmall(0), skippedReadError(0),
                     skippedQuotaReached(0), elapsedSeconds(0),
                     normalSamples(0), damagedSamples(0),
                     repairableSamples(0), unrepairableSamples(0) {}
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
        , incrementalMode(false)
        , appendToExistingCSV(false)
        , saveProgressInterval(100)
    {
        // 默认目标文件类型（分类模式）
        targetTypes = {
            "pdf", "doc", "xls", "ppt",
            "html", "txt", "xml",
            "jpg", "gif", "png"
        };

        // 默认启用所有损坏类型
        enabledDamageTypes = {
            SimulatedDamageType::HEADER_ZEROED,
            SimulatedDamageType::HEADER_RANDOM,
            SimulatedDamageType::PARTIAL_OVERWRITE,
            SimulatedDamageType::TRUNCATED,
            SimulatedDamageType::RANDOM_CORRUPTION
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
    std::string lastUpdateTime;                 // 最后更新时间
    bool csvHeaderWritten;                      // CSV头部是否已写入

    IncrementalProgress()
        : mode(DatasetMode::CLASSIFICATION)
        , totalSamplesWritten(0)
        , totalFilesProcessed(0)
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

    // ==================== 增量处理方法 ====================

    // 加载上次的处理进度
    bool LoadProgress(const std::string& progressPath);

    // 保存当前处理进度
    bool SaveProgress(const std::string& progressPath);

    // 检查文件是否已在之前的会话中处理
    bool IsFileProcessed(const std::wstring& filePath) const;

    // 标记文件为已处理
    void MarkFileProcessed(const std::wstring& filePath);

    // 获取默认进度文件路径
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
