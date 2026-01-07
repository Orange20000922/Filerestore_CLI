#pragma once
/**
 * @file DatasetGenerator.h
 * @brief ML Dataset Generator - 从本地文件系统提取特征生成训练数据集
 *
 * 用于扫描指定目录/卷，提取文件特征并导出为CSV或二进制格式，
 * 供Python端训练ML模型使用。
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
#include "MLClassifier.h"

namespace ML {

// ============================================================================
// 数据集样本信息
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

    DatasetStats() : totalSamples(0), totalFilesScanned(0),
                     skippedTooSmall(0), skippedReadError(0),
                     skippedQuotaReached(0), elapsedSeconds(0) {}
};

// ============================================================================
// 数据集生成器配置
// ============================================================================
struct DatasetGeneratorConfig {
    std::set<std::string> targetTypes;      // 目标文件类型
    size_t maxSamplesPerType;               // 每类最大样本数
    size_t minFileSize;                     // 最小文件大小
    size_t fragmentSize;                    // 特征提取片段大小
    int workerThreads;                      // 工作线程数
    bool includeFilePath;                   // CSV中是否包含文件路径
    bool verbose;                           // 详细输出

    DatasetGeneratorConfig()
        : maxSamplesPerType(2000)
        , minFileSize(4096)
        , fragmentSize(4096)
        , workerThreads(8)
        , includeFilePath(false)
        , verbose(true)
    {
        // 默认目标类型（与Python训练脚本一致）
        targetTypes = {
            "pdf", "doc", "xls", "ppt",
            "html", "txt", "xml",
            "jpg", "gif", "png"
        };
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

    // 导出为CSV格式
    bool ExportCSV(const std::string& outputPath);

    // 导出为二进制格式
    bool ExportBinary(const std::string& outputPath);

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

private:
    // ==================== 内部方法 ====================

    // 处理单个文件
    void ProcessFile(const std::wstring& filePath, const std::string& extension);

    // 提取文件特征
    bool ExtractFileFeatures(const std::wstring& filePath, FileFeatures& outFeatures);

    // 工作线程函数
    void WorkerFunction();

    // 检查是否应该处理该文件类型
    bool ShouldProcessType(const std::string& ext) const;

    // 获取文件扩展名（小写）
    static std::string GetLowerExtension(const std::wstring& path);

    // 报告进度
    void ReportProgress(const std::string& status);

    // ==================== 成员变量 ====================

    DatasetGeneratorConfig m_config;

    // 样本存储
    std::vector<SampleInfo> m_samples;
    std::mutex m_samplesMutex;

    // 每类计数
    std::unordered_map<std::string, std::atomic<size_t>> m_typeCounts;

    // 统计
    std::atomic<size_t> m_totalScanned;
    std::atomic<size_t> m_skippedSmall;
    std::atomic<size_t> m_skippedError;
    std::atomic<size_t> m_skippedQuota;

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
};

} // namespace ML
