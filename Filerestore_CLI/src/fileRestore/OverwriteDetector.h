#pragma once
#include <Windows.h>
#include <vector>
#include <string>
#include <memory>
#include "MFTStructures.h"
#include "MFTReader.h"

using namespace std;

// 前向声明
class OverwriteDetectionThreadPool;

// 存储类型枚举
enum StorageType {
    STORAGE_UNKNOWN,
    STORAGE_HDD,        // 机械硬盘
    STORAGE_SSD,        // SATA SSD
    STORAGE_NVME        // NVMe SSD
};

// 检测模式枚举
enum DetectionMode {
    MODE_FAST,          // 快速模式：采样检测
    MODE_BALANCED,      // 平衡模式：智能检测
    MODE_THOROUGH       // 完整模式：检测所有簇
};

// 多线程策略枚举
enum ThreadingStrategy {
    THREADING_DISABLED,     // 禁用多线程
    THREADING_AUTO,         // 自动根据存储类型决定
    THREADING_ENABLED       // 强制启用多线程
};

// 簇状态信息
struct ClusterStatus {
    ULONGLONG clusterNumber;      // 簇号
    bool isOverwritten;           // 是否被覆盖
    bool isAllocated;             // 是否已分配给其他文件
    double dataEntropy;           // 数据熵值 (0.0-8.0)
    string overwriteReason;       // 覆盖原因描述
};

// 覆盖检测结果
struct OverwriteDetectionResult {
    vector<ClusterStatus> clusterStatuses;  // 每个簇的状态
    ULONGLONG totalClusters;                // 总簇数
    ULONGLONG overwrittenClusters;          // 被覆盖的簇数
    ULONGLONG availableClusters;            // 可用的簇数
    ULONGLONG sampledClusters;              // 实际检测的簇数（采样模式）
    double overwritePercentage;             // 覆盖百分比 (0.0-100.0)
    bool isFullyAvailable;                  // 数据是否完全可用
    bool isPartiallyAvailable;              // 数据是否部分可用
    bool usedSampling;                      // 是否使用了采样
    bool usedMultiThreading;                // 是否使用了多线程
    int threadCount;                        // 使用的线程数
    StorageType detectedStorageType;        // 检测到的存储类型
    double detectionTimeMs;                 // 检测耗时（毫秒）
};

// 数据覆盖检测器类
class OverwriteDetector
{
private:
    MFTReader* reader;
    DWORD bytesPerCluster;

    // 检测配置
    DetectionMode detectionMode;
    ThreadingStrategy threadingStrategy;
    StorageType cachedStorageType;
    bool storageTypeDetected;

    // 线程池
    std::unique_ptr<OverwriteDetectionThreadPool> threadPool;

    // 内部辅助函数
    double CalculateEntropy(const vector<BYTE>& data);
    double CalculateEntropyFromPointer(const BYTE* data, size_t size);
    bool IsAllZeros(const vector<BYTE>& data);
    bool IsAllZerosFromPointer(const BYTE* data, size_t size);
    bool IsAllSameValue(const vector<BYTE>& data);
    bool IsAllSameValueFromPointer(const BYTE* data, size_t size);
    bool HasValidFileStructure(const vector<BYTE>& data);
    bool HasValidFileStructureFromPointer(const BYTE* data, size_t size);
    bool CheckClusterAllocation(ULONGLONG clusterNumber);

    // 解析Data Runs的辅助函数
    ULONGLONG ReadVariableLength(const BYTE* data, int length);
    LONGLONG ReadVariableLengthSigned(const BYTE* data, int length);

    // 存储类型检测
    StorageType DetectStorageType();
    double MeasureRandomReadLatency();
    string GetStorageTypeName(StorageType type);

    // 批量处理
    ClusterStatus CheckClusterFromMemory(const BYTE* clusterData, ULONGLONG clusterNumber);
    vector<ClusterStatus> BatchCheckClusters(const vector<pair<ULONGLONG, ULONGLONG>>& dataRuns);

    // 采样检测
    vector<ClusterStatus> SamplingCheckClusters(const vector<pair<ULONGLONG, ULONGLONG>>& dataRuns,
                                                ULONGLONG totalClusters);

    // 多线程检测
    vector<ClusterStatus> MultiThreadedCheckClusters(const vector<ULONGLONG>& clusterNumbers,
                                                     int threadCount);

    // 智能跳过
    bool ShouldSkipRemaining(const vector<ClusterStatus>& results);

    // 智能策略选择
    bool ShouldUseMultiThreading(ULONGLONG clusterCount, StorageType storageType);
    int GetOptimalThreadCount(ULONGLONG clusterCount, StorageType storageType);

public:
    OverwriteDetector(MFTReader* mftReader);
    ~OverwriteDetector();

    // 设置检测模式
    void SetDetectionMode(DetectionMode mode) { detectionMode = mode; }
    DetectionMode GetDetectionMode() const { return detectionMode; }

    // 设置多线程策略
    void SetThreadingStrategy(ThreadingStrategy strategy) { threadingStrategy = strategy; }
    ThreadingStrategy GetThreadingStrategy() const { return threadingStrategy; }

    // 获取存储类型
    StorageType GetStorageType();

    // 主要功能：检测文件数据是否被覆盖（优化版本）
    OverwriteDetectionResult DetectOverwrite(const vector<BYTE>& mftRecord);

    // 检测单个簇是否被覆盖
    ClusterStatus CheckCluster(ULONGLONG clusterNumber);

    // 从MFT记录中提取Data Runs
    bool ExtractDataRuns(const vector<BYTE>& mftRecord,
                        vector<pair<ULONGLONG, ULONGLONG>>& runs);

    // 批量检测多个簇
    vector<ClusterStatus> CheckClusters(const vector<ULONGLONG>& clusterNumbers);

    // 获取可读的覆盖检测报告
    string GetDetectionReport(const OverwriteDetectionResult& result);
};
