#pragma once
#include <Windows.h>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <set>
#include <unordered_map>
#include <functional>
#include <string>

using namespace std;

// 前向声明
struct FileSignature;
struct CarvedFileInfo;

// ============================================================================
// 扫描任务结构
// ============================================================================
struct ScanTask {
    const BYTE* data;               // 数据指针（不拥有数据）
    size_t dataSize;                // 数据大小
    ULONGLONG baseLCN;              // 基准逻辑簇号
    ULONGLONG bytesPerCluster;      // 每簇字节数
    int taskId;                     // 任务ID（用于结果排序）
};

// ============================================================================
// 任务执行结果
// ============================================================================
struct ScanTaskResult {
    int taskId;                     // 任务ID
    vector<CarvedFileInfo> files;   // 发现的文件
    ULONGLONG bytesScanned;         // 扫描的字节数
    ULONGLONG filesFound;           // 发现的文件数
};

// ============================================================================
// 线程池配置
// ============================================================================
struct ScanThreadPoolConfig {
    int workerCount;                // 工作线程数
    size_t chunkSize;               // 每个任务的数据块大小
    size_t maxQueueSize;            // 任务队列最大长度
    bool autoDetectThreads;         // 自动检测最优线程数

    // 默认配置（针对 NVMe + 16线程 CPU 优化）
    ScanThreadPoolConfig()
        : workerCount(12)           // 留4线程给系统和I/O
        , chunkSize(8 * 1024 * 1024)// 8MB per chunk
        , maxQueueSize(32)          // 最多32个待处理任务
        , autoDetectThreads(true)
    {}
};

// ============================================================================
// 签名扫描线程池
// ============================================================================
class SignatureScanThreadPool {
private:
    // ==================== 线程管理 ====================
    vector<thread> workers;
    atomic<bool> stopFlag;
    atomic<bool> pauseFlag;

    // ==================== 任务队列 ====================
    queue<ScanTask> taskQueue;
    mutex queueMutex;
    condition_variable taskAvailable;       // 通知有新任务
    condition_variable queueNotFull;        // 通知队列有空位

    // ==================== 结果收集 ====================
    vector<ScanTaskResult> results;
    mutex resultsMutex;

    // ==================== 共享只读数据 ====================
    const unordered_map<BYTE, vector<const FileSignature*>>* signatureIndex;
    const set<string>* activeSignatures;

    // ==================== 统计信息 ====================
    atomic<int> completedTasks;
    atomic<int> totalTasks;
    atomic<ULONGLONG> totalFilesFound;
    atomic<ULONGLONG> totalBytesScanned;

    // ==================== 配置 ====================
    ScanThreadPoolConfig config;

    // ==================== 私有方法 ====================

    // 工作线程主函数
    void WorkerFunction();

    // 扫描单个数据块（核心扫描逻辑）
    void ScanChunk(const ScanTask& task, vector<CarvedFileInfo>& localResults);

    // 匹配签名
    bool MatchSignature(const BYTE* data, size_t dataSize, const vector<BYTE>& signature);

    // 估算文件大小
    ULONGLONG EstimateFileSize(const BYTE* data, size_t dataSize, const FileSignature& sig);

    // 查找文件尾
    ULONGLONG FindFooter(const BYTE* data, size_t dataSize,
                         const vector<BYTE>& footer, ULONGLONG maxSearch);

    // 验证文件有效性
    double ValidateFile(const BYTE* data, size_t dataSize, const FileSignature& sig);

public:
    // ==================== 构造和析构 ====================
    SignatureScanThreadPool(
        const unordered_map<BYTE, vector<const FileSignature*>>* sigIndex,
        const set<string>* activeSigs,
        const ScanThreadPoolConfig& cfg = ScanThreadPoolConfig());

    ~SignatureScanThreadPool();

    // ==================== 线程池控制 ====================

    // 启动线程池
    void Start();

    // 停止线程池
    void Stop();

    // 暂停/恢复
    void Pause();
    void Resume();

    // ==================== 任务管理 ====================

    // 提交扫描任务（阻塞式，队列满时等待）
    void SubmitTask(const ScanTask& task);

    // 尝试提交任务（非阻塞式，队列满时返回false）
    bool TrySubmitTask(const ScanTask& task);

    // 等待所有任务完成
    void WaitForCompletion();

    // 获取合并后的结果（按taskId排序）
    vector<CarvedFileInfo> GetMergedResults();

    // 清空结果
    void ClearResults();

    // ==================== 状态查询 ====================

    // 获取进度 (0.0 - 100.0)
    double GetProgress() const;

    // 获取统计信息
    int GetCompletedTasks() const { return completedTasks.load(); }
    int GetTotalTasks() const { return totalTasks.load(); }
    int GetPendingTasks() const { return totalTasks.load() - completedTasks.load(); }
    ULONGLONG GetTotalFilesFound() const { return totalFilesFound.load(); }
    ULONGLONG GetTotalBytesScanned() const { return totalBytesScanned.load(); }

    // 检查是否正在运行
    bool IsRunning() const { return !stopFlag.load(); }
    bool IsPaused() const { return pauseFlag.load(); }

    // ==================== 配置 ====================

    // 获取配置
    const ScanThreadPoolConfig& GetConfig() const { return config; }

    // 获取最优线程数（基于硬件检测）
    static int GetOptimalThreadCount();

    // 获取工作线程数
    int GetWorkerCount() const { return config.workerCount; }
};

// ============================================================================
// 辅助函数：获取系统信息
// ============================================================================
namespace ThreadPoolUtils {
    // 获取CPU逻辑核心数
    int GetLogicalCoreCount();

    // 获取CPU物理核心数（近似）
    int GetPhysicalCoreCount();

    // 获取可用内存大小（MB）
    ULONGLONG GetAvailableMemoryMB();

    // 根据硬件自动计算最优配置
    ScanThreadPoolConfig GetOptimalConfig();
}
