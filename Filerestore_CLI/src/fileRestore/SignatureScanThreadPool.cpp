#include "SignatureScanThreadPool.h"
#include "FileCarver.h"
#include "Logger.h"
#include <iostream>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace std::chrono;

// ============================================================================
// 构造函数
// ============================================================================
SignatureScanThreadPool::SignatureScanThreadPool(
    const unordered_map<BYTE, vector<const FileSignature*>>* sigIndex,
    const set<string>* activeSigs,
    const ScanThreadPoolConfig& cfg)
    : signatureIndex(sigIndex)
    , activeSignatures(activeSigs)
    , config(cfg)
    , stopFlag(false)
    , pauseFlag(false)
    , completedTasks(0)
    , totalTasks(0)
    , totalFilesFound(0)
    , totalBytesScanned(0)
{
    // 如果启用自动检测，计算最优线程数
    if (config.autoDetectThreads) {
        config.workerCount = GetOptimalThreadCount();
    }

    LOG_INFO_FMT("SignatureScanThreadPool created with %d workers, chunk size: %zu MB",
                 config.workerCount, config.chunkSize / (1024 * 1024));
}

// ============================================================================
// 析构函数
// ============================================================================
SignatureScanThreadPool::~SignatureScanThreadPool() {
    Stop();
}

// ============================================================================
// 启动线程池
// ============================================================================
void SignatureScanThreadPool::Start() {
    if (!workers.empty()) {
        LOG_WARNING("Thread pool already started");
        return;
    }

    stopFlag = false;
    pauseFlag = false;
    completedTasks = 0;
    totalTasks = 0;

    LOG_INFO_FMT("Starting thread pool with %d workers", config.workerCount);

    for (int i = 0; i < config.workerCount; ++i) {
        workers.emplace_back(&SignatureScanThreadPool::WorkerFunction, this);
    }

    LOG_INFO("Thread pool started successfully");
}

// ============================================================================
// 停止线程池
// ============================================================================
void SignatureScanThreadPool::Stop() {
    if (stopFlag.load()) return;

    LOG_DEBUG("Stopping thread pool...");

    stopFlag = true;
    pauseFlag = false;

    // 唤醒所有等待的线程
    taskAvailable.notify_all();
    queueNotFull.notify_all();

    // 等待所有线程完成
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers.clear();
    LOG_DEBUG("Thread pool stopped");
}

// ============================================================================
// 暂停/恢复
// ============================================================================
void SignatureScanThreadPool::Pause() {
    pauseFlag = true;
    LOG_DEBUG("Thread pool paused");
}

void SignatureScanThreadPool::Resume() {
    pauseFlag = false;
    taskAvailable.notify_all();
    LOG_DEBUG("Thread pool resumed");
}

// ============================================================================
// 工作线程主函数
// ============================================================================
void SignatureScanThreadPool::WorkerFunction() {
    LOG_DEBUG("Worker thread started");

    while (true) {
        ScanTask task;

        // 从队列获取任务
        {
            unique_lock<mutex> lock(queueMutex);

            // 等待任务或停止信号
            taskAvailable.wait(lock, [this] {
                return stopFlag.load() || (!taskQueue.empty() && !pauseFlag.load());
            });

            // 检查停止条件
            if (stopFlag.load() && taskQueue.empty()) {
                LOG_DEBUG("Worker thread exiting");
                return;
            }

            // 如果暂停，继续等待
            if (pauseFlag.load()) {
                continue;
            }

            // 如果队列为空，继续等待
            if (taskQueue.empty()) {
                continue;
            }

            // 获取任务
            task = taskQueue.front();
            taskQueue.pop();
        }

        // 通知可能在等待队列空位的生产者
        queueNotFull.notify_one();

        // 执行扫描任务
        vector<CarvedFileInfo> localResults;

        try {
            ScanChunk(task, localResults);
        }
        catch (const exception& e) {
            LOG_ERROR_FMT("Exception in worker thread: %s", e.what());
        }
        catch (...) {
            LOG_ERROR("Unknown exception in worker thread");
        }

        // 保存结果
        {
            lock_guard<mutex> lock(resultsMutex);
            ScanTaskResult result;
            result.taskId = task.taskId;
            result.files = move(localResults);
            result.bytesScanned = task.dataSize;
            result.filesFound = result.files.size();
            results.push_back(move(result));
        }

        // 更新统计
        totalFilesFound += localResults.size();
        totalBytesScanned += task.dataSize;
        completedTasks++;

        // 定期输出进度日志
        int completed = completedTasks.load();
        if (completed % 10 == 0) {
            LOG_DEBUG_FMT("Progress: %d/%d tasks (%.1f%%)",
                         completed, totalTasks.load(), GetProgress());
        }
    }
}

// ============================================================================
// 核心扫描逻辑（扫描单个数据块）
// ============================================================================
void SignatureScanThreadPool::ScanChunk(const ScanTask& task,
                                         vector<CarvedFileInfo>& localResults) {
    if (!task.data || task.dataSize == 0) return;

    const BYTE* data = task.data;
    size_t dataSize = task.dataSize;
    size_t offset = 0;

    while (offset < dataSize && !stopFlag.load()) {
        BYTE currentByte = data[offset];

        // 使用首字节索引快速查找可能匹配的签名
        auto it = signatureIndex->find(currentByte);
        if (it != signatureIndex->end()) {
            // 检查该首字节对应的所有签名
            for (const FileSignature* sig : it->second) {
                // 检查是否在活动签名列表中
                if (activeSignatures->find(sig->extension) == activeSignatures->end()) {
                    continue;
                }

                size_t remaining = dataSize - offset;
                if (remaining < sig->header.size()) {
                    continue;
                }

                // 完整签名匹配
                if (MatchSignature(data + offset, remaining, sig->header)) {
                    // 特殊检查：区分 AVI 和 WAV (都是 RIFF)
                    if (sig->extension == "avi" || sig->extension == "wav") {
                        if (remaining >= 12) {
                            bool isAvi = (data[offset + 8] == 'A' && data[offset + 9] == 'V' &&
                                          data[offset + 10] == 'I');
                            bool isWav = (data[offset + 8] == 'W' && data[offset + 9] == 'A' &&
                                          data[offset + 10] == 'V' && data[offset + 11] == 'E');
                            if ((sig->extension == "avi" && !isAvi) ||
                                (sig->extension == "wav" && !isWav)) {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }

                    // 特殊检查：MP4 需要验证 ftyp
                    if (sig->extension == "mp4" && remaining >= 8) {
                        if (!(data[offset + 4] == 'f' && data[offset + 5] == 't' &&
                              data[offset + 6] == 'y' && data[offset + 7] == 'p')) {
                            continue;
                        }
                    }

                    // 估算文件大小
                    ULONGLONG estimatedSize = EstimateFileSize(data + offset, remaining, *sig);

                    // 验证文件
                    double confidence = ValidateFile(data + offset,
                                                     min(remaining, (size_t)sig->maxSize),
                                                     *sig);

                    // 只添加置信度足够的结果
                    if (confidence >= 0.6) {
                        ULONGLONG absoluteLCN = task.baseLCN + (offset / task.bytesPerCluster);
                        ULONGLONG clusterOffset = offset % task.bytesPerCluster;

                        CarvedFileInfo info;
                        info.startLCN = absoluteLCN;
                        info.startOffset = clusterOffset;
                        info.fileSize = estimatedSize;
                        info.extension = sig->extension;
                        info.description = sig->description;
                        info.hasValidFooter = false;  // 简化版不查找footer
                        info.confidence = confidence;

                        localResults.push_back(info);

                        // 跳过当前文件区域（避免重复检测）
                        offset += (size_t)min(estimatedSize, (ULONGLONG)remaining);
                        goto next_position;
                    }
                }
            }
        }

        offset++;
        next_position:;
    }
}

// ============================================================================
// 签名匹配
// ============================================================================
bool SignatureScanThreadPool::MatchSignature(const BYTE* data, size_t dataSize,
                                              const vector<BYTE>& signature) {
    if (dataSize < signature.size()) {
        return false;
    }
    return memcmp(data, signature.data(), signature.size()) == 0;
}

// ============================================================================
// 估算文件大小（简化版）
// ============================================================================
ULONGLONG SignatureScanThreadPool::EstimateFileSize(const BYTE* data, size_t dataSize,
                                                     const FileSignature& sig) {
    // 特殊格式处理
    if (sig.extension == "bmp" && dataSize >= 6) {
        DWORD size = *(DWORD*)(data + 2);
        if (size > sig.minSize && size <= sig.maxSize && size <= dataSize) {
            return size;
        }
    }

    if (sig.extension == "avi" && dataSize >= 12) {
        if (data[8] == 'A' && data[9] == 'V' && data[10] == 'I' && data[11] == ' ') {
            DWORD riffSize = *(DWORD*)(data + 4);
            if (riffSize > 0 && riffSize <= sig.maxSize) {
                return min((ULONGLONG)riffSize + 8, (ULONGLONG)dataSize);
            }
        }
    }

    if (sig.extension == "wav" && dataSize >= 12) {
        if (data[8] == 'W' && data[9] == 'A' && data[10] == 'V' && data[11] == 'E') {
            DWORD riffSize = *(DWORD*)(data + 4);
            if (riffSize > 0 && riffSize <= sig.maxSize) {
                return min((ULONGLONG)riffSize + 8, (ULONGLONG)dataSize);
            }
        }
    }

    // 默认：使用保守估计（可用数据大小的一部分或最大大小）
    return min((ULONGLONG)dataSize, sig.maxSize);
}

// ============================================================================
// 文件验证（简化版）
// ============================================================================
double SignatureScanThreadPool::ValidateFile(const BYTE* data, size_t dataSize,
                                              const FileSignature& sig) {
    double confidence = 0.8;  // 基础置信度（签名已匹配）

    // 特定格式额外验证
    if (sig.extension == "jpg" && dataSize >= 10) {
        if ((data[3] == 0xE0 || data[3] == 0xE1) && data[6] == 0x4A) {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "png" && dataSize >= 24) {
        if (data[12] == 'I' && data[13] == 'H' && data[14] == 'D' && data[15] == 'R') {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "pdf" && dataSize >= 20) {
        if (data[5] == '-' && data[6] >= '1' && data[6] <= '9') {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "zip" && dataSize >= 30) {
        WORD version = *(WORD*)(data + 4);
        WORD flags = *(WORD*)(data + 6);
        if (version <= 63 && (flags & 0xFF00) == 0) {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "mp4" && dataSize >= 12) {
        if (data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p') {
            confidence += 0.15;
        }
    }

    return min(1.0, confidence);
}

// ============================================================================
// 提交任务（阻塞式）
// ============================================================================
void SignatureScanThreadPool::SubmitTask(const ScanTask& task) {
    {
        unique_lock<mutex> lock(queueMutex);

        // 等待队列有空位
        queueNotFull.wait(lock, [this] {
            return taskQueue.size() < config.maxQueueSize || stopFlag.load();
        });

        if (stopFlag.load()) return;

        taskQueue.push(task);
        totalTasks++;
    }

    taskAvailable.notify_one();
}

// ============================================================================
// 尝试提交任务（非阻塞式）
// ============================================================================
bool SignatureScanThreadPool::TrySubmitTask(const ScanTask& task) {
    {
        lock_guard<mutex> lock(queueMutex);

        if (taskQueue.size() >= config.maxQueueSize) {
            return false;
        }

        taskQueue.push(task);
        totalTasks++;
    }

    taskAvailable.notify_one();
    return true;
}

// ============================================================================
// 等待所有任务完成
// ============================================================================
void SignatureScanThreadPool::WaitForCompletion() {
    while (completedTasks.load() < totalTasks.load()) {
        this_thread::sleep_for(chrono::milliseconds(10));
    }
}

// ============================================================================
// 获取合并后的结果
// ============================================================================
vector<CarvedFileInfo> SignatureScanThreadPool::GetMergedResults() {
    lock_guard<mutex> lock(resultsMutex);

    // 按 taskId 排序以保持顺序
    sort(results.begin(), results.end(),
         [](const ScanTaskResult& a, const ScanTaskResult& b) {
             return a.taskId < b.taskId;
         });

    // 合并所有结果
    vector<CarvedFileInfo> merged;
    for (const auto& r : results) {
        merged.insert(merged.end(), r.files.begin(), r.files.end());
    }

    return merged;
}

// ============================================================================
// 清空结果
// ============================================================================
void SignatureScanThreadPool::ClearResults() {
    lock_guard<mutex> lock(resultsMutex);
    results.clear();
    completedTasks = 0;
    totalTasks = 0;
    totalFilesFound = 0;
    totalBytesScanned = 0;
}

// ============================================================================
// 获取进度
// ============================================================================
double SignatureScanThreadPool::GetProgress() const {
    int total = totalTasks.load();
    if (total == 0) return 0.0;
    return (double)completedTasks.load() / total * 100.0;
}

// ============================================================================
// 获取最优线程数
// ============================================================================
int SignatureScanThreadPool::GetOptimalThreadCount() {
    int cores = ThreadPoolUtils::GetLogicalCoreCount();

    // 对于高端CPU（16线程及以上），使用 cores - 4
    // 对于中端CPU（8-15线程），使用 cores - 2
    // 对于低端CPU（4-7线程），使用 cores - 1
    // 最少使用2个线程

    int optimalCount;
    if (cores >= 16) {
        optimalCount = cores - 4;  // 留4个给系统和I/O
    } else if (cores >= 8) {
        optimalCount = cores - 2;
    } else if (cores >= 4) {
        optimalCount = cores - 1;
    } else {
        optimalCount = max(2, cores);
    }

    LOG_INFO_FMT("Detected %d logical cores, using %d worker threads", cores, optimalCount);
    return optimalCount;
}

// ============================================================================
// 辅助函数实现
// ============================================================================
namespace ThreadPoolUtils {

int GetLogicalCoreCount() {
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    int cores = (int)sysInfo.dwNumberOfProcessors;
    return (cores > 0) ? cores : 4;  // 默认4核
}

int GetPhysicalCoreCount() {
    // 简化实现：假设超线程比例为2:1
    return max(1, GetLogicalCoreCount() / 2);
}

ULONGLONG GetAvailableMemoryMB() {
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus)) {
        return memStatus.ullAvailPhys / (1024 * 1024);
    }
    return 4096;  // 默认4GB
}

ScanThreadPoolConfig GetOptimalConfig() {
    ScanThreadPoolConfig config;

    int cores = GetLogicalCoreCount();
    ULONGLONG availMem = GetAvailableMemoryMB();

    // 根据CPU核心数设置工作线程
    if (cores >= 16) {
        config.workerCount = 12;
        config.chunkSize = 8 * 1024 * 1024;   // 8MB
        config.maxQueueSize = 48;
    } else if (cores >= 8) {
        config.workerCount = 6;
        config.chunkSize = 8 * 1024 * 1024;   // 8MB
        config.maxQueueSize = 24;
    } else if (cores >= 4) {
        config.workerCount = 3;
        config.chunkSize = 4 * 1024 * 1024;   // 4MB
        config.maxQueueSize = 12;
    } else {
        config.workerCount = 2;
        config.chunkSize = 4 * 1024 * 1024;   // 4MB
        config.maxQueueSize = 8;
    }

    // 根据可用内存调整队列大小
    // 每个任务最多占用 chunkSize 内存
    ULONGLONG maxMemoryForQueue = availMem / 4;  // 最多使用25%可用内存
    size_t maxQueueByMem = (size_t)(maxMemoryForQueue * 1024 * 1024 / config.chunkSize);
    config.maxQueueSize = min(config.maxQueueSize, maxQueueByMem);
    config.maxQueueSize = max(config.maxQueueSize, (size_t)4);  // 至少4个任务

    LOG_INFO_FMT("Optimal config: %d workers, %zu MB chunks, %zu max queue (Cores: %d, Mem: %llu MB)",
                 config.workerCount, config.chunkSize / (1024 * 1024),
                 config.maxQueueSize, cores, availMem);

    return config;
}

} // namespace ThreadPoolUtils
