#include "OverwriteDetectionThreadPool.h"
#include "Logger.h"
#include <iostream>

using namespace std;

OverwriteDetectionThreadPool::OverwriteDetectionThreadPool(OverwriteDetector* det)
    : detector(det), stopFlag(false), completedTasks(0), totalTasks(0) {
    LOG_DEBUG("OverwriteDetectionThreadPool created");
}

OverwriteDetectionThreadPool::~OverwriteDetectionThreadPool() {
    Stop();
}

// 工作线程函数
void OverwriteDetectionThreadPool::WorkerThread() {
    LOG_DEBUG("Worker thread started");

    while (true) {
        DetectionTask task;

        {
            unique_lock<mutex> lock(queueMutex);
            condition.wait(lock, [this] {
                return stopFlag.load() || !taskQueue.empty();
            });

            if (stopFlag.load() && taskQueue.empty()) {
                LOG_DEBUG("Worker thread stopping");
                return;
            }

            if (taskQueue.empty()) {
                continue;
            }

            task = taskQueue.front();
            taskQueue.pop();
        }

        // 执行检测任务
        try {
            ClusterStatus status = detector->CheckCluster(task.clusterNumber);

            // 保存结果
            {
                lock_guard<mutex> lock(resultsMutex);
                results[task.taskId] = status;
            }

            completedTasks++;

            // 每完成100个任务输出一次进度
            if (completedTasks % 100 == 0) {
                LOG_DEBUG_FMT("Progress: %llu/%llu tasks completed (%.1f%%)",
                             completedTasks.load(), totalTasks.load(), GetProgress());
            }
        }
        catch (const exception& e) {
            LOG_ERROR_FMT("Exception in worker thread: %s", e.what());
        }
        catch (...) {
            LOG_ERROR("Unknown exception in worker thread");
        }
    }
}

// 执行多线程检测
vector<ClusterStatus> OverwriteDetectionThreadPool::DetectClusters(
    const vector<ULONGLONG>& clusterNumbers, int threadCount) {

    LOG_INFO_FMT("Starting multi-threaded detection with %d threads", threadCount);

    // 重置状态
    stopFlag = false;
    completedTasks = 0;
    totalTasks = clusterNumbers.size();
    results.clear();
    results.resize(clusterNumbers.size());

    // 启动工作线程
    workers.clear();
    for (int i = 0; i < threadCount; i++) {
        workers.emplace_back(&OverwriteDetectionThreadPool::WorkerThread, this);
    }

    // 分配任务
    {
        lock_guard<mutex> lock(queueMutex);
        for (size_t i = 0; i < clusterNumbers.size(); i++) {
            DetectionTask task;
            task.clusterNumber = clusterNumbers[i];
            task.taskId = i;
            taskQueue.push(task);
        }
    }
    condition.notify_all();

    LOG_INFO_FMT("Distributed %zu tasks to %d threads", clusterNumbers.size(), threadCount);

    // 等待所有任务完成
    while (completedTasks.load() < totalTasks.load()) {
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    // 通知所有线程停止
    stopFlag = true;
    condition.notify_all();

    // 等待所有线程退出
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers.clear();

    LOG_INFO_FMT("Multi-threaded detection completed: %llu tasks processed", completedTasks.load());

    return results;
}

// 获取进度
double OverwriteDetectionThreadPool::GetProgress() const {
    if (totalTasks == 0) return 0.0;
    return (double)completedTasks.load() / totalTasks.load() * 100.0;
}

// 停止所有线程
void OverwriteDetectionThreadPool::Stop() {
    if (stopFlag.load()) return;

    LOG_DEBUG("Stopping thread pool");

    stopFlag = true;
    condition.notify_all();

    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers.clear();
    LOG_DEBUG("Thread pool stopped");
}
