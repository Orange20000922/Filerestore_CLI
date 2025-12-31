#pragma once
#include <Windows.h>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include "OverwriteDetector.h"

using namespace std;

// 检测任务
struct DetectionTask {
    ULONGLONG clusterNumber;
    int taskId;
};

// 线程池配置
struct ThreadPoolConfig {
    int threadCount;              // 线程数量
    bool enabled;                 // 是否启用多线程
    int minClustersForThreading;  // 启用多线程的最小簇数

    ThreadPoolConfig() : threadCount(4), enabled(true), minClustersForThreading(1000) {}
};

// 覆盖检测线程池
class OverwriteDetectionThreadPool
{
private:
    // 线程池状态
    vector<thread> workers;
    queue<DetectionTask> taskQueue;
    mutex queueMutex;
    condition_variable condition;
    atomic<bool> stopFlag;

    // 结果存储
    vector<ClusterStatus> results;
    mutex resultsMutex;

    // 进度跟踪
    atomic<ULONGLONG> completedTasks;
    atomic<ULONGLONG> totalTasks;

    // 检测器引用
    OverwriteDetector* detector;

    // 工作线程函数
    void WorkerThread();

public:
    OverwriteDetectionThreadPool(OverwriteDetector* det);
    ~OverwriteDetectionThreadPool();

    // 执行多线程检测
    vector<ClusterStatus> DetectClusters(const vector<ULONGLONG>& clusterNumbers, int threadCount);

    // 获取进度
    double GetProgress() const;

    // 停止所有线程
    void Stop();
};
