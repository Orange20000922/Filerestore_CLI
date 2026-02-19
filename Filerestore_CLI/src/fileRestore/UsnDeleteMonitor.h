#pragma once
#include <Windows.h>
#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include "UsnJournalReader.h"
#include "MFTReader.h"
#include "MFTParser.h"
#include "MFTSnapshotStore.h"

using namespace std;

// ============================================================================
// USN 删除监控统计
// ============================================================================
struct UsnMonitorStats {
    atomic<size_t> totalEvents{0};      // 总检测到的删除事件
    atomic<size_t> capturedCount{0};    // 成功捕获快照数
    atomic<size_t> missedCount{0};      // MFT 已复用来不及捕获数
    atomic<size_t> skippedCount{0};     // 跳过数（目录、系统文件等）
};

// ============================================================================
// USN 删除监控器 - 后台轮询 USN Journal，捕获被删文件的 MFT 快照
// ============================================================================
class UsnDeleteMonitor {
public:
    UsnDeleteMonitor(char driveLetter);
    ~UsnDeleteMonitor();

    // ========== 控制 ==========

    // 启动后台监控
    bool Start();

    // 停止监控
    void Stop();

    // 是否正在运行
    bool IsRunning() const { return running.load(); }

    // ========== 快照访问 ==========

    // 获取快照存储引用
    MFTSnapshotStore& GetSnapshotStore() { return snapshotStore; }
    const MFTSnapshotStore& GetSnapshotStore() const { return snapshotStore; }

    // ========== 一次性扫描 ==========

    // 扫描 USN 日志中已有的删除记录并捕获快照（非后台，立即执行）
    // maxTimeHours: 回溯时间（小时）
    // 返回捕获的快照数
    size_t CaptureExistingDeleted(int maxTimeHours = 24);

    // ========== 统计 ==========

    const UsnMonitorStats& GetStats() const { return stats; }

    // ========== 配置 ==========

    // 轮询间隔（毫秒），默认 1000ms
    void SetPollInterval(DWORD ms) { pollIntervalMs = ms; }
    DWORD GetPollInterval() const { return pollIntervalMs; }

    // 自动保存间隔（秒），0 = 不自动保存，默认 60s
    void SetAutoSaveInterval(DWORD seconds) { autoSaveIntervalSec = seconds; }

    // ========== 事件回调 ==========

    using EventCallback = function<void(ULONGLONG recordNum, ULONGLONG fileSize,
                                         FILETIME deleteTime, const wstring& fileName,
                                         bool captured)>;
    void SetEventCallback(EventCallback cb) { eventCallback = std::move(cb); }

private:
    // 后台轮询线程
    void MonitorThread();

    // 处理单个删除事件：读取 MFT 记录，创建快照
    bool HandleDeleteEvent(const UsnDeletedFileInfo& info);

    // 组件
    char driveLetter;
    MFTSnapshotStore snapshotStore;

    // 线程控制
    thread monitorThread;
    atomic<bool> running;

    // USN 游标
    USN lastProcessedUsn;

    // 配置
    DWORD pollIntervalMs;
    DWORD autoSaveIntervalSec;

    // 统计
    UsnMonitorStats stats;

    // 事件回调
    EventCallback eventCallback;
};
