#include "MonitorDaemon.h"
#include "UsnDeleteMonitor.h"
#include "MFTSnapshotStore.h"
#include "Logger.h"
#include <iostream>

using namespace std;

// ============================================================================
// seqlock 辅助函数（跨进程安全）
// ============================================================================

// 写端互斥：获取（保证同一时刻只有一个线程进入 seqlock 写区间）
static void WriterLockAcquire(volatile LONG* lock) {
    while (InterlockedCompareExchange(lock, 1, 0) != 0) {
        YieldProcessor();
    }
}

// 写端互斥：释放
static void WriterLockRelease(volatile LONG* lock) {
    InterlockedExchange(lock, 0);
}

// 写端：开始写入（seqCounter 变为奇数）
static void SeqLockWriteBegin(volatile LONG* seq) {
    InterlockedIncrement(seq);
    MemoryBarrier();
}

// 写端：结束写入（seqCounter 变为偶数）
static void SeqLockWriteEnd(volatile LONG* seq) {
    MemoryBarrier();
    InterlockedIncrement(seq);
}

// 读端：读取序列号（偶数时数据稳定），短暂自旋等待写入完成
static LONG SeqLockReadBegin(volatile LONG* seq) {
    LONG s;
    int spin = 0;
    while ((s = InterlockedCompareExchange(seq, 0, 0)) & 1) {
        if (++spin > 128) return s;  // 短暂自旋后返回，由调用者判断
        YieldProcessor();
    }
    MemoryBarrier();
    return s;
}

// 检查指定 PID 的进程是否仍在运行
static bool IsProcessAlive(DWORD pid) {
    if (pid == 0) return false;
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!hProcess) return false;
    DWORD exitCode;
    BOOL ok = GetExitCodeProcess(hProcess, &exitCode);
    CloseHandle(hProcess);
    return ok && exitCode == STILL_ACTIVE;
}

// 读端：校验序列号未变化
static bool SeqLockReadValidate(volatile LONG* seq, LONG expected) {
    MemoryBarrier();
    return InterlockedCompareExchange(seq, 0, 0) == expected;
}

// ============================================================================
// 析构
// ============================================================================
MonitorDaemon::~MonitorDaemon() {
    DetachSharedMemory();
}

// ============================================================================
// 命名约定
// ============================================================================
wstring MonitorDaemon::GetMutexName(char d) {
    return wstring(L"Global\\FileRestoreMonitor_") + (wchar_t)d;
}

wstring MonitorDaemon::GetSharedMemName(char d) {
    return wstring(L"Global\\FileRestoreMonitor_") + (wchar_t)d + L"_Mem";
}

wstring MonitorDaemon::GetStopEventName(char d) {
    return wstring(L"Global\\FileRestoreMonitor_") + (wchar_t)d + L"_Stop";
}

// ============================================================================
// IsDaemonRunning - 通过 Mutex 检测守护进程是否已在运行
// ============================================================================
bool MonitorDaemon::IsDaemonRunning(char drive) {
    wstring mutexName = GetMutexName(drive);
    HANDLE hMutex = OpenMutexW(SYNCHRONIZE, FALSE, mutexName.c_str());
    if (hMutex) {
        CloseHandle(hMutex);
        return true;
    }
    return false;
}

// ============================================================================
// GetDaemonPID - 从共享内存读取 PID
// ============================================================================
DWORD MonitorDaemon::GetDaemonPID(char drive) {
    if (AttachSharedMemory(drive)) {
        MonitorSharedState state;
        if (ReadState(state)) {
            DetachSharedMemory();
            return state.pid;
        }
        DetachSharedMemory();
    }
    return 0;
}

// ============================================================================
// StartDaemon - 启动独立守护进程
// ============================================================================
bool MonitorDaemon::StartDaemon(char drive) {
    // 检查是否已运行
    if (IsDaemonRunning(drive)) {
        return true;
    }

    // 获取自身路径
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);

    // 构造命令行: "<path>" --monitor-daemon D
    wstring cmdLine = wstring(L"\"") + exePath + L"\" --monitor-daemon " + (wchar_t)drive;

    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};

    // CREATE_NO_WINDOW | DETACHED_PROCESS 确保无窗口后台运行
    BOOL ok = CreateProcessW(
        NULL,
        (LPWSTR)cmdLine.c_str(),
        NULL, NULL, FALSE,
        CREATE_NO_WINDOW | DETACHED_PROCESS,
        NULL, NULL,
        &si, &pi
    );

    if (!ok) {
        LOG_ERROR_FMT("StartDaemon: CreateProcessW failed, error=%u", GetLastError());
        return false;
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    // 轮询等待共享内存出现（最多 3 秒）
    for (int i = 0; i < 30; i++) {
        Sleep(100);
        if (IsDaemonRunning(drive)) {
            return true;
        }
    }

    LOG_ERROR("StartDaemon: Daemon did not start within timeout");
    return false;
}

// ============================================================================
// StopDaemon - 通知守护进程退出
// ============================================================================
bool MonitorDaemon::StopDaemon(char drive) {
    if (!IsDaemonRunning(drive)) {
        return true;  // 已经停止
    }

    wstring eventName = GetStopEventName(drive);
    HANDLE hEvent = OpenEventW(EVENT_MODIFY_STATE, FALSE, eventName.c_str());
    if (!hEvent) {
        LOG_ERROR_FMT("StopDaemon: Cannot open stop event, error=%u", GetLastError());
        return false;
    }

    SetEvent(hEvent);
    CloseHandle(hEvent);

    // 轮询等待守护进程退出（最多 5 秒）
    for (int i = 0; i < 50; i++) {
        Sleep(100);
        if (!IsDaemonRunning(drive)) {
            return true;
        }
    }

    LOG_ERROR("StopDaemon: Daemon did not stop within timeout");
    return false;
}

// ============================================================================
// AttachSharedMemory / DetachSharedMemory / ReadState
// ============================================================================
bool MonitorDaemon::AttachSharedMemory(char drive) {
    if (sharedPtr_ && attachedDrive_ == drive) return true;  // 同盘已附加
    if (sharedPtr_) DetachSharedMemory();  // 不同盘，先释放旧的

    wstring memName = GetSharedMemName(drive);
    hSharedMem_ = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, memName.c_str());
    if (!hSharedMem_) {
        return false;
    }

    sharedPtr_ = (MonitorSharedState*)MapViewOfFile(
        hSharedMem_, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(MonitorSharedState));
    if (!sharedPtr_) {
        CloseHandle(hSharedMem_);
        hSharedMem_ = nullptr;
        return false;
    }

    attachedDrive_ = drive;
    return true;
}

void MonitorDaemon::DetachSharedMemory() {
    if (sharedPtr_) {
        UnmapViewOfFile(sharedPtr_);
        sharedPtr_ = nullptr;
    }
    if (hSharedMem_) {
        CloseHandle(hSharedMem_);
        hSharedMem_ = nullptr;
    }
    attachedDrive_ = 0;
}

bool MonitorDaemon::ReadState(MonitorSharedState& out) {
    if (!sharedPtr_) return false;
    if (sharedPtr_->magic != MONITOR_SHARED_MAGIC) return false;

    // seqlock 读端：重试直到拿到一致快照
    for (int retry = 0; retry < 100; retry++) {
        LONG seq = SeqLockReadBegin(&sharedPtr_->seqCounter);
        if (seq & 1) {
            // seqCounter 为奇数：写端正在写入或已崩溃
            // 通过查询进程存活状态区分，避免误判
            if (IsProcessAlive(sharedPtr_->pid)) {
                // 守护进程仍在运行，写入尚未完成，等待后重试
                Sleep(1);
                continue;
            }
            // 守护进程已退出，seqCounter 不会再恢复为偶数
            // 拷贝一份尽力而为的快照
            memcpy(&out, (const void*)sharedPtr_, sizeof(MonitorSharedState));
            return true;
        }
        memcpy(&out, (const void*)sharedPtr_, sizeof(MonitorSharedState));
        if (SeqLockReadValidate(&sharedPtr_->seqCounter, seq)) {
            return true;  // 一致快照
        }
        YieldProcessor();
    }
    // 超过重试上限（不应发生），仍返回最后一次拷贝
    memcpy(&out, (const void*)sharedPtr_, sizeof(MonitorSharedState));
    return true;
}

// ============================================================================
// 自启动注册表
// ============================================================================
static wstring GetAutoStartValueName(char drive) {
    return wstring(L"FileRestoreMonitor_") + (wchar_t)drive;
}

bool MonitorDaemon::InstallAutoStart(char drive) {
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    wstring cmdLine = wstring(L"\"") + exePath + L"\" --monitor-daemon " + (wchar_t)drive;
    wstring valueName = GetAutoStartValueName(drive);

    HKEY hKey;
    LONG result = RegOpenKeyExW(HKEY_CURRENT_USER,
        L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
        0, KEY_SET_VALUE, &hKey);
    if (result != ERROR_SUCCESS) {
        LOG_ERROR_FMT("InstallAutoStart: RegOpenKeyExW failed, error=%ld", result);
        return false;
    }

    result = RegSetValueExW(hKey, valueName.c_str(), 0, REG_SZ,
        (const BYTE*)cmdLine.c_str(),
        (DWORD)((cmdLine.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(hKey);

    return result == ERROR_SUCCESS;
}

bool MonitorDaemon::UninstallAutoStart(char drive) {
    wstring valueName = GetAutoStartValueName(drive);

    HKEY hKey;
    LONG result = RegOpenKeyExW(HKEY_CURRENT_USER,
        L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
        0, KEY_SET_VALUE, &hKey);
    if (result != ERROR_SUCCESS) return false;

    result = RegDeleteValueW(hKey, valueName.c_str());
    RegCloseKey(hKey);

    return result == ERROR_SUCCESS || result == ERROR_FILE_NOT_FOUND;
}

bool MonitorDaemon::IsAutoStartInstalled(char drive) {
    wstring valueName = GetAutoStartValueName(drive);

    HKEY hKey;
    LONG result = RegOpenKeyExW(HKEY_CURRENT_USER,
        L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
        0, KEY_QUERY_VALUE, &hKey);
    if (result != ERROR_SUCCESS) return false;

    result = RegQueryValueExW(hKey, valueName.c_str(), NULL, NULL, NULL, NULL);
    RegCloseKey(hKey);

    return result == ERROR_SUCCESS;
}

// ============================================================================
// RunDaemonMain - 守护进程主入口
// ============================================================================
int MonitorDaemon::RunDaemonMain(char drive) {
    // 1. 创建单例 Mutex
    wstring mutexName = GetMutexName(drive);
    HANDLE hMutex = CreateMutexW(NULL, TRUE, mutexName.c_str());
    if (!hMutex || GetLastError() == ERROR_ALREADY_EXISTS) {
        if (hMutex) CloseHandle(hMutex);
        return 1;  // 已有实例运行
    }

    // 2. 创建停止事件（手动复位）
    wstring eventName = GetStopEventName(drive);
    HANDLE hStopEvent = CreateEventW(NULL, TRUE, FALSE, eventName.c_str());
    if (!hStopEvent) {
        ReleaseMutex(hMutex);
        CloseHandle(hMutex);
        return 1;
    }

    // 3. 创建共享内存
    wstring memName = GetSharedMemName(drive);
    HANDLE hMapping = CreateFileMappingW(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
        0, sizeof(MonitorSharedState), memName.c_str());
    if (!hMapping) {
        CloseHandle(hStopEvent);
        ReleaseMutex(hMutex);
        CloseHandle(hMutex);
        return 1;
    }

    MonitorSharedState* shared = (MonitorSharedState*)MapViewOfFile(
        hMapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(MonitorSharedState));
    if (!shared) {
        CloseHandle(hMapping);
        CloseHandle(hStopEvent);
        ReleaseMutex(hMutex);
        CloseHandle(hMutex);
        return 1;
    }

    // 4. 初始化共享内存
    memset(shared, 0, sizeof(MonitorSharedState));
    shared->magic = MONITOR_SHARED_MAGIC;
    shared->version = MONITOR_SHARED_VERSION;
    shared->driveLetter = drive;
    shared->pid = GetCurrentProcessId();
    GetSystemTimeAsFileTime(&shared->startTime);
    shared->daemonRunning = 1;
    shared->autoStartEnabled = IsAutoStartInstalled(drive) ? 1 : 0;

    // 5. 创建 UsnDeleteMonitor
    UsnDeleteMonitor monitor(drive);
    monitor.SetPollInterval(1000);

    shared->pollIntervalMs = monitor.GetPollInterval();

    // 设置事件回调 - 写入环形缓冲区（writerLock + seqlock 保护）
    monitor.SetEventCallback([shared](ULONGLONG recordNum, ULONGLONG fileSize,
                                       FILETIME deleteTime, const wstring& fileName,
                                       bool captured) {
        WriterLockAcquire(&shared->writerLock);
        SeqLockWriteBegin(&shared->seqCounter);

        LONG head = shared->recentEventHead;
        LONG idx = head % MONITOR_RECENT_EVENT_MAX;

        MonitorRecentEvent& evt = shared->recentEvents[idx];
        evt.mftRecord = recordNum;
        evt.fileSize = fileSize;
        evt.deleteTime = deleteTime;
        evt.captured = captured ? 1 : 0;

        // 安全拷贝文件名
        size_t copyLen = min(fileName.size(), (size_t)259);
        memcpy(evt.fileName, fileName.c_str(), copyLen * sizeof(wchar_t));
        evt.fileName[copyLen] = L'\0';

        shared->recentEventHead = head + 1;
        LONG count = shared->recentEventCount + 1;
        if (count > MONITOR_RECENT_EVENT_MAX) count = MONITOR_RECENT_EVENT_MAX;
        shared->recentEventCount = count;

        SeqLockWriteEnd(&shared->seqCounter);
        WriterLockRelease(&shared->writerLock);
    });

    // 6. 捕获现有删除记录
    LOG_INFO_FMT("Daemon: Capturing existing deleted files on %c:...", drive);
    size_t captured = monitor.CaptureExistingDeleted(24);
    InterlockedExchange64(&shared->snapshotCount, (LONGLONG)monitor.GetSnapshotStore().GetCount());
    LOG_INFO_FMT("Daemon: Captured %zu existing snapshots", captured);

    // 7. 启动监控
    if (!monitor.Start()) {
        LOG_ERROR("Daemon: Failed to start monitor");
        shared->daemonRunning = 0;
        UnmapViewOfFile(shared);
        CloseHandle(hMapping);
        CloseHandle(hStopEvent);
        ReleaseMutex(hMutex);
        CloseHandle(hMutex);
        return 1;
    }

    LOG_INFO_FMT("Daemon: Monitor started on %c:, PID=%u", drive, GetCurrentProcessId());

    // 8. 主循环 - 等待停止信号，定期更新统计
    while (WaitForSingleObject(hStopEvent, 500) == WAIT_TIMEOUT) {
        // 更新统计到共享内存（writerLock + seqlock 保护）
        auto& stats = monitor.GetStats();
        WriterLockAcquire(&shared->writerLock);
        SeqLockWriteBegin(&shared->seqCounter);
        shared->totalEvents = (LONGLONG)stats.totalEvents.load();
        shared->capturedCount = (LONGLONG)stats.capturedCount.load();
        shared->missedCount = (LONGLONG)stats.missedCount.load();
        shared->skippedCount = (LONGLONG)stats.skippedCount.load();
        shared->snapshotCount = (LONGLONG)monitor.GetSnapshotStore().GetCount();
        GetSystemTimeAsFileTime(&shared->lastUpdate);
        SeqLockWriteEnd(&shared->seqCounter);
        WriterLockRelease(&shared->writerLock);
    }

    // 9. 清理退出
    LOG_INFO("Daemon: Stop signal received, shutting down...");
    monitor.Stop();
    shared->daemonRunning = 0;

    UnmapViewOfFile(shared);
    CloseHandle(hMapping);
    CloseHandle(hStopEvent);
    ReleaseMutex(hMutex);
    CloseHandle(hMutex);

    LOG_INFO("Daemon: Exited cleanly");
    return 0;
}
