#pragma once
#include <Windows.h>
#include <string>

// ============================================================================
// 共享内存常量
// ============================================================================
constexpr DWORD MONITOR_SHARED_MAGIC = 0x46524D44;   // 'FRMD'
constexpr DWORD MONITOR_SHARED_VERSION = 1;
constexpr int MONITOR_RECENT_EVENT_MAX = 16;

// ============================================================================
// 共享内存结构 - 固定大小 POD，无指针/STL
// ============================================================================
#pragma pack(push, 8)

struct MonitorRecentEvent {
    ULONGLONG mftRecord;
    ULONGLONG fileSize;
    FILETIME deleteTime;
    wchar_t fileName[260];
    BYTE captured;           // 1=success, 0=failed
    BYTE padding[7];
};

struct MonitorSharedState {
    DWORD magic, version;
    char driveLetter; char pad1[3];
    DWORD pid;
    FILETIME startTime, lastUpdate;
    // 统计（Interlocked 更新）
    volatile LONGLONG totalEvents, capturedCount, missedCount, skippedCount, snapshotCount;
    DWORD pollIntervalMs; DWORD pad2;
    // 环形缓冲区自旋锁
    volatile LONG spinLock;
    // 环形缓冲区
    volatile LONG recentEventCount, recentEventHead;
    MonitorRecentEvent recentEvents[MONITOR_RECENT_EVENT_MAX];
    // 标志
    volatile LONG daemonRunning, autoStartEnabled;
};

#pragma pack(pop)

// ============================================================================
// MonitorDaemon - 守护进程管理器
// ============================================================================
class MonitorDaemon {
public:
    MonitorDaemon() = default;
    ~MonitorDaemon();

    // ========== 守护进程生命周期 ==========
    bool StartDaemon(char drive);
    bool StopDaemon(char drive);
    bool IsDaemonRunning(char drive);
    DWORD GetDaemonPID(char drive);

    // ========== 共享内存读取（CLI/TUI 侧）==========
    bool AttachSharedMemory(char drive);
    void DetachSharedMemory();
    bool ReadState(MonitorSharedState& out);

    // ========== 自启动注册表 ==========
    static bool InstallAutoStart(char drive);
    static bool UninstallAutoStart(char drive);
    static bool IsAutoStartInstalled(char drive);

    // ========== 命名约定 ==========
    static std::wstring GetMutexName(char d);
    static std::wstring GetSharedMemName(char d);
    static std::wstring GetStopEventName(char d);

    // ========== 守护进程入口（从 Main.cpp 调用）==========
    static int RunDaemonMain(char drive);

private:
    HANDLE hSharedMem_ = nullptr;
    MonitorSharedState* sharedPtr_ = nullptr;
};
