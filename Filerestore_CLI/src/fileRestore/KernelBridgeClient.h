#pragma once

// ============================================================================
// 内核桥接客户端 - 通过 minifilter 通信端口连接 FileRestoreMon 驱动
//
// 默认不启用。要启用，在项目属性中定义 ENABLE_KERNEL_BRIDGE 预处理器宏。
// ============================================================================

#ifdef ENABLE_KERNEL_BRIDGE

#include <Windows.h>
#include <fltUser.h>
#include <string>
#include <vector>

// 共享定义（与内核驱动 common.h 一致）
#include "../../../../Filerestore_sys/Filerestore_sys/common.h"

class MFTSnapshotStore;

class KernelBridgeClient {
public:
    KernelBridgeClient();
    ~KernelBridgeClient();

    // ========== 连接管理 ==========

    // 通过 FilterConnectCommunicationPort 连接到 minifilter 驱动
    bool Connect();

    // 断开连接
    void Disconnect();

    // 是否已连接
    bool IsConnected() const { return hPort != INVALID_HANDLE_VALUE; }

    // ========== 监控控制 ==========

    // 启动文件删除监控
    bool StartMonitor();

    // 停止文件删除监控
    bool StopMonitor();

    // ========== 通知接收 ==========

    // 删除通知结构
    struct DeleteNotification {
        std::wstring fileName;
        ULONGLONG fileSize;
        ULONGLONG allocationSize;
        FILETIME creationTime;
        FILETIME lastWriteTime;
        FILETIME deleteTime;
        DWORD processId;
        DWORD deleteType;
        std::vector<std::pair<ULONGLONG, ULONG>> lcnMappings;  // (startLCN, clusterCount)
    };

    // 从驱动获取待处理的删除通知
    bool PollNotifications(std::vector<DeleteNotification>& notifications);

    // 将通知转换为快照并存入 MFTSnapshotStore
    size_t FlushToSnapshotStore(MFTSnapshotStore& store);

    // ========== 驱动信息 ==========

    // 获取驱动版本
    std::string GetDriverVersion();

    // 驱动统计信息
    struct DriverStats {
        ULONG totalEvents;
        ULONG pendingEvents;
        ULONG droppedEvents;
        bool monitorActive;
    };

    // 获取驱动统计
    bool GetStats(DriverStats& stats);

    // 获取最后的错误信息
    std::string GetLastError() const { return lastError; }

private:
    HANDLE hPort;           // minifilter 通信端口句柄
    std::string lastError;

    // 向驱动发送命令并接收回复
    bool SendCommand(ULONG command, void* outBuffer, ULONG outSize, ULONG* bytesReturned);
};

#endif // ENABLE_KERNEL_BRIDGE
