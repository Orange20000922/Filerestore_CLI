// ============================================================================
// 内核桥接客户端 - 通过 minifilter 通信端口连接 FileRestoreMon 驱动
// ============================================================================

#ifdef ENABLE_KERNEL_BRIDGE

#include "KernelBridgeClient.h"
#include "MFTSnapshotStore.h"
#include "Logger.h"

#pragma comment(lib, "fltlib.lib")

using namespace std;

KernelBridgeClient::KernelBridgeClient()
    : hPort(INVALID_HANDLE_VALUE) {}

KernelBridgeClient::~KernelBridgeClient() {
    Disconnect();
}

// ========== 连接管理 ==========

bool KernelBridgeClient::Connect() {
    if (IsConnected()) return true;

    HRESULT hr = FilterConnectCommunicationPort(
        FILERESTORE_PORT_NAME,
        0,              // options
        nullptr,        // context
        0,              // context size
        nullptr,        // security attributes
        &hPort);

    if (FAILED(hr)) {
        hPort = INVALID_HANDLE_VALUE;
        if (hr == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND)) {
            lastError = "驱动未加载: FileRestoreMon 通信端口不存在";
        } else if (hr == HRESULT_FROM_WIN32(ERROR_ACCESS_DENIED)) {
            lastError = "权限不足: 需要管理员权限连接驱动";
        } else {
            lastError = "连接驱动失败, HRESULT: 0x" +
                        to_string(static_cast<unsigned long>(hr));
        }
        LOG_DEBUG_FMT("KernelBridgeClient: %s", lastError.c_str());
        return false;
    }

    LOG_INFO("KernelBridgeClient: 已连接到 minifilter 驱动");
    return true;
}

void KernelBridgeClient::Disconnect() {
    if (hPort != INVALID_HANDLE_VALUE) {
        CloseHandle(hPort);
        hPort = INVALID_HANDLE_VALUE;
    }
}

// ========== 内部通信 ==========

bool KernelBridgeClient::SendCommand(ULONG command, void* outBuffer, ULONG outSize, ULONG* bytesReturned) {
    if (!IsConnected()) {
        lastError = "未连接到驱动";
        return false;
    }

    COMMAND_MESSAGE msg;
    msg.Command = command;

    DWORD returned = 0;
    HRESULT hr = FilterSendMessage(
        hPort,
        &msg,
        sizeof(msg),
        outBuffer,
        outSize,
        &returned);

    if (FAILED(hr)) {
        lastError = "FilterSendMessage 失败, HRESULT: 0x" +
                    to_string(static_cast<unsigned long>(hr));
        return false;
    }

    if (bytesReturned) {
        *bytesReturned = returned;
    }
    return true;
}

// ========== 监控控制 ==========

bool KernelBridgeClient::StartMonitor() {
    return SendCommand(CMD_START_MONITOR, nullptr, 0, nullptr);
}

bool KernelBridgeClient::StopMonitor() {
    return SendCommand(CMD_STOP_MONITOR, nullptr, 0, nullptr);
}

// ========== 通知接收 ==========

bool KernelBridgeClient::PollNotifications(vector<DeleteNotification>& notifications) {
    notifications.clear();

    // 分配接收缓冲区：能容纳 EVENTS_REPLY 头 + 若干 DELETE_NOTIFICATION
    const ULONG BUFFER_SIZE = FIELD_OFFSET(EVENTS_REPLY, Events) +
                              64 * sizeof(DELETE_NOTIFICATION);
    vector<BYTE> buffer(BUFFER_SIZE, 0);

    ULONG bytesReturned = 0;
    if (!SendCommand(CMD_GET_EVENTS, buffer.data(), BUFFER_SIZE, &bytesReturned)) {
        return false;
    }

    if (bytesReturned < FIELD_OFFSET(EVENTS_REPLY, Events)) {
        return true;  // 无数据
    }

    auto* reply = reinterpret_cast<EVENTS_REPLY*>(buffer.data());
    if (reply->EventCount == 0) {
        return true;
    }

    // 校验返回数据大小
    ULONG expectedSize = FIELD_OFFSET(EVENTS_REPLY, Events) +
                         reply->EventCount * sizeof(DELETE_NOTIFICATION);
    if (bytesReturned < expectedSize) {
        lastError = "回复数据不完整";
        return false;
    }

    notifications.reserve(reply->EventCount);
    for (ULONG i = 0; i < reply->EventCount; i++) {
        const DELETE_NOTIFICATION& src = reply->Events[i];
        DeleteNotification notif;

        notif.fileName = wstring(src.FileName, src.FileNameLength / sizeof(WCHAR));
        notif.fileSize = src.FileSize.QuadPart;
        notif.allocationSize = src.AllocationSize.QuadPart;
        notif.processId = src.ProcessId;
        notif.deleteType = src.DeleteType;

        // LARGE_INTEGER → FILETIME（二进制布局相同）
        memcpy(&notif.creationTime, &src.CreationTime, sizeof(FILETIME));
        memcpy(&notif.lastWriteTime, &src.LastWriteTime, sizeof(FILETIME));
        memcpy(&notif.deleteTime, &src.Timestamp, sizeof(FILETIME));

        // LCN 映射
        notif.lcnMappings.reserve(src.LCNCount);
        for (ULONG j = 0; j < src.LCNCount; j++) {
            notif.lcnMappings.emplace_back(
                src.LCNEntries[j].StartLCN,
                src.LCNEntries[j].ClusterCount);
        }

        notifications.push_back(move(notif));
    }

    return true;
}

size_t KernelBridgeClient::FlushToSnapshotStore(MFTSnapshotStore& store) {
    vector<DeleteNotification> notifications;
    if (!PollNotifications(notifications)) {
        return 0;
    }

    size_t count = 0;
    for (const auto& notif : notifications) {
        MFTSnapshot snapshot;
        snapshot.recordNumber = 0;      // 驱动未提供 MFT 记录号
        snapshot.sequenceNumber = 0;
        snapshot.fileName = notif.fileName;
        snapshot.fileSize = notif.fileSize;
        snapshot.isResident = notif.lcnMappings.empty() && notif.fileSize > 0;

        // LCN 映射转换: (ULONGLONG, ULONG) → (ULONGLONG, ULONGLONG)
        snapshot.dataRuns.reserve(notif.lcnMappings.size());
        for (const auto& [lcn, clusters] : notif.lcnMappings) {
            snapshot.dataRuns.emplace_back(lcn, static_cast<ULONGLONG>(clusters));
        }

        snapshot.deleteTime = notif.deleteTime;
        GetSystemTimeAsFileTime(&snapshot.captureTime);

        store.AddSnapshot(snapshot);
        count++;
    }

    return count;
}

// ========== 驱动信息 ==========

string KernelBridgeClient::GetDriverVersion() {
    if (!IsConnected()) return "未连接";

    VERSION_REPLY reply = {};
    ULONG bytesReturned = 0;

    if (!SendCommand(CMD_GET_VERSION, &reply, sizeof(reply), &bytesReturned)) {
        return "未知版本";
    }

    if (bytesReturned == 0) {
        return "未知版本";
    }

    return string(reply.Version);
}

bool KernelBridgeClient::GetStats(DriverStats& stats) {
    STATS_REPLY reply = {};
    ULONG bytesReturned = 0;

    if (!SendCommand(CMD_GET_STATS, &reply, sizeof(reply), &bytesReturned)) {
        return false;
    }

    if (bytesReturned < sizeof(STATS_REPLY)) {
        lastError = "统计数据不完整";
        return false;
    }

    stats.totalEvents = reply.TotalEvents;
    stats.pendingEvents = reply.PendingEvents;
    stats.droppedEvents = reply.DroppedEvents;
    stats.monitorActive = (reply.MonitorActive != 0);
    return true;
}

#endif // ENABLE_KERNEL_BRIDGE
