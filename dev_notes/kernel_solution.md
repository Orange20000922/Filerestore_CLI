# 内核层删除监控方案 — 技术设计文档

> 背景：MFT 复用后丢失文件的精确大小和 LCN 信息，导致签名扫描退化为全盘盲扫。
> 本文档讨论两种解决方案：用户态 MFT 快照、内核态微过滤驱动实时监控。

---

## 一、问题定义

```
现有痛点:
  文件删除 → MFT 记录被复用 → 丢失大小和 LCN → 签名扫描盲目

解决思路:
  在 MFT 复用之前，保存文件的关键元数据（大小、LCN、文件名）
  恢复时利用这些元数据进行定向扫描
```

---

## 二、方案一：MFT 元数据快照（用户态）

### 2.1 核心优势

| 已知信息 | 收益 |
|---------|------|
| 精确大小 | 避免截断/过度读取 |
| LCN 范围 | 扫描范围缩小 99% |
| 文件名 | 无需人工识别 |

### 2.2 快照数据结构

每条记录约 100-200 字节：

```cpp
struct MFTSnapshotEntry {
    uint64_t mftRecordNo;       // 8 bytes
    uint64_t parentRecordNo;    // 8 bytes
    uint32_t sequenceNo;        // 4 bytes
    uint64_t fileSize;          // 8 bytes
    uint64_t allocSize;         // 8 bytes
    uint16_t lcnCount;          // 2 bytes
    uint64_t timestamps[4];     // 32 bytes (创建/修改/访问/变更)
    uint32_t attributes;        // 4 bytes
    uint16_t nameLen;           // 2 bytes
    wchar_t  name[256];         // 512 bytes (可变长)
    LCNEntry lcnEntries[];      // 变长 (offset + length)
};
```

### 2.3 空间估算（D 盘 731,904 条记录）

- 精简版（无文件名）：~70 MB
- 完整版（含文件名）：~150 MB
- 压缩后（ZSTD）：~30-50 MB

### 2.4 实现策略

```
快照管理系统:
  ├── 全量快照（每日定时 / 手动触发）
  ├── 增量快照（USN 驱动，检测到删除时抢读 MFT）
  └── 快照合并与自动清理
        │
        ▼
  快照存储引擎（二进制 + 索引）
        │
  ┌─────┼──────┐
  ▼     ▼      ▼
签名扫描  大小验证  LCN 过滤
精准定位  完整性校验 范围限制
```

### 2.5 对签名扫描的改进

**当前流程（盲目扫描）：**

```
全盘扫描 → 找到签名头 → 猜测大小 → 读取 → 验证尾部
问题: 大小不确定，可能截断或过度读取
```

**快照辅助流程：**

```
查快照获取 LCN 范围 → 只扫描这些簇 → 已知精确大小 → 精准读取
优势: 扫描范围 1% + 精确截断 + 可验证完整性
```

### 2.6 命令设计

```bash
# 创建快照
listdeleted D --create-snapshot

# 自动后台快照（可选）
listdeleted D --snapshot-schedule hourly

# 使用快照恢复
recover D document.docx D:\out --use-snapshot

# 签名扫描 + 快照辅助
carvepool D all D:\out --snapshot-guided

# 快照管理
snapshot list D
snapshot delete D --older-than 7d
```

### 2.7 效果预估

```
无快照:                     有快照:
  扫描范围: 550 GB            扫描范围: ~5.5 GB (1%)
  扫描时间: ~4 分钟            扫描时间: ~2 秒
  大小精度: 估计值（误差大）    大小精度: 精确值
```

---

## 三、方案二：内核层实时监控（微过滤驱动）

### 3.1 技术路径

使用 Windows 文件系统微过滤驱动（Minifilter），通过 FltMgr 框架注册 Pre-operation 回调，在文件删除操作到达文件系统之前捕获元数据。

### 3.2 可获取的信息

```
内核层能获取的精确信息:
├── 文件大小 (FILE_STANDARD_INFORMATION)
├── LCN 列表 (FSCTL_GET_RETRIEVAL_POINTERS)
├── 完整路径
├── 删除类型 (回收站 vs Shift+Delete)
├── 删除进程 PID/名称
├── 精确时间戳 (微秒级)
└── 文件属性 (压缩/加密/稀疏)
```

### 3.3 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  用户态 (Filerestore_CLI.exe)                                    │
│                                                                  │
│  MonitorClient                                                   │
│  ├── Connect()          → CreateFile(设备)                       │
│  ├── StartMonitoring()  → DeviceIoControl(START)                 │
│  ├── StopMonitoring()   → DeviceIoControl(STOP)                  │
│  ├── GetEvents()        → 读取共享内存队列                        │
│  └── WaitForData()      → WaitForSingleObject(事件)              │
└─────────────────────────┬──────────────────▲─────────────────────┘
                          │ DeviceIoControl  │ 共享内存
                          ▼                  │
┌─────────────────────────────────────────────────────────────────┐
│  内核态 (FileRestoreMonitor.sys)                                 │
│                                                                  │
│  DriverEntry                                                     │
│  ├── 创建设备对象  \Device\FileRestoreMon                        │
│  ├── 创建符号链接  \DosDevices\FileRestoreMon                    │
│  ├── 注册 IRP 派发函数                                           │
│  └── 注册文件系统过滤器                                          │
│          │                                                       │
│  ┌───────┼───────────────────┐                                   │
│  ▼       ▼                   ▼                                   │
│ IRP 派发   过滤回调            共享内存管理                       │
│ IRP_CREATE  PreSetInformation   RingBuffer                       │
│ IRP_CLOSE   PreCleanup          DataSection                      │
│ IRP_IOCTL   PreCreate           Event                            │
│              │                                                   │
│              ▼                                                   │
│         写入环形缓冲区 → 触发事件通知                             │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                    NTFS / ReFS
```

### 3.4 挑战与应对

| 挑战 | 解决方案 |
|------|---------|
| 驱动签名 | 开发阶段：测试签名模式；发布：EV 证书 + 硬件开发人员计划 |
| 稳定性 (BSOD) | 严格错误处理 + 内存池管理 + 参数校验 |
| 性能开销 | 无锁环形缓冲区 + Per-CPU 队列 |
| 安装部署 | 可选组件，用户态降级方案（无驱动时退回快照模式） |

---

## 四、用户态-内核态通信机制对比

### 4.1 DeviceIoControl（传统方式）

```
用户态                                      内核态
  │                                           │
  │  CreateFile("\\\\.\\FileRestoreMon")       │
  │ ─────────────────────────────────────────►│  IRP_MJ_CREATE
  │                                           │
  │  DeviceIoControl(IOCTL_GET_EVENTS)        │
  │ ─────────────────────────────────────────►│  IRP_MJ_DEVICE_CONTROL
  │                                           │  查询缓冲区
  │ ◄─────────────────────────────────────────│  返回数据
  │                                           │
  │  CloseHandle()                            │
  │ ─────────────────────────────────────────►│  IRP_MJ_CLOSE
```

### 4.2 共享内存 + 事件（高性能方式）

```
用户态                                      内核态
  │                                           │
  │  OpenFileMapping()                        │
  │ ─────────────────────────────────────────►│
  │  MapViewOfFile()  ◄──────────────────────►│  ZwCreateSection
  │       ▲                              ▲    │
  │       └────────── 共享内存区域 ────────┘    │
  │                                           │
  │  WaitForSingleObject()  ◄────────────────►│  KeSetEvent()
  │       ▲                              ▲    │
  │       └────────── 事件通知 ───────────┘    │
  │                                           │
  │  等待事件 → 读取共享内存 → 处理数据        │
```

### 4.3 对比

| 维度 | DeviceIoControl | 共享内存 + 事件 |
|------|----------------|----------------|
| 实现复杂度 | 低（标准模式） | 中（需同步机制） |
| 每次通信开销 | 系统调用 + IRP 构建 | 仅内存访问 |
| 适合场景 | 低频、小数据量 | 高频、大数据流 |
| 缓冲区管理 | 内核分配，用户拷贝 | 预分配，零拷贝 |
| 同步机制 | 隐式（调用阻塞） | 显式（事件/信号量） |
| 数据方向 | 双向灵活 | 适合单向流 |

### 4.4 删除监控场景分析

```
删除事件特征:
├── 频率: 高（每秒可能数十次）
├── 方向: 单向（内核 → 用户态）
├── 数据量: 每条 ~200-500 bytes
├── 实时性要求: 中（毫秒级可接受）
└── 缓冲需求: 需要队列（突发流量）
```

**结论：混合方案最优**

```
推荐架构:
├── 控制通道 (DeviceIoControl)
│   ├── 启动/停止监控
│   ├── 配置过滤规则
│   ├── 获取统计信息
│   └── 获取事件句柄
│
└── 数据通道 (共享内存 + 事件)
    ├── 高吞吐量删除事件流
    ├── 零拷贝读取
    └── 事件驱动通知
```

---

## 五、核心代码参考

### 5.1 全局上下文与驱动入口

```cpp
// driver.cpp
#include <ntddk.h>
#include <fltMgr.h>

typedef struct _GLOBAL_CONTEXT {
    PDEVICE_OBJECT  DeviceObject;
    PFLT_FILTER     FilterHandle;
    PFLT_PORT       ServerPort;
    HANDLE          SectionHandle;
    PVOID           SharedMemoryBase;
    SIZE_T          SharedMemorySize;
    KEVENT          DataReadyEvent;
    KSPIN_LOCK      BufferLock;
    ULONG           WriteIndex;
    ULONG           ReadIndex;
} GLOBAL_CONTEXT, *PGLOBAL_CONTEXT;

static GLOBAL_CONTEXT g_Context = {0};

NTSTATUS DriverEntry(
    PDRIVER_OBJECT  DriverObject,
    PUNICODE_STRING RegistryPath
) {
    NTSTATUS status;
    UNICODE_STRING deviceName = RTL_CONSTANT_STRING(L"\\Device\\FileRestoreMon");
    UNICODE_STRING symLink    = RTL_CONSTANT_STRING(L"\\DosDevices\\FileRestoreMon");

    // 1. 创建控制设备
    status = IoCreateDevice(
        DriverObject, 0, &deviceName,
        FILE_DEVICE_UNKNOWN, FILE_DEVICE_SECURE_OPEN,
        FALSE, &g_Context.DeviceObject
    );
    if (!NT_SUCCESS(status)) return status;

    // 2. 创建符号链接（用户态通过 \\\\.\\FileRestoreMon 访问）
    status = IoCreateSymbolicLink(&symLink, &deviceName);
    if (!NT_SUCCESS(status)) {
        IoDeleteDevice(g_Context.DeviceObject);
        return status;
    }

    // 3. 注册 IRP 派发函数
    DriverObject->MajorFunction[IRP_MJ_CREATE]         = DispatchCreate;
    DriverObject->MajorFunction[IRP_MJ_CLOSE]          = DispatchClose;
    DriverObject->MajorFunction[IRP_MJ_DEVICE_CONTROL] = DispatchIoControl;
    DriverObject->DriverUnload = DriverUnload;

    // 4. 初始化共享内存
    status = InitializeSharedMemory();
    if (!NT_SUCCESS(status)) {
        IoDeleteSymbolicLink(&symLink);
        IoDeleteDevice(g_Context.DeviceObject);
        return status;
    }

    // 5. 注册文件系统过滤器
    status = RegisterFilter(DriverObject);
    if (!NT_SUCCESS(status)) {
        CleanupSharedMemory();
        IoDeleteSymbolicLink(&symLink);
        IoDeleteDevice(g_Context.DeviceObject);
        return status;
    }

    return STATUS_SUCCESS;
}
```

### 5.2 共享内存初始化

```cpp
// shared_memory.cpp
#define SHARED_MEMORY_SIZE (16 * 1024 * 1024)  // 16 MB 环形缓冲区

NTSTATUS InitializeSharedMemory() {
    NTSTATUS status;
    OBJECT_ATTRIBUTES attr;
    LARGE_INTEGER maxSize;
    UNICODE_STRING sectionName =
        RTL_CONSTANT_STRING(L"\\BaseNamedObjects\\FileRestoreMonBuffer");

    // 1. 创建命名 Section（用户态可通过名称打开）
    maxSize.QuadPart = SHARED_MEMORY_SIZE;
    InitializeObjectAttributes(
        &attr, &sectionName,
        OBJ_KERNEL_HANDLE | OBJ_OPENIF,
        NULL, NULL
    );

    status = ZwCreateSection(
        &g_Context.SectionHandle,
        SECTION_ALL_ACCESS, &attr, &maxSize,
        PAGE_READWRITE, SEC_COMMIT, NULL
    );
    if (!NT_SUCCESS(status)) return status;

    // 2. 映射到内核地址空间
    SIZE_T viewSize = SHARED_MEMORY_SIZE;
    status = ZwMapViewOfSection(
        g_Context.SectionHandle,
        ZwCurrentProcess(),
        &g_Context.SharedMemoryBase,
        0, 0, NULL, &viewSize,
        ViewUnmap, 0, PAGE_READWRITE
    );
    if (!NT_SUCCESS(status)) {
        ZwClose(g_Context.SectionHandle);
        return status;
    }

    // 3. 初始化环形缓冲区
    RtlZeroMemory(g_Context.SharedMemoryBase, SHARED_MEMORY_SIZE);
    g_Context.WriteIndex = 0;
    g_Context.ReadIndex  = 0;
    KeInitializeSpinLock(&g_Context.BufferLock);

    // 4. 创建通知事件
    KeInitializeEvent(&g_Context.DataReadyEvent, NotificationEvent, FALSE);

    return STATUS_SUCCESS;
}
```

### 5.3 文件系统过滤回调

```cpp
// filter_callback.cpp
#include <fltKernel.h>

// 过滤器操作注册表
FLT_OPERATION_REGISTRATION filterOps[] = {
    { IRP_MJ_SET_INFORMATION, 0, PreSetInformation, NULL },
    { IRP_MJ_CREATE,          0, PreCreate,         NULL },
    { IRP_MJ_OPERATION_END }
};

FLT_REGISTRATION filterRegistration = {
    sizeof(FLT_REGISTRATION),
    FLT_REGISTRATION_VERSION,
    0,
    NULL,           // Context
    filterOps,      // Operation Registration
    UnloadFilter,
    NULL, NULL, NULL, NULL, NULL, NULL
};

NTSTATUS RegisterFilter(PDRIVER_OBJECT DriverObject) {
    return FltRegisterFilter(
        DriverObject, &filterRegistration, &g_Context.FilterHandle
    );
}

// 核心回调：拦截删除操作
FLT_PREOP_CALLBACK_STATUS PreSetInformation(
    PFLT_CALLBACK_DATA    Data,
    PCFLT_RELATED_OBJECTS FltObjects,
    PVOID                *CompletionContext
) {
    FILE_INFORMATION_CLASS infoClass =
        Data->Iopb->Parameters.SetFileInformation.FileInformationClass;

    // 只处理删除和重命名（回收站是重命名操作）
    switch (infoClass) {
    case FileDispositionInformation:
    case FileDispositionInformationEx:
        break;
    case FileRenameInformation:
    case FileRenameInformationEx:
        break;
    default:
        return FLT_PREOP_SUCCESS_NO_CALLBACK;
    }

    // 检查是否真的要删除
    if (infoClass == FileDispositionInformation) {
        PFILE_DISPOSITION_INFO dispInfo =
            (PFILE_DISPOSITION_INFO)Data->Iopb->Parameters
                .SetFileInformation.InfoBuffer;
        if (!dispInfo->DeleteFile) {
            return FLT_PREOP_SUCCESS_NO_CALLBACK;
        }
    }

    // 捕获删除事件
    CaptureDeleteEvent(FltObjects->FileObject, infoClass);

    return FLT_PREOP_SUCCESS_NO_CALLBACK;
}
```

### 5.4 删除事件捕获

```cpp
// capture.cpp

VOID CaptureDeleteEvent(
    PFILE_OBJECT             FileObject,
    FILE_INFORMATION_CLASS   InfoClass
) {
    NTSTATUS status;
    DELETE_EVENT_RECORD record = {0};

    // 1. 获取文件基本信息（时间戳、属性）
    status = QueryFileBasicInfo(FileObject, &record);
    if (!NT_SUCCESS(status)) return;  // 静默失败，不影响原始操作

    // 2. 获取文件大小
    status = QueryFileStandardInfo(FileObject, &record);
    if (!NT_SUCCESS(status)) return;

    // 3. 获取完整路径
    status = QueryFileName(FileObject, record.FullPath, sizeof(record.FullPath));
    if (!NT_SUCCESS(status)) return;

    // 4. 获取 LCN 信息（关键）
    status = QueryFileLCNs(FileObject, &record);
    // 注意：这可能失败（常驻文件没有非常驻数据属性）

    // 5. 填充其他信息
    record.DeleteType = (InfoClass == FileRenameInformation)
                        ? DELETE_TYPE_RECYCLE
                        : DELETE_TYPE_PERMANENT;
    record.Timestamp  = KeQueryPerformanceCounter(NULL).QuadPart;
    record.ProcessId  = (ULONG)(ULONG_PTR)PsGetCurrentProcessId();

    // 6. 写入环形缓冲区
    WriteToRingBuffer(&record);
}
```

### 5.5 获取 LCN 信息（关键技术点）

```cpp
// lcn_query.cpp
//
// 使用 FSCTL_GET_RETRIEVAL_POINTERS 获取文件的 VCN → LCN 映射。
// 这是 NTFS 特有的 FSCTL，返回文件数据在磁盘上的物理位置。

NTSTATUS QueryFileLCNs(
    PFILE_OBJECT          FileObject,
    PDELETE_EVENT_RECORD  Record
) {
    NTSTATUS status;
    IO_STATUS_BLOCK iosb;
    STARTING_VCN_INPUT_BUFFER vcnInput = {0};
    RETRIEVAL_POINTERS_BUFFER *rpBuf = NULL;
    ULONG bufSize = 4096;  // 初始缓冲区

    vcnInput.StartingVcn.QuadPart = 0;

    // 分配缓冲区并查询
    rpBuf = (RETRIEVAL_POINTERS_BUFFER*)
        ExAllocatePoolWithTag(NonPagedPool, bufSize, 'CnLG');
    if (!rpBuf) return STATUS_INSUFFICIENT_RESOURCES;

    status = FltFsControlFile(
        FltObjects->Instance,  // 需要从回调参数传入
        FileObject,
        FSCTL_GET_RETRIEVAL_POINTERS,
        &vcnInput, sizeof(vcnInput),
        rpBuf, bufSize,
        &iosb
    );

    // 缓冲区不够时重新分配
    if (status == STATUS_BUFFER_OVERFLOW) {
        ExFreePoolWithTag(rpBuf, 'CnLG');
        bufSize = (ULONG)iosb.Information;
        rpBuf = (RETRIEVAL_POINTERS_BUFFER*)
            ExAllocatePoolWithTag(NonPagedPool, bufSize, 'CnLG');
        if (!rpBuf) return STATUS_INSUFFICIENT_RESOURCES;

        status = FltFsControlFile(
            FltObjects->Instance,
            FileObject,
            FSCTL_GET_RETRIEVAL_POINTERS,
            &vcnInput, sizeof(vcnInput),
            rpBuf, bufSize,
            &iosb
        );
    }

    if (NT_SUCCESS(status)) {
        // 解析 VCN → LCN 映射
        LARGE_INTEGER prevVcn = rpBuf->StartingVcn;
        Record->LCNCount = 0;

        for (ULONG i = 0;
             i < rpBuf->ExtentCount && Record->LCNCount < MAX_LCN_ENTRIES;
             i++)
        {
            Record->LCNEntries[Record->LCNCount].StartLCN =
                rpBuf->Extents[i].Lcn.QuadPart;
            Record->LCNEntries[Record->LCNCount].ClusterCount =
                (ULONG)(rpBuf->Extents[i].NextVcn.QuadPart - prevVcn.QuadPart);

            prevVcn = rpBuf->Extents[i].NextVcn;
            Record->LCNCount++;
        }
    }

    if (rpBuf) ExFreePoolWithTag(rpBuf, 'CnLG');
    return status;
}
```

### 5.6 环形缓冲区数据结构与写入

```cpp
// ring_buffer.cpp

typedef struct _DELETE_EVENT_RECORD {
    ULONG           Magic;              // 0xDE1E7E54
    ULONG           RecordSize;         // 本条记录总大小
    LARGE_INTEGER   Timestamp;          // 删除时间
    ULONG           ProcessId;          // 删除进程 PID
    ULONG           DeleteType;         // 0=永久删除, 1=回收站
    ULONG           FileNameOffset;     // 文件名在记录中的偏移
    USHORT          FileNameLength;     // 文件名长度 (bytes)

    // 文件基本信息
    LARGE_INTEGER   FileSize;
    LARGE_INTEGER   AllocationSize;
    LARGE_INTEGER   CreationTime;
    LARGE_INTEGER   LastWriteTime;

    // LCN 信息
    ULONG           LCNCount;
    struct {
        ULONGLONG   StartLCN;
        ULONG       ClusterCount;
    } LCNEntries[64];                  // 最多 64 个片段

    // 文件名紧随结构体之后（变长）
    WCHAR           FullPath[1];
} DELETE_EVENT_RECORD, *PDELETE_EVENT_RECORD;


VOID WriteToRingBuffer(PDELETE_EVENT_RECORD Record) {
    KIRQL oldIrql;
    ULONG recordSize = Record->RecordSize;
    ULONG newWriteIndex;

    KeAcquireSpinLock(&g_Context.BufferLock, &oldIrql);

    newWriteIndex = g_Context.WriteIndex + recordSize;

    if (newWriteIndex >= SHARED_MEMORY_SIZE) {
        // 环绕到缓冲区开头
        newWriteIndex = recordSize;
        if (g_Context.ReadIndex < recordSize) {
            // 缓冲区满，丢弃最旧记录
            g_Context.ReadIndex = recordSize;
        }
    }

    // 检查是否会覆盖未读数据
    if (newWriteIndex > g_Context.ReadIndex &&
        g_Context.WriteIndex < g_Context.ReadIndex) {
        KeReleaseSpinLock(&g_Context.BufferLock, oldIrql);
        return;  // 缓冲区满，丢弃本条
    }

    // 写入记录
    RtlCopyMemory(
        (PUCHAR)g_Context.SharedMemoryBase + g_Context.WriteIndex,
        Record, recordSize
    );
    g_Context.WriteIndex = newWriteIndex;

    KeReleaseSpinLock(&g_Context.BufferLock, oldIrql);

    // 通知用户态有新数据
    KeSetEvent(&g_Context.DataReadyEvent, IO_NO_INCREMENT, FALSE);
}
```

### 5.7 IRP 派发（DeviceIoControl 处理）

```cpp
// dispatch.cpp

#define IOCTL_FILE_RESTORE_START_MONITORING \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_FILE_RESTORE_STOP_MONITORING \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_FILE_RESTORE_GET_EVENT_HANDLE \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x802, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_FILE_RESTORE_GET_STATS \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x803, METHOD_BUFFERED, FILE_ANY_ACCESS)

typedef struct _MONITOR_STATS {
    ULONG TotalEvents;
    ULONG DroppedEvents;
    ULONG BufferSize;
    ULONG UsedSize;
} MONITOR_STATS, *PMONITOR_STATS;

NTSTATUS DispatchIoControl(
    PDEVICE_OBJECT DeviceObject,
    PIRP           Irp
) {
    NTSTATUS status = STATUS_SUCCESS;
    PIO_STACK_LOCATION irpSp = IoGetCurrentIrpStackLocation(Irp);
    ULONG ioControlCode      = irpSp->Parameters.DeviceIoControl.IoControlCode;
    ULONG outBufLen           = irpSp->Parameters.DeviceIoControl.OutputBufferLength;
    PVOID outBuf              = Irp->AssociatedIrp.SystemBuffer;

    switch (ioControlCode) {
    case IOCTL_FILE_RESTORE_START_MONITORING:
        // 启动监控（可扩展：从输入缓冲区读取过滤条件）
        status = STATUS_SUCCESS;
        break;

    case IOCTL_FILE_RESTORE_STOP_MONITORING:
        status = STATUS_SUCCESS;
        break;

    case IOCTL_FILE_RESTORE_GET_STATS:
        if (outBufLen >= sizeof(MONITOR_STATS)) {
            PMONITOR_STATS stats = (PMONITOR_STATS)outBuf;
            stats->TotalEvents   = g_Context.TotalEvents;
            stats->DroppedEvents = g_Context.DroppedEvents;
            stats->BufferSize    = SHARED_MEMORY_SIZE;
            stats->UsedSize      = g_Context.WriteIndex - g_Context.ReadIndex;
            Irp->IoStatus.Information = sizeof(MONITOR_STATS);
        } else {
            status = STATUS_BUFFER_TOO_SMALL;
        }
        break;

    default:
        status = STATUS_INVALID_DEVICE_REQUEST;
        break;
    }

    Irp->IoStatus.Status = status;
    if (!NT_SUCCESS(status)) Irp->IoStatus.Information = 0;
    IoCompleteRequest(Irp, IO_NO_INCREMENT);
    return status;
}
```

### 5.8 用户态客户端

```cpp
// monitor_client.cpp (用户态，集成到 Filerestore_CLI)
#include <windows.h>
#include <vector>
#include <string>

struct DeleteEvent {
    std::wstring FilePath;
    uint64_t     FileSize;
    LARGE_INTEGER DeleteTime;
    bool         IsRecycled;
    struct { uint64_t StartLCN; uint32_t ClusterCount; };
    std::vector<decltype(StartLCN)> LCNRanges;  // 简化示意
};

class FileRestoreMonitorClient {
    HANDLE hDevice       = INVALID_HANDLE_VALUE;
    HANDLE hSharedMemory = NULL;
    HANDLE hEvent        = NULL;
    PVOID  pSharedBuffer = nullptr;
    SIZE_T sharedMemorySize = 0;
    ULONG  readIndex     = 0;

public:
    bool Connect() {
        // 1. 打开驱动设备
        hDevice = CreateFileW(
            L"\\\\.\\FileRestoreMon",
            GENERIC_READ | GENERIC_WRITE,
            0, NULL, OPEN_EXISTING, 0, NULL
        );
        if (hDevice == INVALID_HANDLE_VALUE) return false;

        // 2. 打开共享内存
        hSharedMemory = OpenFileMappingW(
            FILE_MAP_READ, FALSE,
            L"Local\\FileRestoreMonBuffer"
        );
        if (!hSharedMemory) {
            CloseHandle(hDevice);
            return false;
        }

        pSharedBuffer = MapViewOfFile(
            hSharedMemory, FILE_MAP_READ, 0, 0, 0
        );

        readIndex = 0;
        return true;
    }

    bool WaitForEvent(DWORD timeoutMs) {
        return WaitForSingleObject(hEvent, timeoutMs) == WAIT_OBJECT_0;
    }

    std::vector<DeleteEvent> ReadEvents() {
        std::vector<DeleteEvent> events;
        PUCHAR buffer = (PUCHAR)pSharedBuffer;

        while (readIndex < sharedMemorySize) {
            auto* record = (PDELETE_EVENT_RECORD)(buffer + readIndex);
            if (record->Magic != 0xDE1E7E54) break;

            DeleteEvent evt;
            evt.FilePath = std::wstring(
                (wchar_t*)((PUCHAR)record + record->FileNameOffset),
                record->FileNameLength / sizeof(wchar_t)
            );
            evt.FileSize   = record->FileSize.QuadPart;
            evt.DeleteTime = record->Timestamp;
            evt.IsRecycled = (record->DeleteType == 1);

            // 提取 LCN 信息
            for (ULONG i = 0; i < record->LCNCount; i++) {
                evt.LCNRanges.push_back({
                    record->LCNEntries[i].StartLCN,
                    record->LCNEntries[i].ClusterCount
                });
            }

            events.push_back(std::move(evt));
            readIndex += record->RecordSize;
        }

        ResetEvent(hEvent);
        return events;
    }

    void PersistToFile(const std::wstring& path) {
        // 将捕获的事件持久化到磁盘，供后续恢复使用
        // 格式与 MFT 快照兼容，便于统一查询
    }

    ~FileRestoreMonitorClient() {
        if (pSharedBuffer)  UnmapViewOfFile(pSharedBuffer);
        if (hSharedMemory)  CloseHandle(hSharedMemory);
        if (hEvent)         CloseHandle(hEvent);
        if (hDevice != INVALID_HANDLE_VALUE) CloseHandle(hDevice);
    }
};
```

---

## 六、关键技术难点

| 难点 | 解决方案 |
|------|---------|
| 获取 LCN 需要 FltInstance | 从 `PCFLT_RELATED_OBJECTS FltObjects` 参数获取 `FltObjects->Instance` |
| 文件名编码 | NTFS 使用 UTF-16LE，内核中使用 `UNICODE_STRING`，用户态转 `std::wstring` |
| 回收站识别 | 检测 `FileRenameInformation` 且目标路径包含 `$Recycle.Bin` |
| 高并发写入 | Per-CPU 队列 + 自旋锁（或无锁环形缓冲区） |
| 内存泄漏防护 | 使用 `ExAllocatePoolWithTag` / `ExFreePoolWithTag` 严格配对，`DriverUnload` 中清理所有资源 |
| BSOD 防护 | `__try/__except` 包裹可能失败的操作，所有指针校验后再访问 |
| 常驻文件无 LCN | `FSCTL_GET_RETRIEVAL_POINTERS` 对常驻文件返回失败，需优雅处理 |

---

## 七、两方案对比

| 维度 | MFT 快照 | 内核监控 |
|------|---------|---------|
| 开发成本 | 低（复用 MFTCache） | 高（全新开发） |
| 实时性 | 延迟（取决于快照/轮询间隔） | 实时（Pre-op 回调） |
| 捕获率 | 取决于快照频率 | ~100% |
| 大文件恢复提升 | 5% → 30-50% | 5% → 80%+ |
| 技术风险 | 低 | 中（BSOD、兼容性） |
| 部署复杂度 | 简单（纯用户态） | 需安装驱动 + 签名 |
| 现有基础 | MFTCache 可直接复用 | 全新开发 |
| 学习价值 | 低 | 很高（内核编程） |

---

## 八、实施路径

### Phase 1：MFT 快照（优先，复用现有基础设施）

```
├── 复用 MFTCache/MFTReader 读取 MFT 元数据
├── 添加快照持久化（二进制格式 + 索引）/ 加载
├── USN 触发模式：轮询 USN Journal，检测到删除后抢读 MFT
├── 修改 carvepool 签名扫描接入快照信息
└── 预计提升: 扫描速度 100x，大文件恢复率 5% → 30%+
```

### Phase 2：内核监控（学习项目，独立开发）

```
├── 搭建 WDK 开发环境，配置测试签名
├── 实现最小化微过滤驱动原型（仅捕获删除事件，打印日志）
├── 逐步添加 LCN 查询、共享内存、环形缓冲区
├── 用户态客户端集成到 Filerestore_CLI
└── 预计提升: 实时删除捕获 ~100%
```

### Phase 3：混合方案（远期）

```
├── 检测驱动是否已安装/运行
├── 有驱动 → 使用实时事件流
├── 无驱动 → 降级到 MFT 快照模式
└── 两种数据源统一存储格式，恢复逻辑无需区分来源
```

---

*文档整理自项目讨论记录，2026-02-17*
