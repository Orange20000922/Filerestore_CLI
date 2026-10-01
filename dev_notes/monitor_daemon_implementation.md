# Monitor Daemon + TUI Dashboard 实现总结

## 概述

本次实现将 `monitor` 命令从进程内 `std::thread` 模式升级为独立守护进程架构。CLI 退出后守护进程继续运行，通过共享内存 IPC 实现状态查询，支持开机自启动和 TUI 实时面板。

## 架构

```
                        共享内存 (MonitorSharedState)
                        ┌─────────────────────────┐
  CLI/TUI 进程          │  magic, version, pid    │       守护进程
  ┌──────────┐   读取   │  统计计数器             │      ┌──────────────┐
  │ monitor  │◄────────►│  环形缓冲区(最近事件)    │◄─────│ RunDaemonMain│
  │ status   │          │  daemonRunning 标志     │ 写入  │              │
  └──────────┘          └─────────────────────────┘      │ UsnDelete    │
                                                         │ Monitor      │
  Named Event                                            └──────┬───────┘
  ┌──────────┐    SetEvent                                      │
  │ StopEvent│─────────────────────────────────────────────────►│ WaitFor
  └──────────┘                                                  │ SingleObject
                                                                ▼
  Named Mutex                                              退出主循环
  ┌──────────┐
  │ Mutex    │  单例保证（同一驱动器只有一个守护进程）
  └──────────┘
```

## 新增/修改文件清单

### 新建文件

| 文件 | 说明 |
|------|------|
| `src/fileRestore/MonitorDaemon.h` | 共享内存 POD 结构体 + MonitorDaemon 类声明 |
| `src/fileRestore/MonitorDaemon.cpp` | 守护进程启停、共享内存 IPC、注册表自启动、RunDaemonMain 主循环 |

### 修改文件

| 文件 | 变更内容 |
|------|----------|
| `src/fileRestore/UsnDeleteMonitor.h` | 新增 `EventCallback` 类型和 `SetEventCallback()` |
| `src/fileRestore/UsnDeleteMonitor.cpp` | `HandleDeleteEvent` 成功/失败路径末尾调用回调 |
| `src/core/Main.cpp` | 新增 `--monitor-daemon <drive>` 参数解析，守护进程入口 |
| `src/commands/UsnRecoverCommands.cpp` | 重写 MonitorCommand，删除 `g_monitor`，改为通过 MonitorDaemon IPC |
| `src/tui/TuiApp.h` | ViewMode 枚举新增 `Monitor`，新增 `monitorDrive_` 成员 |
| `src/tui/TuiApp.cpp` | 菜单新增 "USN Delete Monitor"，Monitor Dashboard 渲染和键盘处理 |
| `src/tui/CommandHelper.cpp` | 新增 `monitor`/`snapshot`/`snapshotquery` 命令元数据 |
| `Filerestore_CLI.vcxproj` | 添加 MonitorDaemon.cpp/.h |
| `Filerestore_CLI.vcxproj.filters` | 添加 MonitorDaemon.cpp/.h 到筛选器 |

## 核心数据结构

### MonitorSharedState（共享内存，固定大小 POD）

```cpp
#pragma pack(push, 8)
struct MonitorSharedState {
    DWORD magic, version;          // 0x46524D44, 1
    char driveLetter;              // 监控的驱动器
    DWORD pid;                     // 守护进程 PID
    FILETIME startTime, lastUpdate;
    volatile LONGLONG totalEvents, capturedCount, missedCount, skippedCount, snapshotCount;
    DWORD pollIntervalMs;
    volatile LONG recentEventCount, recentEventHead;
    MonitorRecentEvent recentEvents[16];  // 环形缓冲区
    volatile LONG daemonRunning, autoStartEnabled;
};
#pragma pack(pop)
```

### 命名对象约定

| 对象 | 命名格式 | 用途 |
|------|----------|------|
| Mutex | `Global\FileRestoreMonitor_D` | 单例保证 |
| FileMapping | `Global\FileRestoreMonitor_D_Mem` | 共享内存 |
| Event | `Global\FileRestoreMonitor_D_Stop` | 停止信号（手动复位） |

## 命令接口

```
monitor <drive> start         启动守护进程（CREATE_NO_WINDOW + DETACHED_PROCESS）
monitor <drive> stop          发送停止事件，等待守护进程退出
monitor <drive> status        读取共享内存，显示统计和最近事件
monitor <drive> autostart     写入 HKCU\...\Run 注册表键
monitor <drive> unautostart   删除注册表键
```

## TUI Monitor Dashboard

### 菜单入口
主菜单第 5 项 "USN Delete Monitor"，选中后进入 `ViewMode::Monitor`。

### 面板布局
- 头部：驱动器、PID、运行状态、自启动状态、启动时间
- 统计区：Events / Captured / Missed / Skipped / Snapshots
- 事件表：最近 8 条删除事件（MFT#、大小、状态、文件名）
- 底部：快捷键提示

### 键盘快捷键
| 按键 | 功能 |
|------|------|
| S | 启动守护进程 |
| T | 停止守护进程 |
| A | 切换自启动开关 |
| Esc | 返回主菜单 |

Dashboard 每 200ms 自动刷新（复用异步刷新线程）。

## 守护进程生命周期

### 启动流程 (`StartDaemon`)
1. `OpenMutexW` 检查是否已有实例
2. `GetModuleFileNameW` 获取自身 exe 路径
3. `CreateProcessW` 以 `CREATE_NO_WINDOW | DETACHED_PROCESS` 启动子进程
4. 命令行：`"<exe_path>" --monitor-daemon D`
5. 轮询最多 3 秒等待 Mutex 出现

### 守护进程主循环 (`RunDaemonMain`)
1. `CreateMutexW` 获取单例锁
2. `CreateEventW` 创建停止事件（手动复位）
3. `CreateFileMappingW` + `MapViewOfFile` 创建共享内存
4. 初始化 `MonitorSharedState`
5. 创建 `UsnDeleteMonitor`，设置 `EventCallback`（写入环形缓冲区）
6. `CaptureExistingDeleted(24)` 捕获已有删除记录
7. `Start()` 启动后台轮询
8. 主循环：`WaitForSingleObject(stopEvent, 500)` + 同步统计到共享内存
9. 收到停止信号后：`Stop()` + 清理所有 Handle

### 停止流程 (`StopDaemon`)
1. `OpenEventW` 打开停止事件
2. `SetEvent` 通知守护进程
3. 轮询最多 5 秒等待 Mutex 消失

## 自启动注册表

- 键路径：`HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run`
- 值名称：`FileRestoreMonitor_D`（D 为驱动器字母）
- 值数据：`"<exe_path>" --monitor-daemon D`
- API：`RegOpenKeyExW` / `RegSetValueExW` / `RegDeleteValueW` / `RegQueryValueExW`

## 构建验证

Release x64 编译通过，0 error，0 warning。

```
Filerestore_CLI.vcxproj -> D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe
```

## 功能验证步骤

```bash
# CLI 验证
Filerestore_CLI.exe --cmd "monitor D start"       # 启动守护进程
tasklist | findstr Filerestore                      # 查看进程
Filerestore_CLI.exe --cmd "monitor D status"       # 查看状态
Filerestore_CLI.exe --cmd "monitor D autostart"    # 启用自启动
Filerestore_CLI.exe --cmd "monitor D stop"         # 停止守护进程

# TUI 验证
Filerestore_CLI.exe --tui                           # 进入 TUI
# 选择 "USN Delete Monitor" → Dashboard 面板
# S=启动  T=停止  A=切换自启动  Esc=返回
```
