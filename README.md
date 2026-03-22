# Filerestore_CLI - NTFS 文件恢复工具

[![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)](https://github.com/Orange20000922/Filerestore_CLI/releases)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)
[![Language](https://img.shields.io/badge/language-C%2B%2B20-orange.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Orange20000922/Filerestore_CLI/msbuild.yml?branch=master)](https://github.com/Orange20000922/Filerestore_CLI/actions)

**简体中文** | [English](#english-documentation)

> NTFS 文件恢复工具，支持 MFT/USN 联合恢复、实时删除监控、签名搜索、ML 文件分类、TUI 界面、内核驱动桥接和多线程优化

---

## 下载

| 版本 | 说明 | 下载 |
|------|------|------|
| **CPU 版** | 标准版，适合大多数用户 (5.6 MB) | [GitHub Releases](https://github.com/Orange20000922/Filerestore_CLI/releases) |
| **CUDA 版** | GPU 加速版，需要 NVIDIA 显卡 (186 MB) | [GitHub Releases](https://github.com/Orange20000922/Filerestore_CLI/releases) |

---

## 最新更新 (2026-02-20)

### v1.0.0 - USN 精准恢复、实时删除监控、坏簇过滤与内核驱动桥接

本次为大版本更新，新增 USN 精准恢复体系、集中式坏簇过滤读取器、实时删除监控守护进程、MFT 快照存储、内核驱动桥接（实验性），并全面优化 TUI 界面的进度显示和交互体验。

---

#### 1. USN 精准恢复体系

新增三个核心命令，基于 USN 变更日志实现精准的文件恢复：

| 命令 | 功能 |
|------|------|
| `usnlist <drive>` | 列出最近删除的文件，结合 MFT 验证和置信度评分 |
| `usnrecover <drive> <target> <output>` | 按索引/文件名/MFT 记录号恢复文件 |
| `recover <drive> [filename] [output]` | 智能恢复向导（USN + MFT + 签名联合扫描） |

- **USN + MFT 交叉验证**：通过序列号比对确认文件数据未被覆盖
- **签名回退**：当 MFT 记录被覆盖时，自动使用签名扫描搜索文件内容
- **三重验证**：USN 元数据 + MFT 数据运行 + 文件签名，给出综合置信度评分
- **批量操作**：支持按大小、扩展名过滤，批量恢复已删除文件

```bash
# 列出最近删除的文件
usnlist C

# 按索引恢复
usnrecover C 3 D:\recovered\

# 智能恢复（交互式向导）
recover C myfile.docx D:\recovered\
```

---

#### 2. MFT 快照存储

新增 `MFTSnapshotStore` 模块，在文件删除的瞬间捕获完整的 MFT 元数据快照：

- **删除时快照**：在 MFT 记录被覆盖前保存文件名、大小、数据运行、时间戳等完整元数据
- **持久化存储**：快照序列化到磁盘，重启后不丢失
- **多维查询**：支持按 MFT 记录号、序列号、文件名模式查找
- **自动过期清理**：可配置时间阈值，自动清理过期快照
- **线程安全**：全部操作支持多线程并发访问

---

#### 3. USN 删除监控

新增 `UsnDeleteMonitor` 后台监控模块，实时轮询 USN 变更日志：

- **实时监控**：持续监听文件删除事件，立即触发 MFT 快照
- **事件回调**：支持注册自定义回调函数，响应删除事件
- **定时保存**：可配置自动保存间隔，防止数据丢失
- **一次性扫描**：支持扫描已有 USN 记录，补全历史删除信息

---

#### 4. 监控守护进程

新增 `MonitorDaemon` 守护进程管理器：

- **共享内存 IPC**：通过命名共享内存与 CLI/TUI 通信，查询守护进程状态
- **事件环形缓冲**：记录最近删除事件，供 CLI/TUI 查询
- **Windows 自启动**：支持注册为开机自启动，实现 7x24 监控
- **状态查询**：PID、事件计数、最近事件列表

---

#### 5. 内核驱动桥接（实验性）

新增 `KernelBridgeClient`，可选连接 FileRestoreMon minifilter 驱动：

- **内核级通知**：通过 `FilterConnectCommunicationPort` 接收内核级删除通知
- **LCN 映射**：直接获取文件的物理簇位置，跳过 MFT 解析
- **默认禁用**：需启用 `ENABLE_KERNEL_BRIDGE` 预处理宏
- **独立分支**：驱动源码位于 `feature/kernel-driver` 分支

> 内核驱动源码（`Filerestore_sys/`）位于 `feature/kernel-driver` 分支，基于 Windows minifilter 框架开发，拦截 IRP_MJ_CREATE 的删除操作并通过通信端口转发给用户态。

---

#### 6. TUI 全面增强

- **多视图模式**：欢迎页、命令输出、参数表单、扫描进度、结果表格五种视图
- **交互式参数填充**：自动生成参数表单，可视化输入
- **FileCarver 进度同步**：6 个扫描函数（Signature Scan / Async Scan / ThreadPool Scan / ML Enhancement / ML Scan / Hybrid ML Scan）的进度全部同步到 TUI 进度条
- **命令历史与自动补全**：支持 Tab 补全和上下键历史浏览

---

#### 7. MFT 缓存 v2

- **序列号字段**：新增 `sequenceNumber` 用于删除验证
- **LCN 映射**：签名扫描结果与 MFT 记录关联
- **全局单例**：`MFTCacheManager` 跨命令共享缓存
- **有效期检查**：按缓存时间自动失效

---

#### 8. 其他改进

- **MFTParser**：新增 `EnrichWithMFT()` 方法，从 MFT 补全文件大小、时间戳等信息
- **UsnTargetedRecovery**：新增大小/扩展名静态过滤器，支持批量操作
- **MFTReader**：优化簇读取性能
- **代码重构**：移除废弃的 `cmd.cpp`，统一命令注册架构
- **多项 Bug 修复**

---

#### 9. 集中式坏簇过滤（ClusterFilteredReader）

新增 `ClusterFilteredReader` 工具类，在数据读取阶段对每个簇进行覆写检测和过滤：

- **三路径统一**：USN 定点恢复、签名雕刻恢复、API 恢复三条路径全部使用同一个过滤器
- **逐簇覆写检测**：复用 OverwriteDetector 的 $Bitmap + 熵分析 + NVMe 多线程基础设施
- **坏簇零填充**：被覆盖的簇以零字节填充，保持偏移对齐，不破坏文件结构
- **格式感知截断**：识别 PNG IEND / ZIP EOCD / JPEG FFD9 / PDF %%EOF 并截断尾部垃圾数据
- **安全阈值**：截断点不得小于文件大小的 50%，防止格式解析器被零字节误导导致灾难性截断
- **簇健康报告**：每次恢复输出 `ClusterHealthReport`，包含健康百分比、检测耗时、截断信息
- **状态自动降级**：检测到覆写簇时自动将恢复状态标记为 `PARTIAL_RECOVERY`

```bash
# 恢复输出示例（有坏簇时）
=== 部分恢复 ===
文件大小: 1048576 bytes
已保存到: D:\recovered\test.docx
簇健康: 850/1000 (85.0%) | 覆写簇: 150 | 检测: 12.3ms
```

---

## 核心功能

### 1. TUI 现代化界面 (v0.3.2+)
```bash
# 启动 TUI
Filerestore_CLI.exe --tui

# TUI 功能
- Smart Recovery: 智能恢复（MFT + USN + Signature 联合）
- Scan Deleted:   扫描已删除文件（MFT 分析）
- Deep Scan:      深度扫描（签名搜索 + ML 分类）
- Repair:         文件修复工具
- Browse Results: 浏览历史扫描结果
```

### 2. USN 精准恢复 (v1.0.0+)
```bash
usnlist C                                    # 列出最近删除文件
usnrecover C 3 D:\recovered\                 # 按索引恢复
recover C important.docx D:\recovered\       # 智能恢复向导
```

### 2. USN 精准恢复 (v1.0.0+)
```bash
usnlist C                                    # 列出最近删除文件
usnrecover C 3 D:\recovered\                 # 按索引恢复
recover C important.docx D:\recovered\       # 智能恢复向导
```

### 3. 实时删除监控 (v1.0.0+)
- 后台守护进程监听 USN 删除事件
- 文件删除瞬间自动捕获 MFT 快照
- 支持内核驱动桥接获取 LCN 映射（实验性）

### 4. MFT 文件恢复
```bash
listdeleted C              # 列出已删除文件
searchdeleted C doc .docx  # 搜索文件
restorebyrecord C 12345 D:\out.docx  # 恢复文件
```

### 5. 签名搜索恢复 (File Carving)
```bash
carve C zip D:\recovered\           # 异步扫描ZIP文件
carvepool C jpg,png D:\recovered\   # 线程池扫描图片
carvepool D all D:\recovered\ 8     # 指定8线程扫描所有类型
```

### 6. 混合扫描模式 (v0.3.0+)
```bash
# 自动选择最佳方式：有签名用签名，无签名用 ML
carvepool C all D:\recovered\

# 扫描纯文本文件（ML 模式）
carvepool C txt,html,xml D:\recovered\ 8 ml
```

---

## 性能对比

### 扫描模式（100GB 磁盘）
| 模式 | 命令 | 16核+NVMe |
|------|------|-----------|
| 同步 | `carve ... sync` | ~500 MB/s |
| 异步I/O | `carve ... async` | ~800 MB/s |
| **线程池** | `carvepool` | **~2500 MB/s** |
| **线程池+SIMD** | `carvepool` (v0.3.2) | **~2700 MB/s** ⚡ |

### SIMD 优化效果 (v0.3.2+)
| 组件 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 签名匹配 | memcmp | SSE2/AVX2 | **+50-70%** |
| 整体扫描吞吐 | 2.5 GB/s | 2.7 GB/s | **+8%** |

---

## 依赖项

### 必需依赖

#### 1. **FTXUI** - Terminal UI 框架
- **版本**: v5.0.0+
- **类型**: CMake 项目
- **用途**: TUI 界面渲染
- **状态**: 自动在 CI 中构建

**本地开发设置**:
```bash
# 克隆 FTXUI
git clone https://github.com/ArthurSonzogni/FTXUI.git Filerestore_CLI/deps/ftxui

# 构建 FTXUI
cd Filerestore_CLI/deps/ftxui
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Debug
cmake --build . --config Release
```

**GitHub Actions**: ✅ 自动克隆、构建、缓存（首次 ~5min，后续 ~2min）

---

#### 2. **nlohmann/json** - JSON 解析库
- **版本**: v3.11.0+
- **类型**: Header-only
- **用途**: 配置文件、缓存序列化
- **状态**: 已包含在仓库 (`third_party/nlohmann/json.hpp`)

---

### 可选依赖

#### 3. **ONNX Runtime** - 机器学习推理引擎
- **版本**: v1.16.0+
- **类型**: 预编译二进制包
- **用途**: ML 文件分类（txt, html, xml 等无签名文件）
- **状态**: 可选，不安装时自动禁用 ML 功能

**下载与配置**:
1. 下载: https://github.com/microsoft/onnxruntime/releases
2. 解压到 `Filerestore_CLI/deps/onnxruntime/`
3. 构建时自动检测并启用

---

### 测试依赖（开发者）

#### 4. **Google Test** - C++ 单元测试框架
- **版本**: v1.14.0
- **类型**: NuGet 包
- **用途**: 单元测试（45 个测试）
- **安装**: 自动通过 NuGet

```bash
cd Filerestore_CLI_Tests
.\build_and_test.ps1  # 自动安装 + 构建 + 测试
```

---

## 系统要求

- **操作系统**: Windows 10/11 (x64)
- **文件系统**: NTFS
- **权限**: 管理员权限
- **编译器**: Visual Studio 2022 (v143 工具集)
- **推荐**: SSD/NVMe + 多核CPU
- **可选**: NVIDIA GPU（CUDA 版，ML 加速）

---

## 构建说明

### 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/Orange20000922/Filerestore_CLI.git
cd Filerestore_CLI

# 2. 设置 FTXUI（必需）
git clone https://github.com/ArthurSonzogni/FTXUI.git Filerestore_CLI/deps/ftxui
cd Filerestore_CLI/deps/ftxui
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cd ../../../..

# 3. 构建主项目
msbuild Filerestore_CLI.vcxproj /p:Configuration=Release /p:Platform=x64

# 4. 运行
.\x64\Release\Filerestore_CLI.exe --tui
```

---

## 命令参考

### 启动选项
| 选项 | 说明 |
|------|------|
| `--tui` 或 `-t` | 启动 TUI 界面 |
| `--cmd <command>` | 非交互式命令执行（CI/CD） |
| 无参数 | 启动传统 CLI 模式 |

### 文件恢复
| 命令 | 说明 |
|------|------|
| `listdeleted <drive>` | 列出已删除文件 |
| `searchdeleted <drive> <pattern>` | 搜索文件 |
| `restorebyrecord <drive> <record> <output>` | 恢复文件 |
| `recover <drive> [filename] [output]` | 智能恢复向导（USN + MFT + 签名） |

### USN 恢复 (v1.0.0+)
| 命令 | 说明 |
|------|------|
| `usnlist <drive>` | 列出最近删除文件（含置信度评分） |
| `usnrecover <drive> <target> <output>` | 按索引/文件名/记录号恢复 |

### 签名搜索
| 命令 | 说明 |
|------|------|
| `carvepool <drive> <types> <dir> [threads]` | 线程池并行扫描 |
| `carvelist [page]` | 列出扫描结果 |
| `carverecover <index> <output>` | 恢复扫描文件 |
| `crp <dir> [options]` | 分页交互式恢复 |

---

## 支持的文件类型

### 签名扫描（14 种）
`zip` `pdf` `jpg` `png` `gif` `bmp` `mp4` `avi` `mp3` `7z` `rar` `doc` `xls` `ppt`

### ML 分类（19 种）
`jpg` `png` `gif` `bmp` `pdf` `doc` `xls` `ppt` `zip` `exe` `dll` `mp4` `mp3` `txt` `html` `xml` `json` `csv` `unknown`

---

## 项目结构

```
Filerestore_CLI/
├── src/
│   ├── tui/                       # TUI 界面 (v0.3.2+)
│   ├── commands/                   # 命令实现
│   │   └── UsnRecoverCommands.cpp  # USN 恢复命令 (v1.0.0+)
│   ├── fileRestore/               # 文件恢复核心
│   │   ├── ClusterFilteredReader.*# 集中式坏簇过滤读取器 (v1.0.0+)
│   │   ├── MFTSnapshotStore.*     # MFT 快照存储 (v1.0.0+)
│   │   ├── UsnDeleteMonitor.*     # USN 删除监控 (v1.0.0+)
│   │   ├── MonitorDaemon.*        # 监控守护进程 (v1.0.0+)
│   │   ├── KernelBridgeClient.*   # 内核驱动桥接 (v1.0.0+)
│   │   └── ...                    # MFT/签名/ML 模块
│   └── ...
├── Filerestore_CLI_Tests/         # 单元测试 (v0.3.2+)
│   ├── tests/                     # 45 个测试
│   └── build_and_test.ps1         # 测试脚本
├── deps/
│   ├── ftxui/                     # FTXUI（手动克隆）
│   └── onnxruntime/               # ONNX（可选）
└── document/                      # 技术文档

# 内核驱动（独立分支 feature/kernel-driver）
Filerestore_sys/
├── Filerestore_sys.sln
└── Filerestore_sys/
    ├── driver.c                   # 驱动入口
    ├── filter.c                   # Minifilter 回调
    ├── communication.c            # 用户态通信
    └── Filerestore_sys.inf        # 驱动安装信息
```

---

## 更新日志

### v1.0.0 (2026-02-20)
- **新增** USN 精准恢复体系（`usnlist`、`usnrecover`、`recover` 命令）
- **新增** MFT 快照存储，删除瞬间捕获完整元数据
- **新增** USN 删除监控后台守护进程
- **新增** 监控守护进程管理器（共享内存 IPC、Windows 自启动）
- **新增** 内核驱动桥接客户端（实验性，minifilter 通信）
- **新增** 集中式坏簇过滤读取器（ClusterFilteredReader）
- **新增** FileCarver 全部扫描函数 TUI 进度同步
- **改进** 三条恢复路径统一使用簇过滤，输出簇健康报告
- **改进** MFT 缓存 v2（序列号验证、全局单例、有效期检查）
- **改进** TUI 多视图模式（参数表单、扫描进度、结果表格）
- **改进** UsnTargetedRecovery 批量操作和 MFT 富化
- **重构** 统一命令注册架构，移除废弃 cmd.cpp
- **修复** JPEG 格式截断使用前向搜索导致误截断到缩略图的问题
- **修复** 多项已知问题

### v0.3.2 (2026-02-07)
- **新增** TUI 现代化界面（FTXUI）
- **新增** Google Test 单元测试（45 个）
- **新增** SIMD 签名匹配优化（+8% 吞吐）
- **新增** `--cmd` 选项自动化测试
- **新增** GitHub Actions CI/CD
- **改进** 依赖管理文档

### v0.3.1 (2026-01-07)
- **新增** `crp` 分页交互式恢复

### v0.3.0 (2026-01-07)
- **新增** ML 文件分类（ONNX）
- **新增** 混合扫描模式

---

## 开发文档

- [自动化测试指南](document/AUTO_TEST_GUIDE.md)
- [FTXUI CI 修复](document/FTXUI_CI_FIX.md)
- [依赖检查报告](document/DEPENDENCY_CHECK.md)
- [单元测试文档](Filerestore_CLI_Tests/README.md)
- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)
- [更新日志](CHANGELOG.md)

---

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

---

## 链接

- [GitHub Repository](https://github.com/Orange20000922/Filerestore_CLI)
- [Releases](https://github.com/Orange20000922/Filerestore_CLI/releases)
- [Issues](https://github.com/Orange20000922/Filerestore_CLI/issues)
- [Actions](https://github.com/Orange20000922/Filerestore_CLI/actions)

---

<a name="english-documentation"></a>

[简体中文](#filerestore_cli---ntfs-文件恢复工具) | **English**

# Filerestore_CLI - NTFS File Recovery Tool

[![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)](https://github.com/Orange20000922/Filerestore_CLI/releases)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)
[![Language](https://img.shields.io/badge/language-C%2B%2B20-orange.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Orange20000922/Filerestore_CLI/msbuild.yml?branch=master)](https://github.com/Orange20000922/Filerestore_CLI/actions)

> NTFS file recovery tool with MFT/USN joint recovery, real-time deletion monitoring, signature-based carving, ML file classification, TUI interface, kernel driver bridge, and multi-threading optimization

---

## Download

| Version | Description | Link |
|---------|-------------|------|
| **CPU Edition** | Standard version for most users (5.6 MB) | [GitHub Releases](https://github.com/Orange20000922/Filerestore_CLI/releases) |
| **CUDA Edition** | GPU-accelerated version, requires NVIDIA GPU (186 MB) | [GitHub Releases](https://github.com/Orange20000922/Filerestore_CLI/releases) |

---

## Latest Update (2026-02-20)

### v1.0.0 - USN Targeted Recovery, Real-time Deletion Monitoring, Bad Cluster Filtering & Kernel Driver Bridge

Major version update with USN-based targeted recovery, centralized bad cluster filtering reader, real-time deletion monitoring daemon, MFT snapshot storage, kernel driver bridge (experimental), and comprehensive TUI progress integration.

---

#### 1. USN Targeted Recovery System

Three new core commands based on USN change journal for precise file recovery:

| Command | Function |
|---------|----------|
| `usnlist <drive>` | List recently deleted files with MFT validation and confidence scoring |
| `usnrecover <drive> <target> <output>` | Recover by index, filename, or MFT record number |
| `recover <drive> [filename] [output]` | Smart recovery wizard (USN + MFT + signature joint scan) |

- **USN + MFT Cross-Validation**: Verify file data is not overwritten via sequence number comparison
- **Signature Fallback**: Auto-fallback to signature scanning when MFT records are overwritten
- **Triple Validation**: USN metadata + MFT data runs + file signature, with composite confidence scoring
- **Batch Operations**: Filter by size/extension, batch recover deleted files

```bash
# List recently deleted files
usnlist C

# Recover by index
usnrecover C 3 D:\recovered\

# Smart recovery (interactive wizard)
recover C myfile.docx D:\recovered\
```

---

#### 2. MFT Snapshot Store

Captures complete MFT metadata snapshots at the moment of file deletion:

- **Deletion-time Snapshots**: Save filename, size, data runs, timestamps before MFT records get overwritten
- **Persistent Storage**: Serialized to disk, survives restarts
- **Multi-dimensional Query**: Lookup by MFT record number, sequence number, or filename pattern
- **Auto-expiry Cleanup**: Configurable time threshold for automatic cleanup
- **Thread-safe**: All operations support concurrent access

---

#### 3. USN Delete Monitor

Background monitor polling the USN change journal in real-time:

- **Real-time Monitoring**: Continuously listen for file deletion events, trigger MFT snapshots immediately
- **Event Callbacks**: Register custom callbacks for deletion events
- **Periodic Auto-save**: Configurable save interval to prevent data loss
- **One-time Scan**: Scan existing USN records to fill in historical deletion info

---

#### 4. Monitor Daemon

Daemon process manager with shared memory IPC:

- **Shared Memory IPC**: Named shared memory for CLI/TUI communication and status queries
- **Event Ring Buffer**: Record recent deletion events for CLI/TUI query
- **Windows Auto-start**: Register as startup service for 24/7 monitoring
- **Status Query**: PID, event counts, recent event list

---

#### 5. Kernel Driver Bridge (Experimental)

Optional connection to the FileRestoreMon minifilter driver:

- **Kernel-level Notifications**: Receive delete notifications via `FilterConnectCommunicationPort`
- **LCN Mapping**: Direct physical cluster location, bypassing MFT parsing
- **Disabled by Default**: Requires `ENABLE_KERNEL_BRIDGE` preprocessor macro
- **Separate Branch**: Driver source code on `feature/kernel-driver` branch

---

#### 6. TUI Enhancements

- **Multi-view Modes**: Welcome, Output, Parameter Form, Scan Progress, Results Table
- **Interactive Parameter Forms**: Auto-generated forms with visual input
- **FileCarver Progress Sync**: All 6 scan functions now forward progress to TUI
- **Command History & Autocomplete**: Tab completion and arrow-key history browsing

---

#### 7. MFT Cache v2

- **Sequence Number Field**: Added for delete validation
- **LCN Mapping**: Correlate signature scan results with MFT records
- **Global Singleton**: `MFTCacheManager` shared across commands
- **Validity Check**: Auto-expire by cache age

---

#### 8. Other Improvements

- **MFTParser**: New `EnrichWithMFT()` for filling file size/timestamp info
- **UsnTargetedRecovery**: Static filters for size/extension, batch operations
- **MFTReader**: Optimized cluster read performance
- **Code Refactoring**: Removed deprecated `cmd.cpp`, unified command registration
- **Multiple Bug Fixes**

---

#### 9. Centralized Bad Cluster Filtering (ClusterFilteredReader)

New `ClusterFilteredReader` utility class performs per-cluster overwrite detection and filtering at the data reading stage:

- **Unified across all 3 recovery paths**: USN targeted recovery, signature carving, and API recovery all use the same filter
- **Per-cluster overwrite detection**: Reuses OverwriteDetector's $Bitmap + entropy analysis + NVMe multi-threading infrastructure
- **Zero-fill bad clusters**: Overwritten clusters are zero-filled to maintain offset alignment without breaking file structure
- **Format-aware truncation**: Detects PNG IEND / ZIP EOCD / JPEG FFD9 / PDF %%EOF and truncates trailing garbage
- **Safety threshold**: Truncation point must exceed 50% of file size, preventing catastrophic truncation from zero-byte-confused format parsers
- **Cluster health report**: Each recovery outputs a `ClusterHealthReport` with health percentage, detection time, and truncation info
- **Automatic status downgrade**: Recovery status is set to `PARTIAL_RECOVERY` when overwritten clusters are detected

```bash
# Recovery output example (with bad clusters)
=== Partial Recovery ===
File size: 1048576 bytes
Saved to: D:\recovered\test.docx
Cluster health: 850/1000 (85.0%) | Overwritten: 150 | Detection: 12.3ms
```

---

## Core Features

### 1. Modern TUI Interface (v0.3.2+)
```bash
# Launch TUI
Filerestore_CLI.exe --tui

# TUI Functions
- Smart Recovery: Intelligent recovery (MFT + USN + Signature combined)
- Scan Deleted:   Scan deleted files (MFT analysis)
- Deep Scan:      Deep scanning (signature search + ML classification)
- Repair:         File repair tools
- Browse Results: Browse historical scan results
```

### 2. USN Targeted Recovery (v1.0.0+)
```bash
usnlist C                                    # List recently deleted files
usnrecover C 3 D:\recovered\                 # Recover by index
recover C important.docx D:\recovered\       # Smart recovery wizard
```

### 3. Real-time Deletion Monitoring (v1.0.0+)
- Background daemon monitors USN deletion events
- Auto-capture MFT snapshots at the moment of file deletion
- Optional kernel driver bridge for LCN mapping (experimental)

### 4. MFT File Recovery
```bash
listdeleted C                       # List deleted files
searchdeleted C doc .docx           # Search files
restorebyrecord C 12345 D:\out.docx # Restore file
```

### 5. Signature-Based Carving
```bash
carve C zip D:\recovered\           # Async scan ZIP files
carvepool C jpg,png D:\recovered\   # Thread pool scan images
carvepool D all D:\recovered\ 8     # Specify 8 threads scan all types
```

### 6. Hybrid Scanning (v0.3.0+)
```bash
# Auto-select best method: signature if available, ML otherwise
carvepool C all D:\recovered\

# Scan plain text files (ML mode)
carvepool C txt,html,xml D:\recovered\ 8 ml
```

---

## Performance

### Scanning Modes (100GB Disk)
| Mode | Command | 16-core + NVMe |
|------|---------|----------------|
| Sync | `carve ... sync` | ~500 MB/s |
| Async I/O | `carve ... async` | ~800 MB/s |
| **Thread Pool** | `carvepool` | **~2500 MB/s** |
| **Thread Pool + SIMD** | `carvepool` (v0.3.2) | **~2700 MB/s** ⚡ |

### SIMD Optimization (v0.3.2+)
| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Signature Matching | memcmp | SSE2/AVX2 | **+50-70%** |
| Overall Throughput | 2.5 GB/s | 2.7 GB/s | **+8%** |

---

## Dependencies

### Required Dependencies

#### 1. **FTXUI** - Terminal UI Framework
- **Version**: v5.0.0+
- **Type**: CMake project
- **Purpose**: TUI interface rendering
- **Status**: Auto-built in CI

**Local Setup**:
```bash
# Clone FTXUI
git clone https://github.com/ArthurSonzogni/FTXUI.git Filerestore_CLI/deps/ftxui

# Build FTXUI
cd Filerestore_CLI/deps/ftxui
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Debug
cmake --build . --config Release
```

**GitHub Actions**: ✅ Auto-clone, build, cache (first time ~5min, subsequent ~2min)

---

#### 2. **nlohmann/json** - JSON Parser
- **Version**: v3.11.0+
- **Type**: Header-only library
- **Purpose**: Config files, cache serialization
- **Status**: Included in repository (`third_party/nlohmann/json.hpp`)

---

### Optional Dependencies

#### 3. **ONNX Runtime** - ML Inference Engine
- **Version**: v1.16.0+
- **Type**: Pre-compiled binary package
- **Purpose**: ML file classification (txt, html, xml, etc.)
- **Status**: Optional, auto-disabled if not installed

**Download & Setup**:
1. Download: https://github.com/microsoft/onnxruntime/releases
2. Extract to `Filerestore_CLI/deps/onnxruntime/`
3. Auto-detected during build

---

### Testing Dependencies (Developers)

#### 4. **Google Test** - C++ Unit Testing Framework
- **Version**: v1.14.0
- **Type**: NuGet package
- **Purpose**: Unit testing (45 tests)
- **Installation**: Auto via NuGet

```bash
cd Filerestore_CLI_Tests
.\build_and_test.ps1  # Auto-install + build + test
```

---

## System Requirements

- **OS**: Windows 10/11 (x64)
- **File System**: NTFS
- **Permissions**: Administrator
- **Compiler**: Visual Studio 2022 (v143 toolset)
- **Recommended**: SSD/NVMe + Multi-core CPU
- **Optional**: NVIDIA GPU (CUDA edition for ML acceleration)

---

## Build Instructions

### Quick Start

```bash
# 1. Clone project
git clone https://github.com/Orange20000922/Filerestore_CLI.git
cd Filerestore_CLI

# 2. Setup FTXUI (required)
git clone https://github.com/ArthurSonzogni/FTXUI.git Filerestore_CLI/deps/ftxui
cd Filerestore_CLI/deps/ftxui
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cd ../../../..

# 3. Build main project
msbuild Filerestore_CLI.vcxproj /p:Configuration=Release /p:Platform=x64

# 4. Run
.\x64\Release\Filerestore_CLI.exe --tui
```

---

## Command Reference

### Launch Options
| Option | Description |
|--------|-------------|
| `--tui` or `-t` | Launch TUI interface |
| `--cmd <command>` | Non-interactive command execution (CI/CD) |
| No arguments | Launch traditional CLI mode |

### File Recovery
| Command | Description |
|---------|-------------|
| `listdeleted <drive>` | List deleted files |
| `searchdeleted <drive> <pattern>` | Search files |
| `restorebyrecord <drive> <record> <output>` | Restore file |
| `recover <drive> [filename] [output]` | Smart recovery wizard (USN + MFT + signature) |

### USN Recovery (v1.0.0+)
| Command | Description |
|---------|-------------|
| `usnlist <drive>` | List recently deleted files (with confidence scoring) |
| `usnrecover <drive> <target> <output>` | Recover by index/filename/record number |

### USN Recovery (v1.0.0+)
| Command | Description |
|---------|-------------|
| `usnlist <drive>` | List recently deleted files (with confidence scoring) |
| `usnrecover <drive> <target> <output>` | Recover by index/filename/record number |

### Signature Carving
| Command | Description |
|---------|-------------|
| `carvepool <drive> <types> <dir> [threads]` | Thread pool parallel scan |
| `carvelist [page]` | List scan results |
| `carverecover <index> <output>` | Recover carved file |
| `crp <dir> [options]` | Interactive paged recovery |

---

## Supported File Types

### Signature Carving (14 types)
`zip` `pdf` `jpg` `png` `gif` `bmp` `mp4` `avi` `mp3` `7z` `rar` `doc` `xls` `ppt`

### ML Classification (19 types)
`jpg` `png` `gif` `bmp` `pdf` `doc` `xls` `ppt` `zip` `exe` `dll` `mp4` `mp3` `txt` `html` `xml` `json` `csv` `unknown`

---

## Project Structure

```
Filerestore_CLI/
├── src/
│   ├── tui/                       # TUI interface (v0.3.2+)
│   ├── commands/                   # Command implementations
│   │   └── UsnRecoverCommands.cpp  # USN recovery commands (v1.0.0+)
│   ├── fileRestore/               # Core file recovery
│   │   ├── ClusterFilteredReader.*# Centralized bad cluster filter (v1.0.0+)
│   │   ├── MFTSnapshotStore.*     # MFT snapshot storage (v1.0.0+)
│   │   ├── UsnDeleteMonitor.*     # USN delete monitor (v1.0.0+)
│   │   ├── MonitorDaemon.*        # Monitor daemon (v1.0.0+)
│   │   ├── KernelBridgeClient.*   # Kernel driver bridge (v1.0.0+)
│   │   └── ...                    # MFT/signature/ML modules
│   └── ...
├── Filerestore_CLI_Tests/         # Unit tests (v0.3.2+)
│   ├── tests/                     # 45 tests
│   └── build_and_test.ps1         # Test script
├── deps/
│   ├── ftxui/                     # FTXUI (manual clone)
│   └── onnxruntime/               # ONNX (optional)
└── document/                      # Technical documentation

# Kernel driver (separate branch: feature/kernel-driver)
Filerestore_sys/
├── Filerestore_sys.sln
└── Filerestore_sys/
    ├── driver.c                   # Driver entry
    ├── filter.c                   # Minifilter callbacks
    ├── communication.c            # User-mode communication
    └── Filerestore_sys.inf        # Driver installation info
```

---

## Changelog

### v1.0.0 (2026-02-20)
- **Added** USN targeted recovery system (`usnlist`, `usnrecover`, `recover` commands)
- **Added** MFT snapshot storage, capturing complete metadata at deletion time
- **Added** USN delete monitor background daemon
- **Added** Monitor daemon manager (shared memory IPC, Windows auto-start)
- **Added** Kernel driver bridge client (experimental, minifilter communication)
- **Added** Centralized bad cluster filtered reader (ClusterFilteredReader)
- **Added** FileCarver progress sync to TUI for all scan functions
- **Improved** All 3 recovery paths now use unified cluster filtering with health reports
- **Improved** MFT cache v2 (sequence number validation, global singleton, expiry check)
- **Improved** TUI multi-view modes (parameter forms, scan progress, results table)
- **Improved** UsnTargetedRecovery batch operations and MFT enrichment
- **Refactored** Unified command registration, removed deprecated cmd.cpp
- **Fixed** JPEG format truncation using forward search incorrectly truncating to thumbnail
- **Fixed** Multiple known issues

### v0.3.2 (2026-02-07)
- **Added** Modern TUI interface (FTXUI)
- **Added** Google Test unit testing (45 tests)
- **Added** SIMD signature matching optimization (+8% throughput)
- **Added** `--cmd` option for automation
- **Added** GitHub Actions CI/CD
- **Improved** Dependency management documentation

### v0.3.1 (2026-01-07)
- **Added** `crp` interactive paged recovery

### v0.3.0 (2026-01-07)
- **Added** ML file classification (ONNX)
- **Added** Hybrid scanning mode

---

## Documentation

- [Automated Testing Guide](document/AUTO_TEST_GUIDE.md)
- [FTXUI CI Fix](document/FTXUI_CI_FIX.md)
- [Dependency Check Report](document/DEPENDENCY_CHECK.md)
- [Unit Test Documentation](Filerestore_CLI_Tests/README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Links

- [GitHub Repository](https://github.com/Orange20000922/Filerestore_CLI)
- [Releases](https://github.com/Orange20000922/Filerestore_CLI/releases)
- [Issues](https://github.com/Orange20000922/Filerestore_CLI/issues)
- [Actions](https://github.com/Orange20000922/Filerestore_CLI/actions)
