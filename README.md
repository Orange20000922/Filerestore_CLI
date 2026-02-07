# Filerestore_CLI - NTFS 文件恢复工具

[![Version](https://img.shields.io/badge/version-v0.3.2-blue.svg)](https://github.com/Orange20000922/Filerestore_CLI/releases)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)
[![Language](https://img.shields.io/badge/language-C%2B%2B20-orange.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Orange20000922/Filerestore_CLI/msbuild.yml?branch=master)](https://github.com/Orange20000922/Filerestore_CLI/actions)

**简体中文** | [English](#english-documentation)

> NTFS 文件恢复工具，支持 MFT 扫描、签名搜索恢复、ML 文件分类、TUI 界面和多线程优化

---

## 下载

| 版本 | 说明 | 下载 |
|------|------|------|
| **CPU 版** | 标准版，适合大多数用户 (5.6 MB) | [GitHub Releases](https://github.com/Orange20000922/Filerestore_CLI/releases) |
| **CUDA 版** | GPU 加速版，需要 NVIDIA 显卡 (186 MB) | [GitHub Releases](https://github.com/Orange20000922/Filerestore_CLI/releases) |

---

## 最新更新 (2026-02-07)

### v0.3.2 - TUI 界面与测试框架

#### 🎨 新增：TUI 现代化界面
- **Terminal UI**：基于 FTXUI 的现代化终端界面
- **三区域布局**：菜单导航 | 命令输入 | 状态面板
- **交互式参数填充**：自动表单生成，可视化参数输入
- **实时进度显示**：集成原有进度条，统一渲染
- **智能恢复向导**：Smart Recovery (USN + Signature 联合扫描)

```bash
# 启动 TUI 模式
Filerestore_CLI.exe --tui

# 传统 CLI 模式
Filerestore_CLI.exe
```

**TUI 特性**：
- 📁 **快速菜单**：Smart Recovery, Scan Deleted, Deep Scan, Repair
- ⌨️ **命令模式**：支持所有 CLI 命令，Tab 自动补全，历史记录
- 📊 **状态面板**：实时显示驱动器、MFT、USN、缓存状态
- 🔄 **进度条**：无缝集成，显示扫描速度和 ETA

#### 🧪 新增：单元测试框架
- **Google Test 集成**：45 个单元测试覆盖核心功能
- **CLI 参数测试**（26个）：命令解析、参数验证、边界条件
- **SIMD 签名匹配测试**（19个）：SSE2/AVX2 优化验证
- **自动化测试脚本**：`build_and_test.ps1` 一键测试
- **CI/CD 集成**：GitHub Actions 自动运行测试

```bash
# 运行单元测试
cd Filerestore_CLI_Tests
.\build_and_test.ps1
```

#### ⚡ 性能优化：SIMD 签名匹配
- **SSE2/AVX2 加速**：签名匹配速度提升 50-70%
- **智能回退**：自动检测 CPU 特性，不支持时回退标量
- **零风险优化**：完整的单元测试验证正确性

#### 🔧 新增：自动化测试支持
- **--cmd 选项**：非交互式命令执行，支持 CI/CD
- **退出码支持**：成功返回 0，失败返回 1
- **日志系统增强**：性能指标、缓存命中率自动记录

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

### 2. MFT 文件恢复
```bash
listdeleted C              # 列出已删除文件
searchdeleted C doc .docx  # 搜索文件
restorebyrecord C 12345 D:\out.docx  # 恢复文件
```

### 3. 签名搜索恢复 (File Carving)
```bash
carve C zip D:\recovered\           # 异步扫描ZIP文件
carvepool C jpg,png D:\recovered\   # 线程池扫描图片
carvepool D all D:\recovered\ 8     # 指定8线程扫描所有类型
```

### 4. 混合扫描模式 (v0.3.0+)
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
| `recover <drive> [filename] [output]` | 智能恢复 |

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
│   ├── fileRestore/               # 文件恢复（SIMD 优化）
│   └── ...
├── Filerestore_CLI_Tests/         # 单元测试 (v0.3.2+)
│   ├── tests/                     # 45 个测试
│   └── build_and_test.ps1         # 测试脚本
├── deps/
│   ├── ftxui/                     # FTXUI（手动克隆）
│   └── onnxruntime/               # ONNX（可选）
└── document/                      # 技术文档
```

---

## 更新日志

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

[![Version](https://img.shields.io/badge/version-v0.3.2-blue.svg)](https://github.com/Orange20000922/Filerestore_CLI/releases)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)
[![Language](https://img.shields.io/badge/language-C%2B%2B20-orange.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Orange20000922/Filerestore_CLI/msbuild.yml?branch=master)](https://github.com/Orange20000922/Filerestore_CLI/actions)

> NTFS file recovery tool with MFT scanning, signature-based carving, ML file classification, TUI interface, and multi-threading optimization

---

## Download

| Version | Description | Link |
|---------|-------------|------|
| **CPU Edition** | Standard version for most users (5.6 MB) | [GitHub Releases](https://github.com/Orange20000922/Filerestore_CLI/releases) |
| **CUDA Edition** | GPU-accelerated version, requires NVIDIA GPU (186 MB) | [GitHub Releases](https://github.com/Orange20000922/Filerestore_CLI/releases) |

---

## Latest Update (2026-02-07)

### v0.3.2 - TUI Interface & Testing Framework

#### 🎨 New: Modern TUI Interface
- **Terminal UI**: Modern terminal interface based on FTXUI
- **Three-Area Layout**: Menu navigation | Command input | Status panel
- **Interactive Parameter Forms**: Auto-generated forms with visual parameter input
- **Real-time Progress**: Integrated progress bar with unified rendering
- **Smart Recovery Wizard**: USN + Signature combined scanning

```bash
# Launch TUI mode
Filerestore_CLI.exe --tui

# Traditional CLI mode
Filerestore_CLI.exe
```

**TUI Features**:
- 📁 **Quick Menu**: Smart Recovery, Scan Deleted, Deep Scan, Repair
- ⌨️ **Command Mode**: All CLI commands supported, Tab autocomplete, command history
- 📊 **Status Panel**: Real-time display of drive, MFT, USN, cache status
- 🔄 **Progress Bar**: Seamlessly integrated, shows scan speed and ETA

#### 🧪 New: Unit Testing Framework
- **Google Test Integration**: 45 unit tests covering core functionality
- **CLI Parameter Tests** (26): Command parsing, argument validation, edge cases
- **SIMD Signature Tests** (19): SSE2/AVX2 optimization verification
- **Automated Test Scripts**: One-click testing with `build_and_test.ps1`
- **CI/CD Integration**: Automatic test execution via GitHub Actions

```bash
# Run unit tests
cd Filerestore_CLI_Tests
.\build_and_test.ps1
```

#### ⚡ Performance: SIMD Signature Matching
- **SSE2/AVX2 Acceleration**: 50-70% faster signature matching
- **Smart Fallback**: Auto-detect CPU features, fallback to scalar when unsupported
- **Zero-Risk Optimization**: Comprehensive unit tests verify correctness

#### 🔧 New: Automation Support
- **--cmd Option**: Non-interactive command execution for CI/CD
- **Exit Codes**: Returns 0 on success, 1 on failure
- **Enhanced Logging**: Performance metrics and cache hit rate auto-logging

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

### 2. MFT File Recovery
```bash
listdeleted C                       # List deleted files
searchdeleted C doc .docx           # Search files
restorebyrecord C 12345 D:\out.docx # Restore file
```

### 3. Signature-Based Carving
```bash
carve C zip D:\recovered\           # Async scan ZIP files
carvepool C jpg,png D:\recovered\   # Thread pool scan images
carvepool D all D:\recovered\ 8     # Specify 8 threads scan all types
```

### 4. Hybrid Scanning (v0.3.0+)
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
| `recover <drive> [filename] [output]` | Smart recovery |

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
│   ├── fileRestore/               # File recovery (SIMD optimized)
│   └── ...
├── Filerestore_CLI_Tests/         # Unit tests (v0.3.2+)
│   ├── tests/                     # 45 tests
│   └── build_and_test.ps1         # Test script
├── deps/
│   ├── ftxui/                     # FTXUI (manual clone)
│   └── onnxruntime/               # ONNX (optional)
└── document/                      # Technical documentation
```

---

## Changelog

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

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Links

- [GitHub Repository](https://github.com/Orange20000922/Filerestore_CLI)
- [Releases](https://github.com/Orange20000922/Filerestore_CLI/releases)
- [Issues](https://github.com/Orange20000922/Filerestore_CLI/issues)
- [Actions](https://github.com/Orange20000922/Filerestore_CLI/actions)
