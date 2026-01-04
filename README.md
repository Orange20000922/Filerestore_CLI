# ImportTableAnalyzer - NTFS 文件恢复工具

[![Version](https://img.shields.io/badge/version-v0.1.1-blue.svg)](https://github.com/yourusername/ImportTableAnalyzer)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)
[![Language](https://img.shields.io/badge/language-C%2B%2B20-orange.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-Research%20Only-red.svg)](LICENSE)

> 一个功能强大的 NTFS 文件恢复工具，支持 MFT 扫描、智能覆盖检测和高性能文件恢复

---

## 🎉 最新更新 (2026-01-04)

### v0.1.1 - 架构升级与扩展性增强

#### 🔧 核心改进
- **✨ 优雅退出机制** - 替代直接 `exit(0)`，确保所有资源正确释放
  - 自动释放日志系统资源
  - 清理崩溃处理程序
  - 释放动态加载的模块
  - 防止内存泄漏

- **📁 日志系统增强** - 日志文件使用绝对路径
  - 自动保存在程序所在目录
  - 启动时显示完整日志路径
  - 支持日志轮转和自动清理

#### 🔌 扩展性功能
- **🚀 动态模块系统** - 支持运行时加载 DLL 扩展模块
  - 自动扫描 `modules\` 目录
  - 支持热插拔式命令扩展
  - 完整的模块生命周期管理
  - 详细的加载日志和错误处理

- **📚 核心功能库分离** - `Filerestore_lib` 静态库
  - 13 个公共头文件，12 个核心实现
  - 完整的 NTFS MFT 操作 API
  - 支持第三方开发者创建扩展模块
  - 详细的 API 文档和示例代码

#### 🛠️ 内存管理优化
- 修复了模块注册系统的内存泄漏
- 改进了命令参数的内存管理
- 优化了动态模块卸载流程
- 使用 `const` 引用避免不必要的复制

#### 📖 文档更新
- 新增 [`Filerestore_lib/README.md`](../Filerestore_lib/README.md) - 扩展开发指南
- 包含 4 个完整的 DLL 模块开发示例
- 详细的 API 参考和最佳实践
- 性能优化和调试技巧

#### 🔗 扩展开发支持
想为 Filerestore_CLI 开发扩展模块？查看：
- 📦 [Filerestore_lib 开发文档](../Filerestore_lib/README.md)
- 💡 示例模块：文件恢复、覆盖检测、USN 分析等
- 🛠️ 完整的项目模板代码

---

这是一个大学生的随心之作，正在持续开发中。

---

## 📋 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [命令参考](#命令参考)
- [性能优化](#性能优化)
- [测试工具](#测试工具)
- [已知问题](#已知问题)
- [开发文档](#开发文档)
- [更新日志](#更新日志)

---

## 🎯 项目简介

ImportTableAnalyzer 是一个基于 NTFS 文件系统的文件恢复工具，通过直接读取和解析 MFT (Master File Table) 来实现已删除文件的检测和恢复。

### 主要特点

- **直接 MFT 访问** - 绕过文件系统 API，直接读取 MFT 原始数据
    - 1.完全使用C++标准库和Windows系统API开发，只要在Windows7以上系统、支持C++11及以上版本的电脑运行基本不会出现依赖和兼容性问题
- **智能覆盖检测** - 多线程优化的覆盖检测引擎，自适应不同存储类型
- **高性能扫描** - 路径缓存、批量读取等多重优化，性能提升高达 7.5 倍
    - 1.以测试主机磁盘（SATA SSD）为例，目前最近的Release测试为准，扫描整个C盘逻辑卷的MFT记录（约137W条）仅需3-5分钟便可得到所有数据
    - 2.搜索功能：依靠文件缓存可在几十毫秒内完成搜索，多次扫描：路径缓存可以帮助二次扫描在1分钟内完成
    - 3.峰值内存占用300-400MB，但会及时释放不会累积，没有内存泄漏
- **完整的 CLI** - 15 个命令支持各种文件恢复和诊断场景
- **自动化测试** - PowerShell 和批处理脚本简化测试流程

---

## 🚀 核心功能

### 1. 文件恢复系统

#### 已删除文件扫描
- **命令**: `listdeleted <drive> [filter_level]`
- **功能**: 扫描并列出驱动器上所有已删除的文件
- **过滤级别**:
  - `none` - 显示所有文件（无过滤）
  - `skip_path` - 低价值文件显示为 `\LowValue\`（默认）
  - `exclude` - 完全排除低价值文件
  - `<number>` - 限制显示数量

```bash
# 扫描 C 盘所有已删除文件
listdeleted C none

# 只显示前 10000 个文件
listdeleted C 10000
```

#### 文件搜索
- **命令**: `searchdeleted <drive> <pattern> <extension> [filter_level]`
- **功能**: 按文件名和扩展名搜索已删除文件

```bash
# 搜索所有 .txt 文件
searchdeleted C * .txt

# 搜索包含 "report" 的 .docx 文件
searchdeleted C report .docx
```

#### 文件恢复
- **命令**: `restorebyrecord <drive> <record_number> <output_path>`
- **功能**: 按 MFT 记录号恢复文件，自动检测覆盖状态

```bash
# 恢复 MFT 记录号为 12345 的文件
restorebyrecord C 12345 C:\recovered\file.txt
```

**恢复流程**:
1. 自动检测文件覆盖状态
2. 显示覆盖百分比和恢复可能性
3. 如果完全覆盖，询问用户是否继续
4. 执行文件恢复

### 2. 智能覆盖检测

#### 覆盖检测命令
- **命令**: `detectoverwrite <drive> <record_number> [mode]`
- **模式**:
  - `fast` - 采样检测（1% 采样率）
  - `balanced` - 智能检测（默认）
  - `thorough` - 完整检测（100% 检测）

```bash
# 使用默认模式检测
detectoverwrite C 12345

# 快速采样检测（大文件推荐）
detectoverwrite C 12345 fast

# 完整检测（小文件推荐）
detectoverwrite C 12345 thorough
```

#### 自适应优化
系统会自动检测存储类型并应用相应优化：

| 存储类型 | 多线程 | 批量读取 | 性能提升 |
|---------|-------|---------|---------|
| HDD     | 禁用   | 启用    | +60%    |
| SATA SSD| 4 线程 | 启用    | +350%   |
| NVMe SSD| 8 线程 | 启用    | +650%   |

### 3. MFT 诊断工具

#### MFT 碎片化诊断
- **命令**: `diagnosemft <drive>`
- **功能**: 分析 MFT 碎片化状态，评估性能影响

```bash
diagnosemft C
```


### 4. PE文件分析功能

用于 PE 文件分析：

```bash
# 列出 DLL 的导出函数
printallfunc kernel32.dll

# 获取函数地址
getfuncaddr kernel32.dll CreateFileW


---

## 💻 系统要求

### 运行环境
- **操作系统**: Windows 10/11（x64）
- **文件系统**: NTFS
- **权限**: 管理员权限（必需）

### 开发环境
- **IDE**: Visual Studio 2019 或更高版本
- **编译器**: MSVC 支持 C++20
- **Windows SDK**: 10.0 或更高版本

---

## 🔧 快速开始

### 1. 编译项目

```bash
# 在 Visual Studio 中
1. 打开 Filerestore_CLI.sln
2. 选择 Release x64 配置
3. 生成 → 重新生成解决方案 (Ctrl+Shift+B)
```

### 2. 运行程序

```bash
# 以管理员权限运行
cd x64\Release
.\Filerestore_CLI.exe
```

### 3. 基础使用流程

```bash
# 步骤 1: 扫描已删除文件
listdeleted C none

# 步骤 2: 搜索特定文件
searchdeleted C document .docx

# 步骤 3: 检测覆盖状态
detectoverwrite C 12345

# 步骤 4: 恢复文件
restorebyrecord C 12345 C:\recovered\document.docx
```

---

## 📖 命令参考

### 文件恢复命令

| 命令 | 语法 | 说明 |
|------|------|------|
| `listdeleted` | `listdeleted <drive> [filter]` | 列出已删除文件 |
| `searchdeleted` | `searchdeleted <drive> <pattern> <ext> [filter]` | 搜索已删除文件 |
| `restorebyrecord` | `restorebyrecord <drive> <record> <output>` | 恢复文件 |
| `detectoverwrite` | `detectoverwrite <drive> <record> [mode]` | 检测覆盖状态 |
| `diagnosemft` | `diagnosemft <drive>` | MFT 诊断 |

### 系统命令

| 命令 | 语法 | 说明 |
|------|------|------|
| `help` | `help [command]` | 显示帮助信息 |
| `printallcommand` | `printallcommand -list` | 列出所有命令 |

### PE文件解析命令

| 命令 | 语法 | 说明 |
|------|------|------|
| `printallfunc` | `printallfunc <dll>` | 列出 DLL 导出函数 |
| `getfuncaddr` | `getfuncaddr <dll> <func>` | 获取函数地址 |

---

## ⚡ 性能优化

### 已实现的优化

#### 1. 路径重建缓存
- **原理**: 缓存已重建的目录路径
- **命中率**: 60-90%
- **性能提升**: 路径重建速度提升 5-10 倍

#### 2. 批量读取优化
- **原理**: 一次读取多个 MFT 记录
- **性能提升**: +30-50%

#### 3. 智能采样检测
- **原理**: 大文件仅检测 1% 的簇
- **性能提升**: +80-95%（大文件）
- **准确度**: 误差 < 5%

#### 4. 自适应多线程
- **HDD**: 禁用（避免随机 I/O）
- **SSD**: 4 线程
- **NVMe**: 8 线程
- **性能提升**: +150-320%（SSD/NVMe）

### 性能对比数据

#### 覆盖检测速度（1GB 文件）

| 存储类型 | 优化前 | 优化后 | 提升倍数 |
|---------|-------|--------|---------|
| HDD     | 113 分钟 | 70 分钟 | 1.6x |
| SATA SSD| 126 分钟 | 28 分钟 | 4.5x |
| NVMe SSD| 135 分钟 | 18 分钟 | 7.5x |

---

## 🧪 测试工具

项目提供了自动化测试脚本，简化测试流程。

### PowerShell 脚本（推荐）

```powershell
# 基础测试
.\test_file_recovery.ps1

# 创建多个测试文件
.\test_file_recovery.ps1 -MultipleFiles

# 自定义文件大小
.\test_file_recovery.ps1 -FileSizeKB 1024

# 自动化测试（跳过确认）
.\test_file_recovery.ps1 -SkipPrompt
```

### 批处理脚本

```cmd
# 双击运行或命令行执行
.\test_file_recovery.bat
```

### 测试流程

1. 在 `C:\Temp` 创建测试文件
2. 显示文件信息（大小、路径、时间）
3. 永久删除文件（绕过回收站）
4. 等待文件系统刷新
5. 自动启动恢复程序
6. 显示建议的测试命令

详细说明请参阅 [TEST_SCRIPT_USAGE.md](TEST_SCRIPT_USAGE.md)

---

## ⚠️ 已知问题

### 正在调查的问题

#### 1. 最近删除文件扫描不到
- **现象**: 刚删除的文件无法立即在扫描结果中找到
- **可能原因**: MFT 缓存、文件系统延迟
- **计划方案**: 实现 USN Journal 支持
- **相关文档**: [USN_JOURNAL_IMPLEMENTATION.md](USN_JOURNAL_IMPLEMENTATION.md)

### 功能限制

- ⚠️ 仅支持 NTFS 文件系统
- ⚠️ 需要管理员权限运行
- ⚠️ $Bitmap 读取未实现（计划 v0.2.0）
- ⚠️ 部分数据恢复未实现（计划 v0.3.0）

---

## 📚 开发文档

项目包含详细的开发文档，覆盖各个功能模块：

### 核心功能文档
- [FINAL_STATUS_REPORT.md](FINAL_STATUS_REPORT.md) - 项目完整功能说明
- [PROJECT_VERIFICATION_REPORT.md](PROJECT_VERIFICATION_REPORT.md) - 项目完整性验证报告
- [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) - 覆盖检测功能集成总结

### 性能优化文档
- [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) - MFT 扫描性能优化方案
- [MULTITHREADING_ANALYSIS.md](MULTITHREADING_ANALYSIS.md) - 多线程性能分析
- [MULTITHREADING_USAGE.md](MULTITHREADING_USAGE.md) - 多线程使用指南

### 功能使用文档
- [OVERWRITE_DETECTION_USAGE.md](OVERWRITE_DETECTION_USAGE.md) - 覆盖检测基础使用
- [OPTIMIZED_DETECTION_USAGE.md](OPTIMIZED_DETECTION_USAGE.md) - 优化功能使用指南
- [TEST_SCRIPT_USAGE.md](TEST_SCRIPT_USAGE.md) - 测试脚本使用说明

### 问题诊断文档
- [SEARCH_BUG_ANALYSIS.md](SEARCH_BUG_ANALYSIS.md) - 搜索功能 Bug 诊断分析
- [DUPLICATE_FUNCTION_FIX.md](DUPLICATE_FUNCTION_FIX.md) - 重复定义修复报告

### 计划功能文档
- [USN_JOURNAL_IMPLEMENTATION.md](USN_JOURNAL_IMPLEMENTATION.md) - USN Journal 实现方案
- [MULTILINGUAL_SYSTEM.md](MULTILINGUAL_SYSTEM.md) - 多语言支持系统（未集成）

---

## 📦 项目结构

```
ConsoleApplication5/
├── 核心源文件
│   ├── Main.cpp                      # 程序入口
│   ├── cli.cpp / cli.h               # 命令行解析器
│   ├── climodule.cpp / climodule.h   # 命令模块管理
│   └── cmd.cpp / cmd.h               # 所有命令实现（1191 行）
│
├── MFT 组件
│   ├── MFTReader.cpp / .h            # MFT 读取器
│   ├── MFTParser.cpp / .h            # MFT 解析器
│   ├── MFTBatchReader.cpp / .h       # 批量读取器
│   ├── MFTStructures.h               # MFT 数据结构
│   └── PathResolver.cpp / .h         # 路径解析器（带缓存）
│
├── 文件恢复组件
│   ├── FileRestore.cpp / .h          # 文件恢复主类
│   └── DeletedFileScanner.cpp / .h   # 删除文件扫描器
│
├── 覆盖检测组件
│   ├── OverwriteDetector.cpp / .h              # 覆盖检测器
│   └── OverwriteDetectionThreadPool.cpp / .h   # 多线程线程池
│
├── 辅助组件
│   ├── Logger.cpp / .h               # 日志系统
│   ├── CrashHandler.cpp / .h         # 崩溃处理
│   ├── ProgressBar.cpp / .h          # 进度条
│   └── ImageTable.cpp / .h           # PE 文件分析
│
├── 测试脚本
│   ├── test_file_recovery.ps1        # PowerShell 测试脚本
│   ├── test_file_recovery.bat        # 批处理测试脚本
│   └── check_cache.ps1               # 缓存检查脚本
│
├── 项目配置
│   ├── ConsoleApplication5.vcxproj         # VS 项目文件
│   └── ConsoleApplication5.vcxproj.filters # 文件筛选器
│
└── 文档
    ├── README.md                     # 本文件
    ├── FINAL_STATUS_REPORT.md        # 最终状态报告
    ├── INTEGRATION_SUMMARY.md        # 集成总结
    └── ... (共 13 个文档文件)
```

---

## 🔄 更新日志

### v0.1.1 (当前版本) - 2026-01-04

#### 架构改进
- ✅ 优雅退出机制（替代 exit(0)）
- ✅ 日志系统使用绝对路径
- ✅ 动态模块加载系统（DLL 插件支持）
- ✅ 核心功能库分离（Filerestore_lib）
- ✅ 内存管理优化（修复多个内存泄漏）

#### 扩展性增强
- ✅ 支持运行时加载 DLL 扩展模块
- ✅ 完整的模块生命周期管理
- ✅ 13 个公共 API 头文件
- ✅ 详细的扩展开发文档和示例

#### 代码质量
- ✅ 使用 const 引用避免不必要复制
- ✅ 改进异常处理和日志记录
- ✅ 清理冗余代码和优化结构

---

### v0.1.0 - 2025-12-31

#### 新增功能
- ✅ 完整的 MFT 扫描和文件恢复系统
- ✅ 智能覆盖检测引擎（三种模式）
- ✅ 自适应多线程优化
- ✅ 路径重建缓存系统
- ✅ 自动化测试脚本
- ✅ 15 个 CLI 命令
- ✅ 完整的帮助系统

#### 性能优化
- ✅ 路径缓存（5-10 倍提升）
- ✅ 批量读取（+30-50%）
- ✅ 采样检测（+80-95% 大文件）
- ✅ 多线程（+150-320% SSD/NVMe）

#### Bug 修复
- ✅ 修复对深度使用的系统,MFT碎片化的查询问题
- ✅ 修复命令参数定义不匹配
- ✅ 添加搜索功能诊断代码

#### 已知问题
- ⚠️ 最近删除文件扫描不到（计划 USN Journal 支持）

### 计划中的功能

#### v0.2.0 (短期)
- [ ] 修复 XML 搜索问题
- [ ] 实现 USN Journal 支持
- [ ] 实现 $Bitmap 读取
- [ ] 添加进度条显示
- [ ] 支持批量恢复命令

#### v0.3.0 (中期)
- [ ] 部分数据恢复
- [ ] 支持更多文件格式识别
- [ ] 添加恢复质量评分
- [ ] 文件诊断功能

#### v1.0.0 (长期)
- [ ] GUI 界面
- [ ] 文件格式感知恢复
- [ ] 支持其他文件系统
- [ ] 完整的多语言支持

---

## 🤝 贡献指南

如有问题或建议，欢迎通过 Issues 反馈。

---

## 📄 许可证

本项目代码仅用于**个人研究和学习目的**，请勿用于商业用途或其他未经授权的用途。

---
