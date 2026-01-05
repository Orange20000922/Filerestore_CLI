# Filerestore_CLI - NTFS 文件恢复工具

[![Version](https://img.shields.io/badge/version-v0.2.0-blue.svg)](https://github.com/yourusername/Filerestore_CLI)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)
[![Language](https://img.shields.io/badge/language-C%2B%2B20-orange.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-Research%20Only-red.svg)](LICENSE)

> NTFS 文件恢复工具，支持 MFT 扫描、签名搜索恢复、智能覆盖检测和多线程优化

---

## 最新更新 (2026-01-05)

### v0.2.0 - 签名搜索与线程池优化

#### 新增：签名搜索恢复 (File Carving)
- **三种扫描模式**：同步、异步I/O、线程池并行
- **14种文件类型**：ZIP/PDF/JPG/PNG/MP4/AVI/MP3/7Z/RAR 等
- **智能验证**：首字节索引 + 多重验证，减少误报
- **性能**：线程池模式可达 2500+ MB/s（16核+NVMe）

```bash
# 异步I/O扫描（默认）
carve C jpg,png,pdf D:\recovered\

# 线程池并行扫描（最快）
carvepool C all D:\recovered\

# 查看支持的文件类型
carvetypes
```

#### 性能对比（100GB磁盘扫描）

| 模式 | 命令 | 16核+NVMe |
|------|------|-----------|
| 同步 | `carve ... sync` | ~500 MB/s |
| 异步I/O | `carve ... async` | ~800 MB/s |
| **线程池** | `carvepool` | **~2500 MB/s** |

#### 其他改进
- 自动检测CPU核心数优化线程配置
- 128MB读取缓冲区优化NVMe性能
- 硬件感知配置（自适应HDD/SSD/NVMe）

---

## 核心功能

### 1. MFT 文件恢复
```bash
listdeleted C              # 列出已删除文件
searchdeleted C doc .docx  # 搜索文件
restorebyrecord C 12345 D:\out.docx  # 恢复文件
```

### 2. 签名搜索恢复 (File Carving)
```bash
carve C zip D:\recovered\           # 异步扫描ZIP文件
carvepool C jpg,png D:\recovered\   # 线程池扫描图片
carvepool D all D:\recovered\ 8     # 指定8线程扫描所有类型
```

### 3. 覆盖检测
```bash
detectoverwrite C 12345           # 检测文件是否被覆盖
detectoverwrite C 12345 fast      # 快速采样检测
```

### 4. USN 日志搜索
```bash
searchusn C document.docx         # 搜索最近删除的文件
scanusn C 24                      # 扫描最近24小时的删除记录
```

---

## 命令参考

### 文件恢复
| 命令 | 说明 |
|------|------|
| `listdeleted <drive>` | 列出已删除文件 |
| `searchdeleted <drive> <pattern> <ext>` | 搜索已删除文件 |
| `restorebyrecord <drive> <record> <output>` | 恢复文件 |
| `batchrestore <drive> <records> <dir>` | 批量恢复 |
| `forcerestore <drive> <record> <output>` | 强制恢复（跳过检测）|

### 签名搜索
| 命令 | 说明 |
|------|------|
| `carve <drive> <types> <dir> [async/sync]` | 签名扫描恢复 |
| `carvepool <drive> <types> <dir> [threads]` | 线程池并行扫描 |
| `carvetypes` | 列出支持的文件类型 |
| `carverecover <index> <output>` | 恢复扫描到的文件 |

### 诊断工具
| 命令 | 说明 |
|------|------|
| `detectoverwrite <drive> <record> [mode]` | 覆盖检测 |
| `searchusn <drive> <filename>` | USN搜索 |
| `diagnosemft <drive>` | MFT碎片诊断 |

---

## 系统要求

- **操作系统**: Windows 10/11 (x64)
- **文件系统**: NTFS
- **权限**: 管理员权限
- **推荐**: SSD/NVMe + 多核CPU（线程池优化）

---

## 快速开始

```bash
# 1. 以管理员权限运行
.\Filerestore_CLI.exe

# 2. 扫描已删除文件
listdeleted C

# 3. 签名搜索恢复
carvepool C jpg,png,pdf D:\recovered\

# 4. 查看帮助
help carvepool
```

---

## 项目结构

```
Filerestore_CLI/
├── src/
│   ├── core/           # CLI核心 (cli, climodule, Main)
│   ├── commands/       # 命令实现 (cmd.cpp)
│   ├── fileRestore/    # 文件恢复组件
│   │   ├── MFTReader/Parser/BatchReader  # MFT操作
│   │   ├── FileCarver                     # 签名搜索
│   │   ├── SignatureScanThreadPool        # 线程池扫描
│   │   ├── OverwriteDetector              # 覆盖检测
│   │   └── UsnJournalReader               # USN日志
│   ├── utils/          # 工具类 (Logger, ProgressBar)
│   └── analysis/       # PE分析 (ImageTable)
├── document/           # 技术文档
└── Filerestore_lib/    # 核心库（供扩展开发）
```

---

## 技术文档

| 文档 | 说明 |
|------|------|
| `SIGNATURE_SCAN_THREADPOOL_ANALYSIS.md` | 线程池优化方案分析 |
| `NVME_IO_IMPACT_ANALYSIS.md` | NVMe I/O影响评估 |
| `THREADPOOL_CODE_REVIEW.md` | 线程池代码审查 |
| `LANGUAGE_MIGRATION_ANALYSIS.md` | C++ vs Rust分析 |

---

## 更新日志

### v0.2.0 (2026-01-05)
- 新增签名搜索恢复（File Carving）
- 新增线程池并行扫描（carvepool）
- 支持14种文件类型签名识别
- 自动硬件检测和配置优化

### v0.1.1 (2026-01-04)
- 动态模块加载系统（DLL插件）
- 核心功能库分离（Filerestore_lib）
- 优雅退出机制
- 内存管理优化

### v0.1.0 (2025-12-31)
- MFT扫描和文件恢复
- 智能覆盖检测（多线程）
- USN日志支持
- 路径缓存优化

---

## 许可证

本项目仅用于**个人研究和学习目的**。

---

## 获取可执行文件

[GitHub Actions 构建](https://github.com/Orange20000922/Filerestore_CLI/actions)
