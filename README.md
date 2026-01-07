# Filerestore_CLI - NTFS 文件恢复工具

[![Version](https://img.shields.io/badge/version-v0.3.1-blue.svg)](https://github.com/Orange20000922/Filerestore_CLI/releases)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)
[![Language](https://img.shields.io/badge/language-C%2B%2B20-orange.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> NTFS 文件恢复工具，支持 MFT 扫描、签名搜索恢复、ML 文件分类和多线程优化

---

## 下载

| 版本 | 说明 | 下载 |
|------|------|------|
| **CPU 版** | 标准版，适合大多数用户 (5.6 MB) | [Filerestore_CLI_v0.3.1_x64.zip](https://github.com/Orange20000922/Filerestore_CLI/releases) |
| **CUDA 版** | GPU 加速版，需要 NVIDIA 显卡 (186 MB) | [Filerestore_CLI_v0.3.1_x64_cuda.zip](https://github.com/Orange20000922/Filerestore_CLI/releases) |

---

## 最新更新 (2026-01-07)

### v0.3.1 - 分页交互式恢复

#### 新增：`crp` 分页交互式恢复命令
- **逐页浏览**：用户可逐页检查扫描结果，决定是否恢复
- **选择性恢复**：支持恢复整页或指定索引的文件
- **自动清理**：输出文件夹文件数达到阈值时提示清理
- **强制恢复**：`f` 命令允许恢复低置信度（<30%）文件

```bash
# 基本用法
crp D:\recovered\

# 自定义参数
crp D:\recovered\ minconf=30 pagesize=20 autoclean=100

# 交互命令：r=恢复, f=强制, n=下页, p=上页, c=清空, q=退出
```

---

### v0.3.0 - ML 文件分类与代码质量改进

#### 新增：ML 文件类型分类
- **ONNX 神经网络模型**：98% 准确率的文件类型识别
- **261 维特征提取**：字节频率分布 + 熵 + 统计特征
- **支持无签名文件**：txt、html、xml 等纯文本文件
- **混合扫描模式**：签名扫描 + ML 扫描自动融合

```bash
# 混合模式（默认）- 签名 + ML 融合
carvepool C all D:\recovered\

# 纯签名模式 - 仅使用文件头签名
carvepool C jpg,png D:\recovered\ 8 sig

# 纯 ML 模式 - 仅使用神经网络
carvepool C txt,html D:\recovered\ 8 ml
```

#### 代码质量改进 (P0)
- **内存安全**：原始指针全部替换为 `unique_ptr`
- **线程安全**：`CarvingStats` 使用 `atomic` 成员
- **RAII 资源管理**：自动释放，防止内存泄漏

#### ML 模型信息
| 指标 | 值 |
|------|-----|
| 模型架构 | 3层全连接 (261→512→256→19) |
| 训练准确率 | 98.23% |
| 验证准确率 | 97.85% |
| 支持类型 | 19 种 (jpg, png, pdf, doc, txt...) |

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

### 3. 混合扫描模式 (v0.3.0+)
```bash
# 自动选择最佳方式：有签名用签名，无签名用 ML
carvepool C all D:\recovered\

# 扫描纯文本文件（ML 模式）
carvepool C txt,html,xml D:\recovered\ 8 ml
```

### 4. 覆盖检测
```bash
detectoverwrite C 12345           # 检测文件是否被覆盖
detectoverwrite C 12345 fast      # 快速采样检测
```

### 5. 分页交互式恢复 (v0.3.1+)
```bash
crp D:\recovered\                           # 进入交互式恢复模式
crp D:\recovered\ minconf=30 pagesize=20    # 自定义置信度和页面大小
crp D:\recovered\ autoclean=100 all         # 设置自动清理阈值，显示所有文件

# 交互命令
# r          - 恢复当前页所有文件
# r 0 2 4    - 恢复指定索引的文件
# f 0 1      - 强制恢复低置信度文件
# n/p        - 下一页/上一页
# c          - 清空输出文件夹
# q          - 退出
```

### 6. USN 日志搜索
```bash
searchusn C document.docx         # 搜索最近删除的文件
scanusn C 24                      # 扫描最近24小时的删除记录
```

---

## 性能对比

### 扫描模式（100GB 磁盘）
| 模式 | 命令 | 16核+NVMe |
|------|------|-----------|
| 同步 | `carve ... sync` | ~500 MB/s |
| 异步I/O | `carve ... async` | ~800 MB/s |
| **线程池** | `carvepool` | **~2500 MB/s** |

### 扫描策略
| 策略 | 参数 | 说明 |
|------|------|------|
| 混合模式 | `hybrid`（默认） | 签名 + ML 融合，最全面 |
| 签名模式 | `sig` | 仅签名扫描，最快速 |
| ML 模式 | `ml` | 仅 ML 扫描，支持无签名文件 |

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
| `carvepool <drive> <types> <dir> [threads] [mode]` | 线程池并行扫描 |
| `carvetypes` | 列出支持的文件类型 |
| `carverecover <index> <output>` | 恢复扫描到的文件 |
| `crp <dir> [options]` | 分页交互式恢复 |
| `carvelist [page]` | 列出扫描结果 |

### 诊断工具
| 命令 | 说明 |
|------|------|
| `detectoverwrite <drive> <record> [mode]` | 覆盖检测 |
| `searchusn <drive> <filename>` | USN搜索 |
| `diagnosemft <drive>` | MFT碎片诊断 |

---

## 支持的文件类型

### 签名扫描（14 种）
`zip` `pdf` `jpg` `png` `gif` `bmp` `mp4` `avi` `mp3` `7z` `rar` `doc` `xls` `ppt`

### ML 分类（19 种）
`jpg` `png` `gif` `bmp` `pdf` `doc` `xls` `ppt` `zip` `exe` `dll` `mp4` `mp3` `txt` `html` `xml` `json` `csv` `unknown`

---

## 系统要求

- **操作系统**: Windows 10/11 (x64)
- **文件系统**: NTFS
- **权限**: 管理员权限
- **推荐**: SSD/NVMe + 多核CPU
- **可选**: NVIDIA GPU（CUDA 版）

---

## 构建说明

> **注意**: 仓库不包含 ONNX Runtime 运行库，需要手动下载后才能编译。

### 依赖项

| 依赖 | 版本 | 说明 |
|------|------|------|
| Visual Studio | 2022+ | C++20 支持 |
| ONNX Runtime | 1.16+ | ML 推理引擎 |
| Windows SDK | 10.0+ | Windows API |

### 配置 ONNX Runtime

1. **下载 ONNX Runtime**
   - 官网: https://github.com/microsoft/onnxruntime/releases
   - 选择 `onnxruntime-win-x64-1.16.x.zip` (CPU) 或 `onnxruntime-win-x64-gpu-1.16.x.zip` (CUDA)

2. **放置文件到项目目录**
   ```
   Filerestore_CLI/deps/onnxruntime/
   ├── include/           # 头文件 (onnxruntime_cxx_api.h 等)
   ├── lib/               # 库文件 (onnxruntime.lib)
   └── README.md
   ```

3. **复制 DLL 到输出目录**
   ```
   x64/Release/
   ├── onnxruntime.dll                    # 必需
   ├── onnxruntime_providers_cuda.dll     # CUDA 版可选
   ├── onnxruntime_providers_shared.dll   # CUDA 版可选
   └── onnxruntime_providers_tensorrt.dll # CUDA 版可选
   ```

### 构建命令

```powershell
# Release 构建
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe' `
  Filerestore_CLI.vcxproj /p:Configuration=Release /p:Platform=x64

# 或使用 Visual Studio 打开 .sln 文件直接构建
```

### ML 模型训练（可选）

> 仅当需要重新训练文件分类模型时才需要配置 Python 环境。
> 预训练模型已包含在 Release 包中，普通用户无需训练。

**Python 依赖** (`ml/requirements.txt`):

| 依赖 | 版本 | 用途 |
|------|------|------|
| torch | ≥2.8.0 | PyTorch 深度学习框架 |
| numpy | ≥1.24.0 | 数值计算 |
| scikit-learn | ≥1.3.0 | 数据集划分、评估指标 |
| tqdm | ≥4.65.0 | 训练进度条 |
| matplotlib | ≥3.7.0 | 训练曲线可视化 |
| seaborn | ≥0.12.0 | 混淆矩阵可视化 |
| onnx | ≥1.14.0 | ONNX 模型格式 |
| onnxruntime | ≥1.16.0 | ONNX 推理验证 |

**训练脚本** (`ml/src/`):

| 脚本 | 用途 |
|------|------|
| `train.py` | 模型训练主脚本 |
| `model.py` | 神经网络架构定义 (261→512→256→19) |
| `dataset.py` | 数据集加载和预处理 |
| `config.py` | 训练超参数配置 |
| `export_onnx.py` | PyTorch → ONNX 模型导出 |
| `cpp_dataset_loader.py` | 加载 C++ 生成的数据集 |

**训练流程**:

```bash
# 1. 创建虚拟环境
python -m venv pytorch_env
pytorch_env\Scripts\activate

# 2. 安装依赖
pip install -r ml/requirements.txt

# 3. 准备数据集（使用 mlscan 命令生成）
Filerestore_CLI.exe
> mlscan C:\samples\ ml\data\training_data.npz

# 4. 训练模型
python ml/src/train.py

# 5. 导出 ONNX
python ml/src/export_onnx.py
```

---

## 快速开始

```bash
# 1. 以管理员权限运行
.\Filerestore_CLI.exe

# 2. 扫描已删除文件
listdeleted C

# 3. 签名搜索恢复（自动 ML 增强）
carvepool C jpg,png,pdf D:\recovered\

# 4. 恢复纯文本文件
carvepool C txt,html D:\recovered\ 8 ml

# 5. 查看帮助
help carvepool
```

---

## 项目结构

```
Filerestore_CLI/
├── src/
│   ├── core/           # CLI核心 (cli, climodule, Main)
│   ├── commands/       # 命令实现
│   ├── fileRestore/    # 文件恢复组件
│   │   ├── MFTReader/Parser        # MFT 操作
│   │   ├── FileCarver              # 签名搜索
│   │   ├── SignatureScanThreadPool # 线程池扫描
│   │   ├── MLClassifier            # ML 文件分类 (NEW)
│   │   ├── OverwriteDetector       # 覆盖检测
│   │   └── UsnJournalReader        # USN 日志
│   └── utils/          # 工具类
├── ml/                 # ML 训练代码
│   ├── train_model.py  # PyTorch 训练脚本
│   └── models/         # 导出的 ONNX 模型
└── document/           # 技术文档
```

---

## 更新日志

### v0.3.1 (2026-01-07)
- **新增** `crp` 分页交互式恢复命令
- **新增** 输出文件夹自动清理功能
- **新增** 低置信度文件强制恢复选项
- **新增** 多语言支持（中/英文）

### v0.3.0 (2026-01-07)
- **新增** ML 文件类型分类（ONNX Runtime）
- **新增** 混合扫描模式（签名 + ML 融合）
- **新增** 支持 txt/html/xml 无签名文件检测
- **改进** 内存安全：`unique_ptr` 替换原始指针
- **改进** 线程安全：`CarvingStats` 原子化
- **修复** carverecover 目录创建错误

### v0.2.0 (2026-01-05)
- 新增签名搜索恢复（File Carving）
- 新增线程池并行扫描（carvepool）
- 支持 14 种文件类型签名识别
- 自动硬件检测和配置优化

### v0.1.1 (2026-01-04)
- 动态模块加载系统（DLL 插件）
- 核心功能库分离（Filerestore_lib）
- 优雅退出机制
- 内存管理优化

### v0.1.0 (2025-12-31)
- MFT 扫描和文件恢复
- 智能覆盖检测（多线程）
- USN 日志支持
- 路径缓存优化

---

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

---

## 链接

- [GitHub Releases](https://github.com/Orange20000922/Filerestore_CLI/releases)
- [问题反馈](https://github.com/Orange20000922/Filerestore_CLI/issues)
