# ML云端训练完整工作流

本文档描述了使用 腾讯云COS + Google Colab 训练连续性检测模型的完整工作流程。

## 架构概述

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据存储策略                              │
├─────────────────────────────────────────────────────────────────┤
│  腾讯云 COS (500GB)      │  本地 (临时)      │  Google Colab    │
│  ─────────────────       │  ────────────     │  ─────────────   │
│  • 原始文件长期存储       │  • 临时处理空间    │  • 训练环境      │
│  • Govdocs/FMA/Pexels    │  • 用完即删       │  • 挂载COS      │
│  • CSV数据集             │  • 不长期占用      │  • 保存模型      │
│  • ONNX模型              │                   │                 │
└─────────────────────────────────────────────────────────────────┘

数据流：
  数据源 ──→ 腾讯云COS ──→ 本地(临时) ──→ C++提取 ──→ CSV ──→ COS ──→ Colab
             (500GB)       (按需同步)                         ↓
                                                          训练 ──→ ONNX ──→ COS ──→ 本地部署
```

## 支持的文件格式

| 类别 | 格式 |
|------|------|
| ZIP-based | zip, docx, xlsx, pptx, jar, apk |
| 音频 | mp3, wav, flac, ogg, m4a |
| 视频 | mp4, mov, avi, mkv, webm, 3gp |
| 图片 | jpg, jpeg, png, gif, bmp |

---

## 第一步：安装依赖

### 1.1 安装 Python 依赖

```bash
cd scripts
pip install -r requirements.txt
```

这会安装：
- `cos-python-sdk-v5` - 腾讯云 COS SDK
- `requests` - HTTP 请求
- `tqdm` - 进度条

### 1.2 安装 ML 训练依赖

```bash
cd ml
pip install -r requirements.txt
```

---

## 第二步：配置腾讯云 COS

### 2.1 获取密钥

1. 登录 [腾讯云控制台](https://console.cloud.tencent.com/)
2. 进入 [API密钥管理](https://console.cloud.tencent.com/cam/capi)
3. 创建或查看 SecretId 和 SecretKey

### 2.2 获取 Bucket 信息

1. 进入 [对象存储控制台](https://console.cloud.tencent.com/cos/bucket)
2. 记录：
   - **Bucket名称**: 如 `ml-training-1234567890`
   - **所属地域**: 如 `ap-guangzhou`

### 2.3 配置 storage.py

配置会保存到 `~/.filerestore/storage.json`，下次使用时自动读取，无需重新输入。

#### 方式一：从JSON文件导入（推荐）

1. 创建配置文件 `cos_config.json`：

```json
{
  "tencent": {
    "region": "ap-guangzhou",
    "access_key": "AKIDxxxxxxxxxxxxxxxx",
    "secret_key": "xxxxxxxxxxxxxxxxxxxxxxxx",
    "bucket": "ml-training-1234567890"
  }
}
```

2. 导入配置：

```bash
python scripts/storage.py config-import cos_config.json
```

> **提示**：项目中提供了示例文件 `scripts/cos_config_example.json`，可以复制后修改使用。

#### 方式二：使用模板生成

```bash
# 生成腾讯云配置模板
python scripts/storage.py config-template -o cos_config.json -b tencent

# 编辑 cos_config.json，填入你的凭据
notepad cos_config.json

# 导入配置
python scripts/storage.py config-import cos_config.json
```

#### 方式三：交互式配置

```bash
python scripts/storage.py config tencent

# 按提示输入：
# SecretId: AKIDxxxxxxxxxxxxxxxx
# SecretKey: xxxxxxxxxxxxxxxxxxxxxxxx
# Region: ap-guangzhou
# Bucket name: ml-training-1234567890
```

#### 配置管理命令

| 命令 | 说明 |
|------|------|
| `config-import <file>` | 从JSON文件导入配置 |
| `config-import <file> --merge` | 合并导入（保留现有配置） |
| `config-export <file>` | 导出配置到JSON文件 |
| `config-export <file> --mask-secrets` | 导出时隐藏密钥 |
| `config-show` | 显示当前配置（密钥会脱敏显示） |
| `config-template` | 生成配置模板文件 |

```bash
# 查看当前配置
python scripts/storage.py config-show

# 备份配置
python scripts/storage.py config-export backup.json
```

### 2.4 测试连接

```bash
python scripts/storage.py test tencent
```

输出示例：
```
Testing tencent...
Connection successful!
Found 0 items in root
Total storage used: 0.0 B
```

---

## 第三步：下载原始数据到腾讯云 COS

### 方式一：分批上传（推荐，节省本地磁盘）

使用 `batch` 命令自动完成：下载 → 上传 → 删除 → 重复

```bash
# Govdocs: 下载 001-100（约50GB），每10个包为一批
python scripts/storage.py batch tencent govdocs --start 1 --end 100 --batch-range 10

# FMA: 下载 medium 子集（22GB）
python scripts/storage.py batch tencent fma --subset medium

# Pexels: 下载 1000 个视频，每 100 个为一批
python scripts/storage.py batch tencent pexels --count 1000 --batch-count 100

# Archive: 下载 500 个音频文件
python scripts/storage.py batch tencent archive --collection audio --count 500
```

**分批上传流程：**
```
┌─────────────────────────────────────────────────────┐
│  [Batch 1]                                          │
│  1. 下载 govdocs 001-010 → D:\temp\ (~5GB)          │
│  2. 上传到 tencent:ML/Govdocs/                      │
│  3. 删除 D:\temp\                                   │
│                                                     │
│  [Batch 2]                                          │
│  1. 下载 govdocs 011-020 → D:\temp\ (~5GB)          │
│  2. 上传到 tencent:ML/Govdocs/                      │
│  3. 删除 D:\temp\                                   │
│                                                     │
│  ... 重复直到完成 ...                                │
└─────────────────────────────────────────────────────┘

本地磁盘占用：始终 < 10GB
```

### 方式二：手动下载后上传

```bash
# 创建临时目录
mkdir D:\temp\downloads

# 1. 下载 Govdocs (每个包约500MB)
python scripts/download_govdocs.py download --range 001-020 --dest D:\temp\downloads\govdocs

# 2. 下载 FMA 音频 (small=7.2GB)
python scripts/download_fma.py download --subset small --dest D:\temp\downloads\fma

# 3. 下载 Pexels 视频
python scripts/download_pexels.py auth --key YOUR_PEXELS_API_KEY
python scripts/download_pexels.py download --query diverse --count 500 --dest D:\temp\downloads\pexels

# 4. 下载 Internet Archive
python scripts/download_archive.py download --collection audio --count 300 --dest D:\temp\downloads\archive

# 上传各个目录
python scripts/storage.py upload tencent D:\temp\downloads\govdocs ML/Govdocs/
python scripts/storage.py upload tencent D:\temp\downloads\fma ML/FMA/
python scripts/storage.py upload tencent D:\temp\downloads\pexels ML/Pexels/
python scripts/storage.py upload tencent D:\temp\downloads\archive ML/Archive/

# 查看上传结果
python scripts/storage.py list tencent ML/
python scripts/storage.py size tencent ML/

# 清理本地临时文件
rmdir /s /q D:\temp\downloads
```

### COS 目录结构（建议）

```
cos://ml-training-1234567890/
└── ML/
    ├── Govdocs/           # ~10GB (20个包)
    │   ├── 001/
    │   ├── 002/
    │   └── ...
    ├── FMA/               # ~7.2GB (small)
    │   └── *.mp3
    ├── Pexels/            # ~5GB (500个视频)
    │   └── *.mp4
    ├── Archive/           # ~3GB
    │   └── *.mp3
    ├── datasets/          # CSV数据集
    │   └── continuity_dataset.csv
    └── models/            # 训练好的模型
        └── continuity/
            ├── continuity_cnn.onnx
            └── continuity_cnn.json
```

---

## 第四步：从 COS 同步数据到本地（按需）

每次训练前，同步需要的部分数据到本地。

```bash
# 创建临时工作目录
mkdir D:\temp\ml_training

# 下载 ZIP 文件
python scripts/storage.py download tencent ML/Govdocs/ D:\temp\ml_training\govdocs

# 下载 MP3 文件
python scripts/storage.py download tencent ML/FMA/ D:\temp\ml_training\fma

# 下载 MP4 文件
python scripts/storage.py download tencent ML/Pexels/ D:\temp\ml_training\pexels

# 查看本地文件
dir /s D:\temp\ml_training
```

---

## 第五步：C++ 特征提取

使用 C++ 提取器生成 64 维连续性特征。

```bash
# 启动 CLI
D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe

# 提取特征（在CLI内执行）
> mlscan D:\temp\ml_training --continuity --output D:\temp\continuity_dataset.csv --samples-per-file=10

# 或限制最大样本数
> mlscan D:\temp\ml_training --continuity --output D:\temp\continuity_dataset.csv --max=50000
```

### 输出 CSV 格式

```csv
f0,f1,f2,...,f63,is_continuous,file_type,sample_type
0.823,0.412,...,0.156,1,mp3,same_file
0.654,0.321,...,0.089,0,zip,different_files
```

### 备选：Python 特征提取器

```bash
python ml/continuity/feature_extractor.py D:\temp\ml_training \
    --output D:\temp\continuity_dataset.csv \
    --samples 10
```

---

## 第六步：上传 CSV 到腾讯云 COS

```bash
# 上传数据集
python scripts/storage.py upload tencent D:\temp\continuity_dataset.csv ML/datasets/continuity_dataset.csv

# 验证上传
python scripts/storage.py list tencent ML/datasets/
```

---

## 第七步：清理本地临时文件

```bash
# 删除临时训练数据
rmdir /s /q D:\temp\ml_training

# CSV 确认上传成功后也可删除
del D:\temp\continuity_dataset.csv
```

---

## 第八步：Colab 训练

### 8.1 在 Colab 中安装 COS SDK

```python
# Cell 1: 安装依赖
!pip install cos-python-sdk-v5 torch pandas numpy onnx

# Cell 2: 配置 COS
from qcloud_cos import CosConfig, CosS3Client

config = CosConfig(
    Region='ap-guangzhou',  # 你的区域
    SecretId='AKIDxxxxxxxx',  # 你的 SecretId
    SecretKey='xxxxxxxx',     # 你的 SecretKey
)
client = CosS3Client(config)
bucket = 'ml-training-1234567890'  # 你的 Bucket
```

### 8.2 下载数据集

```python
# Cell 3: 下载数据集到 Colab
client.download_file(
    Bucket=bucket,
    Key='ML/datasets/continuity_dataset.csv',
    DestFilePath='/content/dataset.csv'
)

import pandas as pd
df = pd.read_csv('/content/dataset.csv')
print(f"Loaded {len(df)} samples")
```

### 8.3 训练模型

```python
# Cell 4: 训练（使用现有的训练脚本）
# ... 训练代码 ...
```

### 8.4 上传模型到 COS

```python
# Cell 5: 上传训练好的模型
client.upload_file(
    Bucket=bucket,
    Key='ML/models/continuity/continuity_cnn.onnx',
    LocalFilePath='/content/model.onnx'
)

client.upload_file(
    Bucket=bucket,
    Key='ML/models/continuity/continuity_cnn.json',
    LocalFilePath='/content/model_meta.json'
)

print("Model uploaded to COS!")
```

### 输出文件

训练完成后，COS 中将包含：
```
ML/models/continuity/
├── continuity_cnn.onnx      # ONNX 模型 (~200KB)
└── continuity_cnn.json      # 元数据（含标准化参数）
```

---

## 第九步：下载模型

```bash
# 创建本地模型目录
mkdir models\continuity

# 下载模型
python scripts/storage.py download tencent ML/models/continuity/ models/continuity/

# 查看下载的文件
dir models\continuity
```

---

## 第十步：部署到 C++

```bash
# 创建目标目录
mkdir Filerestore_CLI\x64\Release\models\continuity

# 复制模型文件
copy models\continuity\continuity_cnn.onnx Filerestore_CLI\x64\Release\models\continuity\
copy models\continuity\continuity_cnn.json Filerestore_CLI\x64\Release\models\continuity\

# 验证部署
Filerestore_CLI.exe

# 测试 carve 命令使用 continuity 模式
> carve D: zip D:\output 0 continuity

# 或使用 carvepool（多线程）
> carvepool D zip D:\output 0 continuity
```

---

## 完整流程示例

### 快速开始（推荐配置 ~50GB）

```bash
# ============================================================
# 一次性设置（首次使用）
# ============================================================

# 1. 安装依赖
pip install -r scripts/requirements.txt
pip install -r ml/requirements.txt

# 2. 配置腾讯云 COS
python scripts/storage.py config tencent

# 3. 测试连接
python scripts/storage.py test tencent

# 4. 分批上传数据（自动下载→上传→删除→重复）
python scripts/storage.py batch tencent govdocs --start 1 --end 50 --batch-range 10
python scripts/storage.py batch tencent fma --subset medium
python scripts/storage.py batch tencent pexels --count 500 --batch-count 100

# 5. 查看已上传数据
python scripts/storage.py size tencent ML/

# ============================================================
# 每次训练流程（可重复执行）
# ============================================================

# 6. 从 COS 下载数据到本地
mkdir D:\temp\ml_training
python scripts/storage.py download tencent ML/Govdocs/ D:\temp\ml_training\govdocs
python scripts/storage.py download tencent ML/FMA/ D:\temp\ml_training\fma

# 7. C++ 特征提取
Filerestore_CLI.exe
> mlscan D:\temp\ml_training\govdocs(or other directory) --continuity --output=cont.csv --pos-neg-ratio=0.5 --samples-per-file=50 --incremental
> exit

# 8. 上传 CSV 到 COS
python scripts/storage.py upload tencent D:\temp\dataset.csv ML/datasets/dataset.csv

# 9. 清理本地临时文件
rmdir /s /q D:\temp\ml_training

# 10. 在 Colab 中训练（手动操作）
#     - 从 COS 下载 CSV
#     - 训练模型
#     - 上传 ONNX 到 COS

# 11. 下载训练好的模型
python scripts/storage.py download tencent ML/models/continuity/ ./models/continuity/

# 12. 部署
copy models\continuity\continuity_cnn.onnx Filerestore_CLI\x64\Release\models\continuity\
copy models\continuity\continuity_cnn.json Filerestore_CLI\x64\Release\models\continuity\
```

### 完整配置（填满 500GB）

```bash
# 分批上传所有数据源（可能需要几天时间）

# Govdocs: 400个包，每10个一批
python scripts/storage.py batch tencent govdocs --start 1 --end 400 --batch-range 10

# FMA large: 93GB
python scripts/storage.py batch tencent fma --subset large

# Pexels: 5000个视频
python scripts/storage.py batch tencent pexels --count 5000 --batch-count 200

# Archive audio: 5000个文件
python scripts/storage.py batch tencent archive --collection audio --count 5000

# Archive movies: 1000个文件
python scripts/storage.py batch tencent archive --collection movies --count 1000

# 查看总大小
python scripts/storage.py size tencent ML/
```

---

## 存储空间估算

### 基础配置（~25GB，快速验证）

| 内容 | 估算大小 |
|------|----------|
| Govdocs (20包) | ~10 GB |
| FMA small | ~7.2 GB |
| Pexels (500视频) | ~5 GB |
| Archive | ~3 GB |
| CSV 数据集 | ~50 MB |
| ONNX 模型 | ~1 MB |
| **总计** | **~25 GB** |

### 推荐配置（~50GB，最佳性价比）

| 内容 | 命令 | 大小 |
|------|------|------|
| Govdocs 001-050 | `batch tencent govdocs --start 1 --end 50` | ~25 GB |
| FMA medium | `batch tencent fma --subset medium` | ~22 GB |
| Pexels | `batch tencent pexels --count 500` | ~5 GB |
| **总计** | | **~52 GB** |

### 完整配置（~500GB，填满套餐）

| 内容 | 命令 | 大小 |
|------|------|------|
| Govdocs 001-400 | `batch tencent govdocs --start 1 --end 400` | ~200 GB |
| FMA large | `batch tencent fma --subset large` | ~93 GB |
| Pexels | `batch tencent pexels --count 5000` | ~50 GB |
| Archive audio | `batch tencent archive --collection audio --count 5000` | ~50 GB |
| Archive movies | `batch tencent archive --collection movies --count 1000` | ~100 GB |
| **总计** | | **~493 GB** |

### 数据量与模型效果

| 数据量 | 预估样本数 | 预期效果 |
|--------|-----------|----------|
| 25 GB | 50-100万 | 基线模型 |
| 50 GB | 100-300万 | **最佳性价比** |
| 200 GB | 500万+ | 小幅提升 |
| 500 GB | 1000万+ | 接近饱和 |

**建议**：先用 50GB 训练基线，看验证集准确率再决定是否扩展

---

## 目录结构

```
Filerestore_CLI/
├── scripts/
│   ├── storage.py               # 统一存储接口（腾讯云COS）
│   ├── download_govdocs.py      # Govdocs 下载器
│   ├── download_fma.py          # FMA 音频下载器
│   ├── download_pexels.py       # Pexels 视频下载器
│   ├── download_archive.py      # Internet Archive 下载器
│   ├── collect_training_data.py # 统一数据收集脚本
│   └── requirements.txt
│
├── ml/
│   ├── continuity/
│   │   ├── model_cnn.py         # 1D CNN 模型定义
│   │   ├── train_cnn.py         # 训练脚本
│   │   └── feature_extractor.py # Python 特征提取器（备用）
│   ├── notebooks/
│   │   └── train_continuity_cnn.ipynb  # Colab notebook
│   └── requirements.txt
│
├── document/
│   └── ML_CLOUD_TRAINING_WORKFLOW.md  # 本文档
│
└── Filerestore_CLI/
    └── src/fileRestore/
        ├── BlockContinuityDetector.cpp  # C++ 特征提取 & 推理
        ├── BlockContinuityDetector.h
        ├── DatasetGenerator.cpp         # 数据集生成
        └── DatasetGenerator.h
```

---

## 模型架构

### 1D CNN (ContinuityCNN1D)

```
Input: (batch, 64) 特征向量
  ↓ Reshape → (batch, 1, 64)
  ↓ Conv1d(1→32, k=5) + BN + ReLU + Dropout(0.3)
  ↓ Conv1d(32→64, k=3) + BN + ReLU + Dropout(0.3)
  ↓ Conv1d(64→128, k=3) + BN + ReLU + Dropout(0.3)
  ↓ Global Average Pooling → (batch, 128)
  ↓ FC(128→64) + ReLU + Dropout(0.3)
  ↓ FC(64→2)
Output: (batch, 2) logits [不连续, 连续]

参数量: ~40,000
模型大小: ~200 KB (ONNX)
```

---

## 64维特征说明

| 索引 | 类别 | 说明 |
|------|------|------|
| 0-15 | Block 1 | 块1统计：熵、均值、标准差、零字节比、高字节比、可打印比、8区间直方图、PK签名、压缩分数 |
| 16-31 | Block 2 | 块2统计（同上） |
| 32-47 | Boundary | 边界特征：熵差、熵梯度、均值差、分布相似度、边界平滑度、跨边界相关、转移直方图、本地头检测、EOCD接近度 |
| 48-63 | Format | 格式特定：ZIP(DEFLATE/结构)/MP3(帧同步/ID3)/MP4(Box/mdat) |

---

## 故障排除

### 问题：COS 连接失败

```bash
# 检查配置
python scripts/storage.py backends

# 重新配置
python scripts/storage.py config tencent

# 测试连接
python scripts/storage.py test tencent
```

### 问题：上传/下载速度慢

腾讯云 COS 内网访问免流量费，如果在腾讯云服务器上操作会更快。

本地上传建议：
- 使用有线网络
- 避开高峰时段
- 分批上传大文件

### 问题：mlscan 显示 "No samples collected"

**可能原因：**
- 文件太小（< 32KB）
- 文件签名不匹配
- 目录中没有支持的文件类型

**解决：**
```bash
# 检查目录内容
dir /s D:\temp\ml_training\*.mp3
dir /s D:\temp\ml_training\*.zip

# 确保下载了足够的文件
python scripts/storage.py download tencent ML/FMA/ D:\temp\ml_training\fma
```

### 问题：Colab 训练中断

- 模型每个 epoch 自动保存到 COS
- 可以从检查点恢复训练

### 问题：ONNX 模型加载失败

**检查：**
1. 模型文件路径：`models/continuity/continuity_cnn.onnx`
2. 元数据文件存在：`models/continuity/continuity_cnn.json`
3. ONNX Runtime 版本兼容

---

## storage.py 快速参考

```bash
# ========== 配置（推荐使用JSON文件）==========
python storage.py config-template -o cos_config.json -b tencent  # 生成模板
# 编辑 cos_config.json 填入凭据
python storage.py config-import cos_config.json                   # 导入配置

# ========== 配置管理 ==========
python storage.py config tencent      # 交互式配置腾讯云 COS
python storage.py config-show         # 查看当前配置（密钥脱敏）
python storage.py config-export backup.json                       # 导出配置
python storage.py config-export backup.json --mask-secrets        # 导出（隐藏密钥）
python storage.py backends            # 列出已配置的后端
python storage.py test tencent        # 测试连接

# ========== 查看 ==========
python storage.py list tencent ML/    # 列出文件
python storage.py size tencent ML/    # 统计大小

# ========== 上传 ==========
python storage.py upload tencent local.csv ML/datasets/data.csv     # 单文件
python storage.py upload tencent ./data/ ML/data/                   # 目录

# ========== 下载 ==========
python storage.py download tencent ML/datasets/data.csv ./local.csv # 单文件
python storage.py download tencent ML/models/ ./models/             # 目录

# ========== 删除 ==========
python storage.py delete tencent ML/old_file.csv

# ========== 分批上传（推荐）==========
# Govdocs: 自动分批下载→上传→删除
python storage.py batch tencent govdocs --start 1 --end 100 --batch-range 10

# FMA: 下载整个子集后上传
python storage.py batch tencent fma --subset medium

# Pexels: 分批下载视频
python storage.py batch tencent pexels --count 1000 --batch-count 100

# Archive: 下载音频/视频
python storage.py batch tencent archive --collection audio --count 500
```

### Python 代码使用

```python
from storage import get_storage

# 获取存储实例
storage = get_storage("tencent")

# 上传
storage.upload("local.csv", "ML/datasets/data.csv")

# 下载
storage.download("ML/models/model.onnx", "./model.onnx")

# 列出文件
for f in storage.list_files("ML/"):
    print(f.key, f.size)

# 上传整个目录
storage.upload_directory("./data", "ML/data/")

# 下载整个目录
storage.download_directory("ML/models/", "./models/")
```

---

## 数据源参考

| 数据源 | URL | 说明 |
|--------|-----|------|
| Govdocs | https://digitalcorpora.org/corpora/files | 100万办公文档 |
| FMA | https://github.com/mdeff/fma | 10万首音乐 |
| Pexels | https://www.pexels.com/api/ | 免费视频API |
| Archive | https://archive.org | 公共领域媒体 |
| 腾讯云COS | https://cloud.tencent.com/product/cos | 对象存储服务 |

---

## 版本历史

- **2026-01-10**: 更新为使用腾讯云 COS (500GB套餐)
- **2025-01-10**: 初始版本，支持 ZIP/MP3/MP4 等20+种格式
