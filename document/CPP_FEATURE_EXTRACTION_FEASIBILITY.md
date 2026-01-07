# C++ 特征提取与本地数据集训练方案可行性分析

## 1. 方案概述

### 当前架构
```
Python端: 数据收集 → 特征提取 → 模型训练 → ONNX导出
C++端:   ONNX加载 → 特征提取 → 推理
```

### 提议架构
```
C++端:   磁盘扫描(线程池) → 特征提取 → 特征数据集(CSV/二进制)
Python端: 加载特征 → 模型训练 → ONNX导出
C++端:   ONNX加载 → 特征提取 → 推理
```

## 2. 可行性评估

### 2.1 技术可行性: ✅ 高

#### 特征提取算法完全一致
C++ 端 (`MLClassifier.cpp`) 已实现与 Python 端 (`dataset.py`) 完全相同的特征提取算法：

| 特征 | Python实现 | C++实现 | 一致性 |
|------|-----------|---------|-------|
| 字节频率 (256维) | `np.bincount()` | 手动计数 | ✅ |
| Shannon熵 | `np.sum(p * log2(p))` | 手动计算 | ✅ |
| 均值/标准差 | `np.mean/std` | 手动计算 | ✅ |
| 唯一字节比例 | `len(np.unique())` | 手动计数 | ✅ |
| ASCII比例 | 范围判断 | 范围判断 | ✅ |

#### 线程池基础设施已就绪
`SignatureScanThreadPool` 已实现：
- 多线程扫描
- 任务队列
- 结果聚合
- 进度报告

### 2.2 性能优势: ✅ 显著

| 方面 | Python | C++ | 提升倍数 |
|------|--------|-----|---------|
| 文件I/O | 单线程顺序读 | 多线程并行读 | 8-16x |
| 特征提取 | 解释执行 | 编译优化 | 3-5x |
| 内存效率 | 高GC开销 | 精确控制 | 2-3x |
| 磁盘访问 | 跨进程调用 | 直接原始访问 | 5-10x |

**估算性能**：
- Python扫描20GB: ~2-4小时
- C++扫描20GB: ~10-20分钟

### 2.3 数据质量优势: ✅ 显著

#### 本地数据的优势
1. **类型覆盖全面**: 用户实际使用的所有文件类型
2. **真实分布**: 反映实际文件类型比例
3. **版本多样性**: 不同软件版本生成的同类型文件
4. **数据量充足**: 深度使用的机器可能有TB级数据

#### 与Govdocs对比
| 方面 | Govdocs | 本地数据 |
|------|---------|---------|
| 数据新鲜度 | 2009年 | 实时 |
| 类型覆盖 | 政府文档为主 | 用户实际使用 |
| 软件版本 | 旧版Office等 | 最新版本 |
| 数据量 | 需下载 | 就地可用 |

## 3. 实现方案

### 3.1 C++ 数据集生成器

```cpp
// FileFeatureExtractor.h
class DatasetGenerator {
public:
    struct SampleInfo {
        std::string filePath;
        std::string extension;      // 真实标签
        ML::FileFeatures features;  // 261维特征
    };

    // 扫描指定目录，提取特征
    void ScanDirectory(const std::wstring& path,
                       int maxSamplesPerType = 1000);

    // 导出为CSV格式（Python易读）
    bool ExportCSV(const std::string& outputPath);

    // 导出为二进制格式（高效）
    bool ExportBinary(const std::string& outputPath);

private:
    std::vector<SampleInfo> samples;
    std::unordered_map<std::string, int> typeCount;
};
```

### 3.2 扫描策略

```cpp
// 利用现有线程池架构
void DatasetGenerator::ScanVolume(char driveLetter) {
    // 1. 遍历MFT获取所有文件记录
    MFTReader reader;
    reader.OpenVolume(driveLetter);

    // 2. 按扩展名过滤目标类型
    std::set<std::string> targetTypes = {
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        "jpg", "png", "gif", "bmp", "mp3", "mp4", "zip", "exe"
    };

    // 3. 使用线程池并行提取特征
    ThreadPool pool(12);
    for (auto& record : mftRecords) {
        if (targetTypes.count(record.extension)) {
            pool.Submit([&]() {
                auto features = ExtractFeatures(record);
                AddSample(record.path, record.extension, features);
            });
        }
    }
}
```

### 3.3 Python 训练脚本适配

```python
# train_from_cpp_features.py
import numpy as np
import pandas as pd

def load_cpp_dataset(csv_path):
    """加载C++生成的特征数据集"""
    df = pd.read_csv(csv_path)

    # 前261列是特征，最后一列是标签
    features = df.iloc[:, :261].values.astype(np.float32)
    labels = df['extension'].values

    # 编码标签
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return features, labels, label_encoder.classes_

# 使用现有训练流程
features, labels, classes = load_cpp_dataset("local_dataset.csv")
train_loader, val_loader, norm_params = create_data_loaders(features, labels)
model = train(train_loader, val_loader)
```

## 4. 数据集文件格式

### 4.1 CSV格式（开发/调试用）
```csv
f0,f1,f2,...,f260,extension,file_path
0.012,0.003,...,0.156,pdf,C:\Documents\report.pdf
0.008,0.001,...,0.089,jpg,C:\Pictures\photo.jpg
```

### 4.2 二进制格式（生产用）
```
[Header: 16 bytes]
  - magic: 4 bytes ("MLFD")
  - version: 4 bytes
  - sample_count: 4 bytes
  - feature_dim: 4 bytes

[Sample Records]
  For each sample:
    - features: 261 * 4 = 1044 bytes (float32)
    - extension_len: 1 byte
    - extension: variable (max 16 bytes)
    - path_len: 2 bytes
    - path: variable (UTF-8)
```

## 5. 实现步骤

### Phase 1: 核心功能 (1-2天)
1. [ ] 创建 `DatasetGenerator` 类
2. [ ] 实现目录遍历和文件过滤
3. [ ] 集成现有特征提取代码
4. [ ] 实现CSV导出

### Phase 2: 性能优化 (1天)
1. [ ] 集成线程池并行提取
2. [ ] 实现二进制导出格式
3. [ ] 添加进度报告

### Phase 3: Python集成 (0.5天)
1. [ ] 修改训练脚本支持外部特征文件
2. [ ] 添加数据验证和清洗
3. [ ] 更新normalization参数保存

### Phase 4: CLI命令 (0.5天)
1. [ ] 添加 `mlscan` 命令
2. [ ] 支持参数：目录、类型、采样数
3. [ ] 输出统计信息

## 6. 预期输出示例

```
> mlscan D: --types=pdf,jpg,png,docx,xlsx --max=2000 --output=dataset.csv

=== ML Dataset Generation ===
Target directory: D:\
Target types: pdf, jpg, png, docx, xlsx
Max samples per type: 2000

Scanning MFT...
Found 156,234 candidate files

Extracting features...
[████████████████████████████████████████] 100%

=== Dataset Statistics ===
pdf:    2000 samples
jpg:    2000 samples
png:    1856 samples
docx:   2000 samples
xlsx:   1234 samples
------------------------
Total:  9090 samples
Feature dimension: 261

Output saved to: dataset.csv (12.3 MB)
Time elapsed: 3m 42s
```

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 特征计算精度差异 | 低 | 高 | 单元测试验证一致性 |
| 大文件内存溢出 | 中 | 中 | 限制读取大小(4KB) |
| 类型分布不均 | 高 | 中 | 设置每类上限 |
| 隐私数据泄露 | 低 | 高 | 仅存储特征不存储内容 |

## 8. 结论

### 可行性评级: ✅ 强烈推荐实施

**优势**:
1. 性能提升10-20倍
2. 数据质量显著提高
3. 代码复用率高（特征提取已实现）
4. 实现复杂度低

**ROI分析**:
- 开发时间: ~4天
- 每次训练节省: ~2-3小时
- 模型准确率预期提升: 5-15%

### 下一步行动
1. 确认是否开始实现
2. 确定优先支持的文件类型列表
3. 确定数据集大小目标

---

*文档版本: 1.0*
*日期: 2026-01-07*
