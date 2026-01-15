# ML 文件修复可行性分析

## 1. 问题定义

### 1.1 什么是"轻微覆盖损坏"

在文件恢复场景中，轻微覆盖损坏通常指：

| 损坏类型 | 描述 | 典型场景 |
|----------|------|----------|
| 头部覆盖 | 文件签名/元数据被覆盖，但主体数据完好 | MFT 记录重用后新文件写入少量数据 |
| 尾部截断 | 文件末尾部分簇被新数据覆盖 | 碎片化文件部分簇被重新分配 |
| 稀疏损坏 | 文件中间少量字节被覆盖 | 坏扇区、随机写入 |
| 结构损坏 | 文件内部索引/目录结构损坏 | ZIP/Office 文档的中央目录损坏 |

### 1.2 修复目标

```
损坏文件 + ML模型 → 可用文件（或部分可用）

目标不是完美还原，而是：
1. 恢复文件可打开/可读取状态
2. 最大化保留有效内容
3. 标记不确定区域
```

---

## 2. 技术可行性分析

### 2.1 按文件类型分类

#### 高可行性（结构化格式）

| 文件类型 | 修复策略 | 可行性 | 说明 |
|----------|----------|--------|------|
| **JPEG** | 重建 JFIF 头部、修复 Huffman 表 | ★★★★★ | 图像可容忍部分数据丢失，显示灰色块 |
| **PNG** | 重建 IHDR 块、跳过损坏 chunk | ★★★★☆ | CRC 校验可定位损坏区域 |
| **ZIP/DOCX/XLSX** | 重建中央目录、提取未损坏条目 | ★★★★☆ | 本地文件头可独立解析 |
| **PDF** | 重建 xref 表、提取页面流 | ★★★☆☆ | 交叉引用表可重建 |
| **MP4/MOV** | 重建 moov atom、定位 mdat | ★★★☆☆ | 视频帧可独立解码 |

#### 中等可行性

| 文件类型 | 修复策略 | 可行性 | 说明 |
|----------|----------|--------|------|
| **MP3** | 重建 ID3 标签、同步帧边界 | ★★★☆☆ | 音频帧有同步字 |
| **SQLite** | 重建 B-tree 页、提取记录 | ★★★☆☆ | 页面结构固定 |
| **EXE/DLL** | 重建 PE 头部 | ★★☆☆☆ | 代码段难以验证正确性 |

#### 低可行性

| 文件类型 | 原因 |
|----------|------|
| **加密文件** | 任何字节损坏都导致解密失败 |
| **压缩流** | DEFLATE 流损坏会导致后续全部无法解压 |
| **纯文本** | 无结构可利用，难以区分损坏和正常内容 |

### 2.2 ML 可以解决的问题

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML 修复能力矩阵                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ ML 擅长                      ❌ ML 不擅长                    │
│  ─────────────                  ─────────────                   │
│  • 模式识别与补全                • 精确字节恢复                   │
│  • 结构重建                      • 加密数据恢复                   │
│  • 相似内容推断                  • 完全覆盖的数据                 │
│  • 图像/音频插值                 • 逻辑一致性保证                 │
│  • 异常检测定位损坏              • 跨文件依赖修复                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 具体修复方案

### 3.1 方案一：文件头重建（高优先级）

**目标**: 修复被覆盖的文件签名和元数据头部

**原理**:
```
损坏的文件: [覆盖数据][原始数据主体............]
                ↓ ML 预测
修复的文件: [重建头部][原始数据主体............]
```

**实现方式**:

```cpp
// 训练数据：大量正常文件头部样本
// 输入特征：文件主体数据的统计特征
// 输出：预测的文件头部

struct HeaderRepairModel {
    // 输入: 文件前 N 个簇的数据（跳过损坏头部）
    // 输出: 预测的文件类型 + 重建的头部

    FileType PredictType(const BYTE* data, size_t offset, size_t size);
    vector<BYTE> RebuildHeader(FileType type, const BYTE* bodyData);
};
```

**训练数据需求**:
- 每种文件类型需要 10,000+ 样本
- 需要包含各种变体（不同版本、不同创建工具）

**预期效果**:
| 文件类型 | 头部重建成功率 | 文件可用率 |
|----------|----------------|------------|
| JPEG | 95%+ | 90%+ |
| PNG | 90%+ | 85%+ |
| ZIP/Office | 85%+ | 70%+ |
| PDF | 80%+ | 60%+ |

### 3.2 方案二：结构修复（中优先级）

**目标**: 修复文件内部结构（索引、目录、引用表）

**适用场景**:
- ZIP 中央目录损坏
- PDF xref 表损坏
- SQLite B-tree 损坏

**实现方式**:

```cpp
// ZIP 中央目录重建
class ZipRepairModel {
public:
    // 扫描所有本地文件头，重建中央目录
    bool RepairCentralDirectory(vector<BYTE>& zipData) {
        vector<LocalFileHeader> localHeaders = ScanLocalHeaders(zipData);

        // ML 预测缺失的元数据
        for (auto& header : localHeaders) {
            if (header.isIncomplete) {
                header.compressedSize = PredictCompressedSize(header);
                header.crc32 = PredictCRC32(header);  // 或设为 0 跳过校验
            }
        }

        // 重建中央目录
        return BuildCentralDirectory(zipData, localHeaders);
    }
};
```

### 3.3 方案三：内容插值（低优先级）

**目标**: 对图像/音频的损坏区域进行内容填充

**仅适用于**: 可容忍数据丢失的媒体文件

**实现方式**:

```
损坏的 JPEG:
┌──────────────────────────────────────┐
│ [正常区域] [损坏区域] [正常区域]      │
│     ↓          ↓          ↓          │
│   解码OK    灰色块     解码OK        │
└──────────────────────────────────────┘
                 ↓ ML 图像修复
┌──────────────────────────────────────┐
│ [正常区域] [AI填充]  [正常区域]      │
│   完整的可视图像（有AI生成内容）      │
└──────────────────────────────────────┘
```

**技术选型**:
- 图像修复: Stable Diffusion Inpainting, LaMa
- 音频修复: Audio Super Resolution 模型

**注意**: 这种修复会引入 AI 生成内容，需要明确标记

---

## 4. 实现架构

### 4.1 集成到当前项目

```
┌─────────────────────────────────────────────────────────────────┐
│                    Filerestore_CLI                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  USN 定点恢复                                                    │
│       ↓                                                         │
│  签名验证失败 ──→ ML 修复模块                                    │
│       │              │                                          │
│       │              ├─→ 文件头重建                              │
│       │              ├─→ 结构修复                                │
│       │              └─→ 内容插值（可选）                        │
│       │              │                                          │
│       ↓              ↓                                          │
│  恢复结果 ←──────────┘                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 模块设计

```cpp
// MLFileRepair.h

enum class RepairType {
    HEADER_REBUILD,      // 头部重建
    STRUCTURE_REPAIR,    // 结构修复
    CONTENT_INTERPOLATE  // 内容插值
};

enum class RepairResult {
    SUCCESS,             // 完全修复
    PARTIAL,             // 部分修复
    FAILED,              // 无法修复
    NOT_APPLICABLE       // 不适用（文件类型不支持）
};

struct RepairReport {
    RepairResult result;
    double confidence;           // 修复置信度
    vector<string> repairActions;// 执行的修复操作
    size_t bytesModified;        // 修改的字节数
    vector<pair<size_t, size_t>> modifiedRanges; // 修改区域
};

class MLFileRepair {
public:
    // 分析文件损坏情况
    DamageAnalysis AnalyzeDamage(const vector<BYTE>& data,
                                  const string& expectedType);

    // 尝试修复
    RepairReport TryRepair(vector<BYTE>& data,
                           const string& expectedType,
                           RepairType repairType = RepairType::HEADER_REBUILD);

    // 批量修复
    vector<RepairReport> BatchRepair(const vector<CarvedFile>& files);

private:
    unique_ptr<Ort::Session> headerModel;
    unique_ptr<Ort::Session> structureModel;

    // 各文件类型的专用修复器
    map<string, unique_ptr<FileTypeRepairer>> repairers;
};
```

### 4.3 与现有 ML 分类器集成

```cpp
// 在 FileCarver 中集成修复流程
void FileCarver::ProcessCarvedFile(CarvedFile& file) {
    // 1. 签名验证
    double confidence;
    string detectedType = mlClassifier->Classify(file.data, confidence);

    if (confidence < 0.5 || detectedType != file.expectedType) {
        // 2. 尝试 ML 修复
        MLFileRepair repairer;
        auto damage = repairer.AnalyzeDamage(file.data, file.expectedType);

        if (damage.isRepairable) {
            auto report = repairer.TryRepair(file.data, file.expectedType);
            file.repaired = (report.result == RepairResult::SUCCESS ||
                            report.result == RepairResult::PARTIAL);
            file.repairConfidence = report.confidence;
        }
    }
}
```

---

## 5. 训练数据与模型

### 5.1 头部重建模型

**数据准备**:
```python
# 训练数据生成
def generate_training_data(files_dir, output_dir):
    for file_path in glob(files_dir + "/*"):
        data = read_file(file_path)
        file_type = detect_type(data)

        # 提取特征
        header = data[:HEADER_SIZE]
        body_features = extract_body_features(data[HEADER_SIZE:])

        # 保存训练样本
        save_sample(output_dir, {
            'file_type': file_type,
            'header': header,
            'body_features': body_features
        })
```

**模型架构**:
```
输入层 (body_features: 256维)
    ↓
Dense(512) + ReLU + Dropout(0.3)
    ↓
Dense(256) + ReLU + Dropout(0.3)
    ↓
输出层:
  - 分类头: Softmax(num_file_types)
  - 生成头: Dense(header_size) → 重建的头部字节
```

### 5.2 所需模型清单

| 模型 | 用途 | 输入 | 输出 | 大小估计 |
|------|------|------|------|----------|
| HeaderClassifier | 从主体数据推断文件类型 | 256维特征 | 文件类型概率 | ~5MB |
| JPEGHeaderGen | JPEG 头部重建 | 图像 DCT 特征 | JFIF 头部 | ~10MB |
| ZIPStructureRepair | ZIP 结构修复 | 本地头部列表 | 中央目录 | ~8MB |
| PDFXrefRepair | PDF xref 重建 | 对象偏移列表 | xref 表 | ~5MB |

### 5.3 训练资源估计

| 项目 | 估计值 |
|------|--------|
| 训练样本 | 每类型 50,000+ 文件 |
| 训练时间 | 4-8 小时 (GPU) |
| 模型总大小 | 30-50MB |
| 推理时间 | <100ms/文件 |

---

## 6. 风险与限制

### 6.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 过拟合 | 仅对训练集类似的文件有效 | 增加数据多样性，正则化 |
| 误修复 | 将正常内容"修复"成错误内容 | 严格的损坏检测前置条件 |
| 版本差异 | 新版本文件格式不兼容 | 持续更新训练数据 |

### 6.2 法律/伦理风险

| 风险 | 说明 | 建议 |
|------|------|------|
| 数据完整性 | 修复后的文件不等于原始文件 | 明确标记为"ML修复版本" |
| 证据效力 | 司法场景可能不接受 ML 修复的文件 | 同时保留原始损坏版本 |
| AI 生成内容 | 内容插值会引入不存在的内容 | 仅用于预览，不用于取证 |

### 6.3 不适用场景

```
❌ 以下场景不建议使用 ML 修复：

1. 完全覆盖 - 原始数据完全丢失，无法恢复
2. 加密文件 - 任何修改都会导致解密失败
3. 校验敏感 - 带完整性校验的文件（签名的可执行文件）
4. 司法取证 - 需要保持数据原始性
5. 压缩流 - DEFLATE 等流式压缩损坏会扩散
```

---

## 7. 实施路线图

### Phase 1: 基础头部修复（建议首先实现）

**目标**: JPEG/PNG 头部重建

**工作量**: 2-3 周

**步骤**:
1. 收集 JPEG/PNG 训练样本
2. 实现头部特征提取
3. 训练分类 + 生成模型
4. 集成到 FileCarver

### Phase 2: 结构化文件修复

**目标**: ZIP/Office/PDF 结构修复

**工作量**: 3-4 周

**步骤**:
1. 实现各格式的结构解析器
2. 训练结构预测模型
3. 实现重建算法

### Phase 3: 媒体内容修复（可选）

**目标**: 图像/视频损坏区域填充

**工作量**: 4-6 周

**依赖**: 需要大型生成模型（Stable Diffusion 等）

---

## 8. 结论与建议

### 8.1 可行性总结

| 方面 | 评估 |
|------|------|
| 技术可行性 | ★★★★☆ - 头部重建和结构修复技术成熟 |
| 实现难度 | ★★★☆☆ - 需要针对每种文件格式单独实现 |
| 实用价值 | ★★★★☆ - 可显著提高轻微损坏文件的恢复率 |
| 资源需求 | ★★★☆☆ - 模型较小，推理快速 |

### 8.2 建议实施优先级

```
高优先级（立即可行）:
├─ JPEG/PNG 头部重建 - 最常见，效果好
├─ ZIP 中央目录重建 - 可恢复 Office 文档
└─ 损坏检测与定位 - 所有修复的前提

中优先级（价值高但复杂）:
├─ PDF xref 重建
├─ MP4 moov atom 重建
└─ SQLite 页面修复

低优先级（需要大型模型）:
└─ 图像/视频内容插值 - 资源需求高
```

### 8.3 最终建议

**推荐首先实现 Phase 1（JPEG/PNG 头部修复）**：

1. 技术风险低，效果可预期
2. 图像是最常见的恢复目标
3. 可快速验证 ML 修复方案的可行性
4. 模型小（~10MB），易于集成

实现后可根据效果决定是否继续扩展到其他文件类型。

---

## 附录 A: 参考资源

### 开源项目
- [JPEG-Repair](https://github.com/example/jpeg-repair) - JPEG 修复工具
- [zip-recovery](https://github.com/example/zip-recovery) - ZIP 结构恢复
- [LaMa](https://github.com/saic-mdal/lama) - 图像修复模型

### 论文
- "Deep Learning for File Fragment Classification" (2019)
- "Neural Network Based File Carving" (2020)
- "Automated Digital Forensics with Machine Learning" (2021)

### 文件格式规范
- JPEG: ITU-T T.81
- PNG: RFC 2083
- ZIP: APPNOTE.TXT
- PDF: ISO 32000-2:2020
