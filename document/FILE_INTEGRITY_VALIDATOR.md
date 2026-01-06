# 文件完整性验证系统 (File Integrity Validator)

## 概述

文件完整性验证系统是 Filerestore_CLI 的核心模块之一，用于检测通过签名搜索（File Carving）恢复的文件是否完整或已损坏。该系统采用多维度分析方法，结合熵值分析、结构验证、统计分析和文件尾检测，提供可靠的完整性评估。

### 为什么需要完整性验证？

签名搜索恢复文件时可能遇到以下问题：

| 问题 | 原因 | 影响 |
|------|------|------|
| **文件尾未检测** | 部分格式没有明确的文件尾签名 | 文件大小估算不准确 |
| **文件碎片化** | NTFS文件可能分散在非连续簇 | 中间数据丢失 |
| **部分覆盖** | 原文件被新数据部分覆盖 | 文件头完好但内容损坏 |
| **误报匹配** | 数据巧合匹配文件签名 | 完全无效的"文件" |
| **嵌入文件** | 文档内嵌的缩略图/资源 | 恢复出碎片化的嵌入内容 |

---

## 技术原理

### 1. 熵值分析 (Entropy Analysis)

**原理**: 不同类型的数据具有不同的信息熵特征。

```
熵值范围 (0-8 bits/byte):
├── 0-2: 高度重复数据（全零、简单模式）
├── 2-5: 文本、未压缩数据
├── 5-7: 压缩数据、可执行代码
└── 7-8: 加密数据、随机数据
```

**香农熵计算公式**:
```
H(X) = -Σ p(x) * log₂(p(x))
```

**各文件类型期望熵值**:

| 文件类型 | 期望熵值 | 异常指标 |
|---------|---------|---------|
| JPEG | 7.0-7.95 | < 7.0 或 > 7.95 |
| PNG | 6.5-7.9 | < 6.5 或 > 7.9 |
| PDF | 4.0-7.8 | < 4.0 (纯文本) |
| ZIP | 7.2-7.98 | < 7.2 |
| 7z | 7.5-7.99 | < 7.5 |
| MP3 | 6.5-7.9 | < 6.5 |
| EXE | 5.0-7.5 | < 5.0 或 > 7.5 |
| BMP | 2.0-6.0 | > 6.0 |

**熵值突变检测**:

系统将文件分成 4KB 块，计算每块的熵值。如果相邻块熵值差异超过阈值（默认1.5），则表明可能存在数据损坏边界。

```
正常文件熵值分布:
Block 1: 7.5  Block 2: 7.4  Block 3: 7.6  Block 4: 7.5  ✓ 稳定

损坏文件熵值分布:
Block 1: 7.5  Block 2: 7.4  Block 3: 2.1  Block 4: 0.5  ✗ 突变
                            ↑
                        损坏边界
```

---

### 2. 结构验证 (Structure Validation)

针对不同文件格式进行特定的结构检查。

#### JPEG 验证

```
JPEG 必要结构:
├── SOI (0xFFD8) - Start of Image
├── APP0/APP1 - 应用段
├── DQT (0xFFDB) - 量化表
├── DHT (0xFFC4) - 霍夫曼表
├── SOF (0xFFC0-0xCF) - Start of Frame
├── SOS (0xFFDA) - Start of Scan
└── EOI (0xFFD9) - End of Image  ← 关键！
```

验证项:
- SOI 标记存在 (+20%)
- EOI 标记存在 (+30%)
- DHT 表有效 (+10%)
- DQT 表有效 (+10%)
- SOF 段存在 (+15%)
- SOS 段存在 (+15%)

#### PNG 验证

```
PNG 必要结构:
├── Signature (89 50 4E 47 0D 0A 1A 0A)
├── IHDR chunk - 图像头
├── IDAT chunk(s) - 图像数据
└── IEND chunk - 结束标记
```

验证项:
- PNG 签名有效 (+30%)
- IHDR 块存在 (+20%)
- IEND 块存在 (+30%)
- CRC 校验通过 (+20%)

#### ZIP 验证

```
ZIP 必要结构:
├── Local File Header (PK 03 04)
├── File Data
├── Central Directory (PK 01 02)
└── End of Central Directory (PK 05 06)
```

验证项:
- 有效本地文件头 (+30%)
- 中央目录存在 (+20%)
- EOCD 存在 (+30%)
- 文件计数匹配 (+20%)

#### PDF 验证

```
PDF 必要结构:
├── Header (%PDF-x.x)
├── Body (objects)
├── Cross-reference Table (xref)
├── Trailer
└── EOF marker (%%EOF)
```

---

### 3. 统计分析 (Statistical Analysis)

#### 零字节比例检测

```cpp
double zeroRatio = zeroCount / totalBytes;
```

期望值:
| 文件类型 | 最大零字节比例 |
|---------|---------------|
| JPEG/PNG/ZIP | < 5% |
| EXE/DLL | < 15% |
| PDF | < 10% |
| 其他 | < 20% |

高零字节比例可能表示:
- 数据被清零覆盖
- SSD TRIM 操作
- 文件碎片化

#### 卡方检验 (Chi-Square Test)

用于检测数据是否接近随机分布：

```cpp
χ² = Σ (observed[i] - expected)² / expected
```

- 自由度 = 255
- p=0.05 临界值 ≈ 293
- χ² < 350 表示接近均匀分布（可能是压缩/加密数据）

---

### 4. 综合评分模型

```
总分 = 熵值得分 × 25% + 结构得分 × 35% + 统计得分 × 20% + 尾部得分 × 20%
```

**评分解释**:

| 分数范围 | 状态 | 说明 |
|---------|------|------|
| ≥ 80% | [OK] | 高置信度，文件很可能完好 |
| 60-80% | [WARN] | 中置信度，可能有轻微问题 |
| 50-60% | [LOW] | 低置信度，可能已损坏 |
| < 50% | [FAIL] | 极低置信度，很可能已损坏 |

---

## API 参考

### FileIntegrityValidator 类

静态类，提供文件完整性验证功能。

#### 主要方法

```cpp
// 完整验证，返回详细评分
static FileIntegrityScore Validate(
    const BYTE* data,           // 文件数据
    size_t size,                // 数据大小
    const string& extension     // 文件扩展名
);

// 快速检查是否损坏
static bool IsLikelyCorrupted(
    const BYTE* data,
    size_t size,
    const string& extension
);

// 单独获取各项评分
static double GetEntropyScore(const BYTE* data, size_t size, const string& extension);
static double GetStructureScore(const BYTE* data, size_t size, const string& extension);
static double GetStatisticalScore(const BYTE* data, size_t size, const string& extension);
```

#### FileIntegrityScore 结构

```cpp
struct FileIntegrityScore {
    // 各项评分 (0.0-1.0)
    double entropyScore;        // 熵值分析得分
    double structureScore;      // 结构验证得分
    double statisticalScore;    // 统计分析得分
    double footerScore;         // 文件尾验证得分
    double overallScore;        // 综合得分

    // 诊断信息
    string diagnosis;           // 可读诊断
    bool isLikelyCorrupted;     // 是否可能损坏

    // 详细数据
    double entropy;             // 原始熵值 (0-8)
    double zeroRatio;           // 零字节比例
    double chiSquare;           // 卡方值
    bool hasValidHeader;        // 头部有效
    bool hasValidFooter;        // 尾部有效
    bool hasEntropyAnomaly;     // 检测到熵值异常
    size_t anomalyOffset;       // 异常位置（块编号）
};
```

#### 常量阈值

```cpp
static constexpr double MIN_INTEGRITY_SCORE = 0.5;      // 最低可接受分数
static constexpr double HIGH_CONFIDENCE_SCORE = 0.8;    // 高置信度阈值
static constexpr double ENTROPY_VARIANCE_THRESHOLD = 1.5; // 熵值突变阈值
static constexpr double MAX_ZERO_RATIO_COMPRESSED = 0.05; // 压缩文件最大零字节比例
static constexpr double MAX_ZERO_RATIO_GENERAL = 0.15;    // 一般文件最大零字节比例
```

### FileCarver 集成

```cpp
class FileCarver {
public:
    // 启用/禁用完整性验证
    void SetIntegrityValidation(bool enabled);
    bool IsIntegrityValidationEnabled() const;

    // 验证单个文件
    FileIntegrityScore ValidateFileIntegrity(const CarvedFileInfo& info);

    // 批量验证
    void ValidateIntegrityForResults(vector<CarvedFileInfo>& results, bool showProgress = true);

    // 过滤损坏文件
    vector<CarvedFileInfo> FilterCorruptedFiles(
        const vector<CarvedFileInfo>& results,
        double minIntegrityScore = 0.5
    );
};
```

### CarvedFileInfo 扩展字段

```cpp
struct CarvedFileInfo {
    // ... 原有字段 ...

    // 完整性验证信息
    double integrityScore;      // 完整性评分 (0-1)
    bool integrityValidated;    // 是否已验证
    string integrityDiagnosis;  // 完整性诊断信息
};
```

---

## CLI 命令

### carvevalidate - 批量验证

验证所有 carved 文件的完整性。

```bash
# 基本用法
carvevalidate

# 验证并过滤损坏文件（默认阈值 0.5）
carvevalidate filter

# 使用自定义阈值过滤
carvevalidate filter 0.7
```

**输出示例**:
```
=== File Integrity Validation ===
Files to validate: 150
Minimum score threshold: 0.50

--- Validating File Integrity ---
Files to validate: 150
Validating: 100.0% | Processed: 150/150 | Corrupted: 23

--- Integrity Validation Complete ---
Time: 3.2 seconds
Files validated: 150
High confidence (>= 80%): 89 (59.3%)
Low confidence (< 50%): 23 (15.3%)
Likely corrupted: 23 (15.3%)

=== Validation Results ===
------------------------------------------------------------------------------------------
[  0] [OK]    jpg | 87% | High confidence - likely intact
[  1] [OK]    png | 92% | High confidence - likely intact
[  2] [WARN]  zip | 65% | Medium confidence - may have minor issues
[  3] [FAIL]  jpg | 32% | Very low confidence - probably corrupted [Missing footer]
...
```

### carveintegrity - 详细分析

对单个文件进行详细的完整性分析。

```bash
carveintegrity <index>

# 示例
carveintegrity 3
```

**输出示例**:
```
=== Detailed Integrity Analysis ===
File index: 3
Type: JPEG Image (.jpg)
Size: 245760 bytes
LCN: 1234567

--- Entropy Analysis ---
  Raw entropy: 7.234 bits/byte
  Entropy score: 85.0%
  Anomaly detected: YES
  Anomaly offset: block #45

--- Structure Validation ---
  Structure score: 45.0%
  Valid header: Yes
  Valid footer: NO

--- Statistical Analysis ---
  Zero ratio: 2.34%
  Chi-square: 287.5
  Statistical score: 90.0%

--- Footer Validation ---
  Footer score: 30.0%

=== Overall Assessment ===
Overall score: 32.0%
Diagnosis: Very low confidence - probably corrupted [Missing footer] [Entropy anomaly detected]
Likely corrupted: YES
```

### carvelist - 查看列表（含完整性信息）

执行 `carvevalidate` 后，`carvelist` 会显示完整性信息：

```
[0] JPEG Image (.jpg)
    LCN: 1234567 | Offset: 0
    Size: 245760 bytes (240 KB)
    Confidence: 85%
    Timestamp Source: Embedded
    Modified: 2024-01-15 14:30:22
    Integrity: [OK] 87% - High confidence - likely intact
```

---

## 使用流程

### 推荐工作流程

```
1. 执行签名搜索
   carvepool C jpg,png,zip D:\recovered\

2. 验证完整性
   carvevalidate

3. 过滤损坏文件（可选）
   carvevalidate filter 0.6

4. 查看详细信息
   carvelist

5. 分析可疑文件
   carveintegrity 3

6. 恢复确认完好的文件
   carverecover 0 D:\recovered\image.jpg
```

### 性能考虑

| 操作 | 时间复杂度 | 建议 |
|-----|-----------|------|
| 熵值计算 | O(n) | 最大读取 1MB |
| 结构验证 | O(n) | 只读取必要部分 |
| 卡方检验 | O(n) | 采样前 64KB |
| 批量验证 | O(n×m) | 每文件最大读取 2MB |

---

## 局限性

1. **无法检测语义损坏** - 图片可能打开但内容错位
2. **碎片化文件** - 无法重组非连续数据
3. **加密文件** - 与随机数据难以区分
4. **新格式支持** - 需要为每种格式编写验证逻辑

---

## 代码示例

### 直接使用验证器

```cpp
#include "FileIntegrityValidator.h"

// 读取文件数据
vector<BYTE> fileData = ReadFile("test.jpg");

// 执行验证
FileIntegrityScore score = FileIntegrityValidator::Validate(
    fileData.data(),
    fileData.size(),
    "jpg"
);

// 检查结果
if (score.isLikelyCorrupted) {
    cout << "File may be corrupted: " << score.diagnosis << endl;
} else {
    cout << "File appears intact (score: " << score.overallScore * 100 << "%)" << endl;
}
```

### 与 FileCarver 集成使用

```cpp
#include "FileCarver.h"

MFTReader reader;
reader.OpenVolume('C');

FileCarver carver(&reader);

// 扫描文件
auto results = carver.ScanForFileTypesThreadPool({"jpg", "png"}, CARVE_SMART, 500);

// 验证完整性
carver.ValidateIntegrityForResults(results, true);

// 过滤损坏文件
auto goodFiles = carver.FilterCorruptedFiles(results, 0.6);

// 只恢复通过验证的文件
for (const auto& file : goodFiles) {
    if (file.integrityScore >= 0.8) {
        carver.RecoverCarvedFile(file, "D:\\recovered\\" + GetFilename(file));
    }
}
```

---

## 参考文献

1. Shannon, C.E. "A Mathematical Theory of Communication" (1948)
2. Garfinkel, S. "Carving contiguous and fragmented files with fast object validation" (2007)
3. NIST SP 800-86 "Guide to Integrating Forensic Techniques into Incident Response"
4. JPEG File Interchange Format (JFIF) Specification
5. PNG (Portable Network Graphics) Specification, Version 1.2
6. ZIP File Format Specification, PKWARE Inc.

---

## 版本历史

| 版本 | 日期 | 更改 |
|-----|------|------|
| 1.0 | 2025-01 | 初始实现：熵值分析、结构验证、统计分析 |

---

## 相关文档

- [签名搜索优化 (SIGNATURE_SCAN_OPTIMIZATION.md)](./SIGNATURE_SCAN_OPTIMIZATION.md)
- [线程池实现 (THREADPOOL_IMPLEMENTATION.md)](./THREADPOOL_IMPLEMENTATION.md)
- [时间戳提取](./TIMESTAMP_EXTRACTION.md)
