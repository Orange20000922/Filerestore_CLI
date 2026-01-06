# 签名恢复文件完整性验证

## 问题分析

### 为什么会恢复出损坏的文件？

| 原因 | 说明 | 影响 |
|------|------|------|
| **文件尾未检测** | 部分格式没有明确的文件尾签名 | 文件大小估算不准确 |
| **文件碎片化** | NTFS文件可能分散在非连续簇 | 中间数据丢失 |
| **部分覆盖** | 原文件被新数据部分覆盖 | 文件头完好但内容损坏 |
| **误报匹配** | 数据巧合匹配文件签名 | 完全无效的"文件" |
| **嵌入文件** | 文档内嵌的缩略图/资源 | 恢复出碎片化的嵌入内容 |

### 文件碎片化示意

```
正常文件（连续）:
簇: [100][101][102][103][104]  → 完整恢复 ✓

碎片化文件:
簇: [100][101]...[500][501][502]  → 签名扫描只能恢复 [100][101] 部分 ✗

部分覆盖:
簇: [100][NEW][NEW][103][104]  → 中间被新数据覆盖 ✗
```

---

## 检测方法

### 1. 熵值分析 (Entropy Analysis)

**原理**: 不同类型的数据有不同的信息熵特征

```
熵值范围 (0-8 bits/byte):
├── 0-2: 高度重复数据（全零、简单模式）
├── 2-5: 文本、未压缩数据
├── 5-7: 压缩数据、可执行代码
└── 7-8: 加密数据、随机数据
```

**检测逻辑**:

```cpp
// 计算香农熵
double CalculateEntropy(const BYTE* data, size_t size) {
    int frequency[256] = {0};
    for (size_t i = 0; i < size; i++) {
        frequency[data[i]]++;
    }

    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            double p = (double)frequency[i] / size;
            entropy -= p * log2(p);
        }
    }
    return entropy;  // 0-8
}
```

**文件类型熵值特征**:

| 文件类型 | 期望熵值 | 异常指标 |
|---------|---------|---------|
| JPEG | 7.5-7.9 | < 6.5 或 > 7.95 |
| PNG | 7.0-7.8 | < 6.0 或 > 7.95 |
| PDF | 4.0-7.5 | < 3.0 (纯文本) |
| ZIP | 7.5-7.95 | < 7.0 |
| MP3 | 7.0-7.8 | < 6.0 |
| EXE | 5.5-7.0 | < 4.0 或 > 7.5 |

**熵值突变检测**:

```cpp
// 分块熵值分析 - 检测数据损坏边界
vector<double> CalculateEntropyBlocks(const BYTE* data, size_t size, size_t blockSize) {
    vector<double> entropies;
    for (size_t i = 0; i < size; i += blockSize) {
        size_t len = min(blockSize, size - i);
        entropies.push_back(CalculateEntropy(data + i, len));
    }
    return entropies;
}

// 检测熵值突变点（可能的损坏边界）
bool DetectEntropyAnomaly(const vector<double>& entropies, double threshold = 1.5) {
    if (entropies.size() < 2) return false;

    for (size_t i = 1; i < entropies.size(); i++) {
        double diff = abs(entropies[i] - entropies[i-1]);
        if (diff > threshold) {
            return true;  // 发现突变，可能损坏
        }
    }
    return false;
}
```

---

### 2. 文件结构验证 (Structure Validation)

#### JPEG 验证

```cpp
struct JPEGValidation {
    bool hasSOI;           // Start of Image (FFD8)
    bool hasEOI;           // End of Image (FFD9)
    bool hasValidMarkers;  // 有效的段标记
    bool hasValidDHT;      // 霍夫曼表
    bool hasValidDQT;      // 量化表
    bool hasSOF;           // Start of Frame
    bool hasSOS;           // Start of Scan
    int markerCount;       // 标记数量
    double confidence;     // 整体置信度
};

JPEGValidation ValidateJPEG(const BYTE* data, size_t size) {
    JPEGValidation result = {0};

    // 检查 SOI
    if (size < 2 || data[0] != 0xFF || data[1] != 0xD8) {
        return result;
    }
    result.hasSOI = true;

    // 遍历标记
    size_t pos = 2;
    while (pos + 2 < size) {
        if (data[pos] != 0xFF) {
            pos++;
            continue;
        }

        BYTE marker = data[pos + 1];

        // EOI
        if (marker == 0xD9) {
            result.hasEOI = true;
            result.hasValidMarkers = true;
            break;
        }

        // 检查关键标记
        if (marker == 0xC4) result.hasValidDHT = true;
        if (marker == 0xDB) result.hasValidDQT = true;
        if (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
            result.hasSOF = true;
        }
        if (marker == 0xDA) result.hasSOS = true;

        result.markerCount++;

        // 跳过段
        if (marker >= 0xD0 && marker <= 0xD9) {
            pos += 2;
        } else if (pos + 4 < size) {
            WORD segLen = (data[pos + 2] << 8) | data[pos + 3];
            pos += 2 + segLen;
        } else {
            break;
        }
    }

    // 计算置信度
    result.confidence = 0.0;
    if (result.hasSOI) result.confidence += 0.2;
    if (result.hasEOI) result.confidence += 0.3;
    if (result.hasValidDHT) result.confidence += 0.1;
    if (result.hasValidDQT) result.confidence += 0.1;
    if (result.hasSOF) result.confidence += 0.15;
    if (result.hasSOS) result.confidence += 0.15;

    return result;
}
```

#### PNG 验证 (CRC 校验)

```cpp
// PNG 使用 CRC32 校验每个 chunk
bool ValidatePNGChunk(const BYTE* chunkData, size_t chunkLen) {
    if (chunkLen < 12) return false;

    // chunk 结构: [4字节长度][4字节类型][数据][4字节CRC]
    DWORD declaredLen = ReadBigEndian32(chunkData);
    const BYTE* typeAndData = chunkData + 4;
    size_t dataLen = declaredLen + 4;  // type + data

    if (8 + declaredLen + 4 > chunkLen) return false;

    // 计算 CRC
    DWORD calculatedCRC = CalculateCRC32(typeAndData, dataLen);
    DWORD storedCRC = ReadBigEndian32(chunkData + 8 + declaredLen);

    return calculatedCRC == storedCRC;
}
```

#### ZIP 验证

```cpp
struct ZIPValidation {
    bool hasValidLocalHeader;
    bool hasValidCentralDir;
    bool hasEndOfCentralDir;
    DWORD declaredFileCount;
    DWORD actualFileCount;
    double confidence;
};

ZIPValidation ValidateZIP(const BYTE* data, size_t size) {
    ZIPValidation result = {0};

    // 检查 Local File Header (PK..)
    if (size < 30 || data[0] != 0x50 || data[1] != 0x4B ||
        data[2] != 0x03 || data[3] != 0x04) {
        return result;
    }
    result.hasValidLocalHeader = true;

    // 搜索 End of Central Directory (PK..)
    for (size_t i = size - 22; i > 0 && i > size - 65536; i--) {
        if (data[i] == 0x50 && data[i+1] == 0x4B &&
            data[i+2] == 0x05 && data[i+3] == 0x06) {
            result.hasEndOfCentralDir = true;
            result.declaredFileCount = *(WORD*)(data + i + 8);
            break;
        }
    }

    // 计算置信度
    result.confidence = 0.3;  // 有效头部
    if (result.hasEndOfCentralDir) result.confidence += 0.5;

    return result;
}
```

---

### 3. 统计分析 (Statistical Analysis)

#### 字节频率分布

```cpp
// 计算卡方值 - 检测数据随机性
double CalculateChiSquare(const BYTE* data, size_t size) {
    int observed[256] = {0};
    double expected = size / 256.0;

    for (size_t i = 0; i < size; i++) {
        observed[data[i]]++;
    }

    double chiSquare = 0.0;
    for (int i = 0; i < 256; i++) {
        double diff = observed[i] - expected;
        chiSquare += (diff * diff) / expected;
    }

    return chiSquare;
}

// 判断数据是否像随机/加密数据
bool IsLikelyRandom(const BYTE* data, size_t size) {
    double chi = CalculateChiSquare(data, size);
    // 自由度=255，p=0.05 的临界值约为 293
    return chi < 350;  // 卡方值低 = 更接近均匀分布 = 可能是随机数据
}
```

#### 零字节比例检测

```cpp
// 检测异常的零字节分布
double CalculateZeroRatio(const BYTE* data, size_t size) {
    size_t zeroCount = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] == 0) zeroCount++;
    }
    return (double)zeroCount / size;
}

// 不同文件类型的期望零字节比例
bool IsZeroRatioNormal(double ratio, const string& ext) {
    if (ext == "jpg" || ext == "png" || ext == "zip") {
        return ratio < 0.05;  // 压缩数据零字节很少
    }
    if (ext == "exe" || ext == "dll") {
        return ratio < 0.15;  // PE文件有一些零填充
    }
    if (ext == "pdf") {
        return ratio < 0.10;
    }
    return ratio < 0.20;
}
```

---

### 4. 综合评分模型

```cpp
struct FileIntegrityScore {
    double entropyScore;      // 熵值得分 (0-1)
    double structureScore;    // 结构验证得分 (0-1)
    double statisticalScore;  // 统计分析得分 (0-1)
    double footerScore;       // 文件尾验证得分 (0-1)
    double overallScore;      // 综合得分 (0-1)
    string diagnosis;         // 诊断信息
};

FileIntegrityScore CalculateIntegrityScore(
    const BYTE* data, size_t size, const string& ext) {

    FileIntegrityScore score = {0};

    // 1. 熵值分析 (权重: 25%)
    double entropy = CalculateEntropy(data, size);
    score.entropyScore = EvaluateEntropyForType(entropy, ext);

    // 2. 结构验证 (权重: 35%)
    score.structureScore = ValidateFileStructure(data, size, ext);

    // 3. 统计分析 (权重: 20%)
    double zeroRatio = CalculateZeroRatio(data, size);
    double chiSquare = CalculateChiSquare(data, min(size, (size_t)65536));
    score.statisticalScore = EvaluateStatistics(zeroRatio, chiSquare, ext);

    // 4. 文件尾验证 (权重: 20%)
    score.footerScore = ValidateFooter(data, size, ext);

    // 综合评分
    score.overallScore =
        score.entropyScore * 0.25 +
        score.structureScore * 0.35 +
        score.statisticalScore * 0.20 +
        score.footerScore * 0.20;

    // 生成诊断
    if (score.overallScore >= 0.8) {
        score.diagnosis = "High confidence - likely intact";
    } else if (score.overallScore >= 0.6) {
        score.diagnosis = "Medium confidence - may have minor issues";
    } else if (score.overallScore >= 0.4) {
        score.diagnosis = "Low confidence - likely damaged";
    } else {
        score.diagnosis = "Very low confidence - probably corrupted";
    }

    return score;
}
```

---

## 实现建议

### 验证流程

```
1. 签名匹配 → 基础置信度
2. 文件尾检测 → 调整置信度
3. 熵值分析 → 检测损坏区域
4. 结构验证 → 验证文件格式
5. 统计分析 → 辅助判断
6. 综合评分 → 最终决策
```

### 性能考虑

| 检测方法 | 时间复杂度 | 建议 |
|---------|-----------|------|
| 熵值计算 | O(n) | 可采样计算 |
| 结构验证 | O(n) | 只读取必要部分 |
| CRC校验 | O(n) | 对大文件可跳过 |
| 卡方检验 | O(n) | 采样前64KB |

### 阈值设置

```cpp
// 推荐阈值
const double MIN_INTEGRITY_SCORE = 0.5;   // 最低完整性得分
const double MIN_ENTROPY_JPEG = 6.5;      // JPEG最低熵值
const double MAX_ZERO_RATIO = 0.10;       // 最大零字节比例
const double ENTROPY_VARIANCE_THRESHOLD = 1.5;  // 熵值突变阈值
```

---

## 局限性

1. **无法检测语义损坏** - 图片可能打开但内容错位
2. **碎片化文件** - 无法重组非连续数据
3. **加密文件** - 与随机数据难以区分
4. **新格式** - 需要为每种格式编写验证逻辑

---

## 参考文献

1. Shannon, C.E. "A Mathematical Theory of Communication" (1948)
2. Garfinkel, S. "Carving contiguous and fragmented files with fast object validation" (2007)
3. NIST SP 800-86 "Guide to Integrating Forensic Techniques into Incident Response"
