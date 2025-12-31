# 数据覆盖检测功能使用指南

## 功能概述

数据覆盖检测功能可以分析已删除文件的数据簇，判断哪些数据已被覆盖，哪些数据仍然可以恢复。这对于评估文件恢复的可行性非常重要。

## 核心组件

### 1. OverwriteDetector 类
负责检测数据簇是否被覆盖的核心类。

### 2. ClusterStatus 结构
存储单个簇的状态信息：
- `clusterNumber`: 簇号
- `isOverwritten`: 是否被覆盖
- `isAllocated`: 是否已分配给其他文件
- `dataEntropy`: 数据熵值 (0.0-8.0)
- `overwriteReason`: 覆盖原因描述

### 3. OverwriteDetectionResult 结构
存储整个文件的覆盖检测结果：
- `totalClusters`: 总簇数
- `overwrittenClusters`: 被覆盖的簇数
- `availableClusters`: 可用的簇数
- `overwritePercentage`: 覆盖百分比
- `isFullyAvailable`: 数据是否完全可用
- `isPartiallyAvailable`: 数据是否部分可用

## 使用方法

### 方法1：通过 FileRestore 类使用（推荐）

```cpp
#include "FileRestore.h"

int main() {
    // 创建 FileRestore 实例
    FileRestore* restore = new FileRestore();

    // 检测指定MFT记录号的文件覆盖情况
    char driveLetter = 'C';
    ULONGLONG recordNumber = 12345;  // 要检测的MFT记录号

    OverwriteDetectionResult result = restore->DetectFileOverwrite(driveLetter, recordNumber);

    // 分析结果
    if (result.isFullyAvailable) {
        cout << "好消息！文件数据完全可用，可以完整恢复。" << endl;
    } else if (result.isPartiallyAvailable) {
        cout << "文件部分可用，可以尝试部分恢复。" << endl;
        cout << "可恢复比例: " << (100.0 - result.overwritePercentage) << "%" << endl;
    } else {
        cout << "文件数据已完全被覆盖，无法恢复。" << endl;
    }

    delete restore;
    return 0;
}
```

### 方法2：直接使用 OverwriteDetector 类

```cpp
#include "OverwriteDetector.h"
#include "MFTReader.h"

int main() {
    // 创建 MFTReader
    MFTReader* reader = new MFTReader();
    if (!reader->OpenVolume('C')) {
        cout << "Failed to open volume" << endl;
        return 1;
    }

    // 创建 OverwriteDetector
    OverwriteDetector* detector = new OverwriteDetector(reader);

    // 读取MFT记录
    vector<BYTE> mftRecord;
    ULONGLONG recordNumber = 12345;
    if (!reader->ReadMFT(recordNumber, mftRecord)) {
        cout << "Failed to read MFT record" << endl;
        return 1;
    }

    // 执行覆盖检测
    OverwriteDetectionResult result = detector->DetectOverwrite(mftRecord);

    // 获取详细报告
    string report = detector->GetDetectionReport(result);
    cout << report << endl;

    // 清理
    delete detector;
    reader->CloseVolume();
    delete reader;

    return 0;
}
```

### 方法3：检测单个簇

```cpp
#include "OverwriteDetector.h"

// 假设已经创建了 detector 和 reader
OverwriteDetector* detector = new OverwriteDetector(reader);

// 检测单个簇
ULONGLONG clusterNumber = 1000;
ClusterStatus status = detector->CheckCluster(clusterNumber);

cout << "Cluster " << status.clusterNumber << ": ";
if (status.isOverwritten) {
    cout << "OVERWRITTEN - " << status.overwriteReason << endl;
} else {
    cout << "AVAILABLE" << endl;
}
cout << "Data Entropy: " << status.dataEntropy << endl;
```

### 方法4：批量检测多个簇

```cpp
// 准备要检测的簇号列表
vector<ULONGLONG> clusterNumbers = {1000, 1001, 1002, 1003, 1004};

// 批量检测
vector<ClusterStatus> results = detector->CheckClusters(clusterNumbers);

// 分析结果
int availableCount = 0;
for (const auto& status : results) {
    if (!status.isOverwritten) {
        availableCount++;
    }
}

cout << "Available clusters: " << availableCount << "/" << results.size() << endl;
```

## 检测原理

### 1. 全零检测
如果簇中所有字节都是0，说明可能被快速格式化或清零。

### 2. 相同值检测
如果簇中所有字节都是同一个值，说明可能被擦除工具覆盖。

### 3. 熵值分析
- **低熵值 (<1.0)**: 可能被清零或填充
- **中等熵值 (1.0-6.0)**: 可能是原始数据
- **高熵值 (6.0-7.8)**: 可能是压缩或加密数据
- **极高熵值 (>7.8)**: 可能是随机覆盖或加密数据

### 4. 文件签名检测
检测常见文件格式的签名：
- JPEG: `FF D8 FF`
- PNG: `89 50 4E 47`
- PDF: `25 50 44 46`
- ZIP/DOCX: `50 4B 03 04`
- EXE/DLL: `4D 5A`

### 5. 文本特征检测
检测是否包含可打印的ASCII字符（文本文件）。

## 输出示例

```
=== Overwrite Detection Started ===
Drive: C:
MFT Record Number: 12345

File Status: DELETED
File Name: document.txt

Analyzing data clusters...
This may take a while for large files...

=== Overwrite Detection Report ===
Total Clusters: 10
Available Clusters: 7
Overwritten Clusters: 3
Overwrite Percentage: 30.00%

Status: PARTIALLY AVAILABLE - Some data can be recovered

Cluster Details (first 10):
  Cluster 1000: AVAILABLE (Entropy: 4.52) - Medium entropy, possibly original data
  Cluster 1001: AVAILABLE (Entropy: 5.23) - Medium entropy, possibly original data
  Cluster 1002: OVERWRITTEN (Entropy: 0.00) - All zeros (formatted or wiped)
  Cluster 1003: AVAILABLE (Entropy: 4.87) - Contains valid file structure
  Cluster 1004: OVERWRITTEN (Entropy: 7.95) - Very high entropy, likely random data or encrypted
  Cluster 1005: AVAILABLE (Entropy: 3.21) - Medium entropy, possibly original data
  Cluster 1006: OVERWRITTEN (Entropy: 0.00) - All zeros (formatted or wiped)
  Cluster 1007: AVAILABLE (Entropy: 4.65) - Medium entropy, possibly original data
  Cluster 1008: AVAILABLE (Entropy: 5.01) - Medium entropy, possibly original data
  Cluster 1009: AVAILABLE (Entropy: 4.33) - Medium entropy, possibly original data
===================================
```

## 集成到扫描流程

可以在扫描已删除文件时自动进行覆盖检测：

```cpp
FileRestore* restore = new FileRestore();

// 扫描已删除文件
vector<DeletedFileInfo> deletedFiles = restore->ScanDeletedFiles('C', 10000);

// 对每个文件进行覆盖检测（可选）
for (auto& fileInfo : deletedFiles) {
    if (!fileInfo.isDirectory && fileInfo.fileSize > 0) {
        OverwriteDetectionResult result = restore->DetectFileOverwrite('C', fileInfo.recordNumber);

        // 更新文件信息
        fileInfo.overwriteDetected = (result.overwrittenClusters > 0);
        fileInfo.overwritePercentage = result.overwritePercentage;
        fileInfo.totalClusters = result.totalClusters;
        fileInfo.availableClusters = result.availableClusters;
        fileInfo.overwrittenClusters = result.overwrittenClusters;

        // 只显示部分可恢复的文件
        if (result.isPartiallyAvailable || result.isFullyAvailable) {
            wcout << L"File: " << fileInfo.fileName << endl;
            cout << "  Recovery possibility: " << (100.0 - result.overwritePercentage) << "%" << endl;
        }
    }
}
```

## 性能考虑

1. **大文件检测**: 对于大文件（数千个簇），检测可能需要较长时间
2. **批量检测**: 使用 `CheckClusters()` 批量检测可以提高效率
3. **采样检测**: 对于非常大的文件，可以只检测部分簇作为样本

## 注意事项

1. **权限要求**: 需要管理员权限才能读取磁盘簇数据
2. **只读操作**: 覆盖检测是只读操作，不会修改任何数据
3. **准确性**: 检测结果基于启发式算法，可能存在误判
4. **常驻文件**: 对于常驻文件（数据存储在MFT记录中），始终返回完全可用

## 未来改进

1. **$Bitmap集成**: 读取NTFS的$Bitmap属性，准确判断簇是否已分配
2. **更多文件格式**: 支持更多文件格式的签名检测
3. **机器学习**: 使用机器学习算法提高覆盖检测的准确性
4. **并行检测**: 使用多线程加速大文件的覆盖检测
