# MFT 批量读取优化指南

## 背景

原先尝试使用 Windows 内存映射 (`CreateFileMapping` / `MapViewOfFile`) 来优化 MFT 读取性能，但由于 **Windows 不支持对原始卷设备进行内存映射**，会导致错误码 87 (ERROR_INVALID_PARAMETER)。

第二次尝试直接使用大缓冲区 `ReadFile()` 批量读取，但由于 **MFT 可能是碎片化的**（分散在磁盘不同位置），直接计算偏移量会读取到错误的数据。

最终方案使用 **包装器模式**：MFTBatchReader 作为 MFTReader 的缓存包装器，利用 MFTReader 已有的碎片化处理逻辑，同时添加记录缓存来减少重复读取。

---

## 架构设计

```
┌─────────────────────────────────────────┐
│            DeletedFileScanner           │
│   (扫描逻辑，调用 BatchReader)            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│            MFTBatchReader               │
│   (包装器 + LRU 记录缓存)                 │
│   - 缓存最近访问的 1024 条记录            │
│   - 减少重复读取                          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│              MFTReader                  │
│   (底层读取，处理碎片化 MFT)              │
│   - GetMFTRecordLCN() 处理碎片化         │
│   - ReadClusters() 读取簇数据            │
└─────────────────────────────────────────┘
```

---

## 使用方法

### 自动模式（推荐）
```cpp
MFTReader reader;
reader.OpenVolume('C');

MFTParser parser(&reader);
PathResolver resolver(&reader, &parser);
DeletedFileScanner scanner(&reader, &parser, &resolver);

// 自动选择最优扫描方式
// - 扫描 > 1000条记录 → 使用批量缓冲读取（高性能）
// - 扫描 ≤ 1000条记录 → 使用传统方式
vector<DeletedFileInfo> files = scanner.ScanDeletedFiles(0);  // 0 = 扫描全部
```

### 手动控制
```cpp
// 强制使用批量缓冲读取
scanner.SetUseBatchReading(true);
vector<DeletedFileInfo> files = scanner.ScanDeletedFiles(100000);

// 强制使用传统方式
scanner.SetUseBatchReading(false);
vector<DeletedFileInfo> files = scanner.ScanDeletedFiles(100000);

// 直接调用批量读取扫描
vector<DeletedFileInfo> files = scanner.ScanDeletedFilesBatch(0);
```

---

## 技术细节

### MFTBatchReader 类（包装器模式）

```cpp
class MFTBatchReader {
private:
    MFTReader* reader;  // 底层MFT读取器

    // 缓存配置
    static const ULONGLONG CACHE_SIZE = 1024;  // 缓存1024条记录

    // LRU记录缓存
    map<ULONGLONG, CachedRecord> recordCache;

public:
    bool Initialize(MFTReader* mftReader);
    bool ReadMFTRecord(ULONGLONG recordNumber, vector<BYTE>& record);
    bool ReadMFTBatch(ULONGLONG startRecord, ULONGLONG count, ...);
    void ClearCache();
};
```

### 关键特性

1. **包装器模式** - 不直接访问磁盘，而是包装 MFTReader
2. **利用现有逻辑** - 自动处理 MFT 碎片化（通过 MFTReader）
3. **LRU 记录缓存** - 缓存最近访问的 1024 条记录
4. **顺序扫描优化** - 在顺序扫描时缓存命中率接近 0%（因为不重复访问），但代码简洁可靠

### 性能说明

对于**顺序扫描**（扫描所有删除文件）：
- 缓存命中率 ≈ 0%（每条记录只访问一次）
- 性能与直接使用 MFTReader 相同
- 主要优势：代码可靠，正确处理碎片化

对于**随机访问或重复查询**：
- 缓存命中率可能很高
- 减少重复磁盘读取

---

## 为什么不用内存映射？

Windows 的 `CreateFileMapping()` 和 `MapViewOfFile()` API 设计用于常规文件，**不支持原始卷设备**（如 `\\.\C:`）。

当尝试对卷句柄调用 `CreateFileMapping()` 时，返回错误码 87 (ERROR_INVALID_PARAMETER)。

## 为什么不用直接偏移读取？

MFT 可能是**碎片化的**（分散在磁盘不同位置）。直接使用 `mftStartOffset + recordOffset` 计算偏移量，在碎片化的 MFT 上会读取到错误的数据（导致99%记录签名验证失败）。

必须使用 `MFTReader::GetMFTRecordLCN()` 来获取记录的正确物理位置。

---

## 注意事项

### 1. 权限要求
- 需要**管理员权限**才能访问原始卷
- 建议以管理员身份运行程序

### 2. 内存占用
- 缓冲区最大占用 8MB
- 比内存映射方案更可控

### 3. 系统兼容性
- ✅ Windows 7/8/10/11
- ✅ 32位和64位系统
- ✅ NTFS卷
- ❌ FAT32/exFAT（不支持MFT）

---

**更新时间**: 2025-12-31
**版本**: 0.4.0 (包装器模式 + LRU缓存)

## 变更历史

- **v0.4.0** (2025-12-31) - 改为包装器模式，修复MFT碎片化问题
- **v0.3.0** (2025-12-31) - 尝试批量缓冲读取（失败：未处理碎片化）
- **v0.2.0** (2025-12-30) - 尝试内存映射（失败：错误码87）
- **v0.1.0** - 原始传统读取方式
