# MFT 扫描性能优化方案

## 已实现：路径重建缓存（方案1）

### 优化原理：
- 使用 `map<ULONGLONG, wstring>` 缓存已重建的路径
- 当遇到已缓存的父目录时，直接使用缓存结果
- 避免重复读取相同的 MFT 记录

### 性能提升：
- **缓存命中率预期：60-90%**（取决于目录结构）
- **路径重建速度提升：5-10倍**（对于同一目录下的多个文件）
- **内存开销：每个缓存条目约 100-200 字节**

### 使用示例：
```cpp
// 自动使用缓存，无需额外代码
wstring path = pathResolver->ReconstructPath(recordNumber);

// 查看缓存统计
ULONGLONG hits, misses;
pathResolver->GetCacheStats(hits, misses);
cout << "Cache hit rate: " << (hits * 100.0 / (hits + misses)) << "%" << endl;
```

---

## 可选方案：多线程扫描（方案2）

### 实现思路：

#### **1. 分段扫描**
```cpp
// 将 MFT 记录分成多个段，每个线程处理一段
const int THREAD_COUNT = 4;
ULONGLONG recordsPerThread = totalRecords / THREAD_COUNT;

vector<thread> threads;
vector<vector<DeletedFileInfo>> results(THREAD_COUNT);

for (int i = 0; i < THREAD_COUNT; i++) {
    ULONGLONG startRecord = i * recordsPerThread;
    ULONGLONG endRecord = (i == THREAD_COUNT - 1) ? totalRecords : (i + 1) * recordsPerThread;

    threads.push_back(thread([&, i, startRecord, endRecord]() {
        results[i] = ScanRange(startRecord, endRecord);
    }));
}

// 等待所有线程完成
for (auto& t : threads) {
    t.join();
}

// 合并结果
vector<DeletedFileInfo> allResults;
for (const auto& result : results) {
    allResults.insert(allResults.end(), result.begin(), result.end());
}
```

#### **2. 线程安全的 MFTReader**
需要为每个线程创建独立的 MFTReader 实例，或者使用互斥锁保护共享资源：

```cpp
class ThreadSafeMFTReader {
private:
    MFTReader* reader;
    mutex mtx;

public:
    bool ReadMFT(ULONGLONG recordNumber, vector<BYTE>& record) {
        lock_guard<mutex> lock(mtx);
        return reader->ReadMFT(recordNumber, record);
    }
};
```

#### **3. 线程池实现**
```cpp
class MFTScannerThreadPool {
private:
    vector<thread> workers;
    queue<ScanTask> tasks;
    mutex queueMutex;
    condition_variable condition;
    bool stop;

public:
    MFTScannerThreadPool(size_t threads);
    void EnqueueTask(ScanTask task);
    vector<DeletedFileInfo> GetResults();
};
```

### 性能分析：

#### **优点：**
- ✅ CPU 密集型操作（路径重建、属性解析）可以并行化
- ✅ 理论上可以提升 2-4 倍速度（取决于 CPU 核心数）

#### **缺点：**
- ❌ 磁盘 I/O 是瓶颈，多线程可能导致随机读取，反而降低性能
- ❌ 实现复杂，需要处理线程同步、资源竞争
- ❌ 调试困难，容易出现竞态条件

### 建议：
**不推荐实现多线程扫描**，原因：
1. **磁盘 I/O 是主要瓶颈**，不是 CPU
2. **顺序读取比随机读取快得多**（HDD 尤其明显）
3. **路径缓存已经解决了主要的 CPU 瓶颈**

---

## 其他优化建议

### **方案3：批量读取 MFT 记录**

当前实现是逐条读取 MFT 记录，可以改为批量读取：

```cpp
// 一次读取多个簇，包含多个 MFT 记录
const int RECORDS_PER_BATCH = 64;
vector<BYTE> batchData;
ReadClusters(startLCN, clustersNeeded, batchData);

// 从批量数据中提取单个记录
for (int i = 0; i < RECORDS_PER_BATCH; i++) {
    BYTE* recordPtr = batchData.data() + i * bytesPerFileRecord;
    ProcessRecord(recordPtr);
}
```

**性能提升：** 减少磁盘 I/O 次数，提升 20-50%

---

### **方案4：跳过明显无效的记录**

```cpp
// 快速检查记录是否有效
PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)record.data();

// 跳过从未使用的记录
if (header->Signature != 'ELIF') {
    continue; // 不是有效的 FILE 记录
}

// 跳过活动文件（只扫描已删除）
if (header->Flags & 0x01) {
    continue; // 文件仍在使用
}
```

**性能提升：** 减少不必要的处理，提升 10-30%

---

### **方案5：限制扫描范围**

```cpp
// 只扫描最近的 N 条记录（新删除的文件）
ULONGLONG recentRecords = min(totalRecords, 50000ULL);

// 或者只扫描特定范围
ULONGLONG startRecord = 100000;
ULONGLONG endRecord = 200000;
```

**性能提升：** 大幅减少扫描时间，适合快速查找最近删除的文件

---

## 性能对比（预估）

| 优化方案 | 实现难度 | 性能提升 | 推荐度 |
|---------|---------|---------|--------|
| 路径缓存 | ⭐ 简单 | 5-10倍（路径重建） | ⭐⭐⭐⭐⭐ 强烈推荐 |
| 批量读取 | ⭐⭐ 中等 | 20-50% | ⭐⭐⭐⭐ 推荐 |
| 跳过无效记录 | ⭐ 简单 | 10-30% | ⭐⭐⭐⭐ 推荐 |
| 限制扫描范围 | ⭐ 简单 | 50-90% | ⭐⭐⭐ 适合特定场景 |
| 多线程扫描 | ⭐⭐⭐⭐ 复杂 | 可能降低性能 | ⭐ 不推荐 |

---

## 总结

**已实现的路径缓存是最有效的优化**，可以显著提升性能且实现简单。

如果需要进一步优化，建议按以下顺序实施：
1. ✅ **路径缓存**（已实现）
2. **批量读取 MFT 记录**
3. **跳过无效记录的优化**
4. **根据需求限制扫描范围**

**不建议实现多线程扫描**，因为磁盘 I/O 是瓶颈，多线程反而可能降低性能。
