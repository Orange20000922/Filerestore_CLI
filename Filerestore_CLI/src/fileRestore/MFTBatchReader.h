#pragma once
#include <Windows.h>
#include <vector>
#include <map>
#include "MFTStructures.h"
#include "MFTReader.h"

// 定义ULONGLONG的最大值(用于标记无效值)
#ifndef ULONGLONG_MAX
#define ULONGLONG_MAX 0xFFFFFFFFFFFFFFFFULL
#endif

using namespace std;

// MFT 批量缓冲读取器 - 高性能版本
// 使用MFTReader作为底层，添加缓存来减少重复读取
class MFTBatchReader
{
private:
    MFTReader* reader;  // 底层MFT读取器（处理碎片化等复杂情况）
    ULONGLONG totalMFTRecords;

    // 缓存配置
    static const ULONGLONG CACHE_SIZE = 1024;  // 缓存1024条记录

    // 记录缓存结构
    struct CachedRecord {
        vector<BYTE> data;      // 记录数据
        DWORD lastAccessTime;   // 最后访问时间（用于LRU淘汰）
        bool valid;             // 是否有效
    };

    map<ULONGLONG, CachedRecord> recordCache;  // 记录缓存（key=recordNumber）
    DWORD accessCounter;  // 访问计数器

    // 内部方法
    void EvictOldestRecord();

public:
    MFTBatchReader();
    ~MFTBatchReader();

    // 初始化
    bool Initialize(MFTReader* mftReader);

    // 读取单条记录
    bool ReadMFTRecord(ULONGLONG recordNumber, vector<BYTE>& record);

    // 批量读取
    bool ReadMFTBatch(ULONGLONG startRecord, ULONGLONG count,
                      vector<vector<BYTE>>& records);

    // 清理缓存
    void ClearCache();

    // 信息
    ULONGLONG GetTotalRecords() const { return totalMFTRecords; }
    ULONGLONG GetMFTSize() const { return reader ? reader->GetTotalMFTRecords() : 0; }
    DWORD GetBytesPerRecord() const { return reader ? reader->GetBytesPerFileRecord() : 0; }

    // 获取缓存统计
    size_t GetCacheSize() const { return recordCache.size(); }
};
