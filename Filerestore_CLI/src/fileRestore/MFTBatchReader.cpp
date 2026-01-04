#include "MFTBatchReader.h"
#include "Logger.h"
#include <iostream>
#include <algorithm>

using namespace std;

MFTBatchReader::MFTBatchReader()
    : reader(nullptr), totalMFTRecords(0), accessCounter(0) {
}

MFTBatchReader::~MFTBatchReader() {
    ClearCache();
}

bool MFTBatchReader::Initialize(MFTReader* mftReader) {
    if (mftReader == nullptr) {
        LOG_ERROR("MFTReader is null");
        return false;
    }

    reader = mftReader;
    totalMFTRecords = reader->GetTotalMFTRecords();

    LOG_INFO_FMT("MFT Batch Reader initialized (wrapper mode):");
    LOG_INFO_FMT("  Total records: %llu", totalMFTRecords);
    LOG_INFO_FMT("  Bytes per record: %u", reader->GetBytesPerFileRecord());
    LOG_INFO_FMT("  Cache size: %llu records", CACHE_SIZE);

    return true;
}

void MFTBatchReader::EvictOldestRecord() {
    if (recordCache.empty()) return;

    // 找到最旧的记录（最小的lastAccessTime）
    auto oldest = recordCache.begin();
    for (auto it = recordCache.begin(); it != recordCache.end(); ++it) {
        if (it->second.lastAccessTime < oldest->second.lastAccessTime) {
            oldest = it;
        }
    }

    recordCache.erase(oldest);
}

bool MFTBatchReader::ReadMFTRecord(ULONGLONG recordNumber, vector<BYTE>& record) {
    if (reader == nullptr) {
        LOG_ERROR("Reader is not initialized");
        return false;
    }

    if (recordNumber >= totalMFTRecords) {
        LOG_WARNING_FMT("Record number %llu out of range (max: %llu)",
                       recordNumber, totalMFTRecords);
        return false;
    }

    // 检查缓存
    auto it = recordCache.find(recordNumber);
    if (it != recordCache.end()) {
        // 缓存命中
        it->second.lastAccessTime = ++accessCounter;
        if (it->second.valid) {
            record = it->second.data;
            return true;
        } else {
            // 之前读取失败，返回空
            return false;
        }
    }

    // 缓存未命中，需要从底层读取

    // 如果缓存已满，淘汰最旧的记录
    if (recordCache.size() >= CACHE_SIZE) {
        EvictOldestRecord();
    }

    // 使用MFTReader读取记录
    CachedRecord cached;
    cached.lastAccessTime = ++accessCounter;
    cached.valid = reader->ReadMFT(recordNumber, cached.data);

    // 添加到缓存
    recordCache[recordNumber] = cached;

    if (cached.valid) {
        record = cached.data;
        return true;
    }

    return false;
}

bool MFTBatchReader::ReadMFTBatch(ULONGLONG startRecord, ULONGLONG count,
                                   vector<vector<BYTE>>& records) {
    if (reader == nullptr) {
        LOG_ERROR("Reader is not initialized");
        return false;
    }

    records.clear();
    records.reserve((size_t)count);

    // 使用MFTReader的批量读取功能（已优化）
    // 但首先检查缓存
    ULONGLONG cachedCount = 0;
    ULONGLONG missCount = 0;

    for (ULONGLONG i = 0; i < count; i++) {
        ULONGLONG recordNumber = startRecord + i;
        if (recordNumber >= totalMFTRecords) {
            break;
        }

        // 检查缓存
        auto it = recordCache.find(recordNumber);
        if (it != recordCache.end()) {
            // 缓存命中
            it->second.lastAccessTime = ++accessCounter;
            cachedCount++;
            if (it->second.valid) {
                records.push_back(it->second.data);
            } else {
                records.push_back(vector<BYTE>());
            }
        } else {
            missCount++;

            // 缓存未命中，从底层读取
            if (recordCache.size() >= CACHE_SIZE) {
                EvictOldestRecord();
            }

            CachedRecord cached;
            cached.lastAccessTime = ++accessCounter;
            cached.valid = reader->ReadMFT(recordNumber, cached.data);

            // 添加到缓存
            recordCache[recordNumber] = cached;

            if (cached.valid) {
                records.push_back(cached.data);
            } else {
                records.push_back(vector<BYTE>());
            }
        }
    }

    if (cachedCount + missCount > 0) {
        LOG_DEBUG_FMT("ReadMFTBatch: cached=%llu, miss=%llu, hit_rate=%.1f%%",
                     cachedCount, missCount,
                     (double)cachedCount / (cachedCount + missCount) * 100.0);
    }

    return true;
}

void MFTBatchReader::ClearCache() {
    recordCache.clear();
    accessCounter = 0;
    LOG_DEBUG("Record cache cleared");
}
