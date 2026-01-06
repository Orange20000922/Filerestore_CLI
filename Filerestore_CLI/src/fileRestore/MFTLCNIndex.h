#pragma once
#include <Windows.h>
#include <vector>
#include <map>
#include <set>
#include "MFTReader.h"
#include "MFTStructures.h"

using namespace std;

// LCN 到 MFT 记录的映射信息
struct LCNMappingInfo {
    ULONGLONG mftRecordNumber;  // MFT 记录号
    ULONGLONG startLCN;         // 数据起始 LCN
    ULONGLONG clusterCount;     // 簇数量
    FILETIME creationTime;      // 创建时间
    FILETIME modificationTime;  // 修改时间
    FILETIME accessTime;        // 访问时间
    wstring fileName;           // 文件名
    bool isDeleted;             // 是否已删除
};

// MFT LCN 索引 - 用于快速查找
class MFTLCNIndex {
private:
    MFTReader* reader;

    // LCN 范围到 MFT 记录的映射
    // key: startLCN, value: 映射信息
    map<ULONGLONG, LCNMappingInfo> lcnIndex;

    // 是否已构建索引
    bool indexBuilt;

    // 解析 Data Runs
    bool ParseDataRuns(const BYTE* dataRun, size_t maxLen,
                       vector<pair<ULONGLONG, ULONGLONG>>& runs);

    // 从 MFT 记录提取时间戳
    bool ExtractTimestamps(const BYTE* record, size_t recordSize,
                           FILETIME& creation, FILETIME& modification, FILETIME& access);

    // 从 MFT 记录提取文件名
    wstring ExtractFileName(const BYTE* record, size_t recordSize);

public:
    MFTLCNIndex(MFTReader* mftReader);
    ~MFTLCNIndex();

    // 构建 LCN 索引（扫描所有 MFT 记录）
    // 这是一个耗时操作，建议在开始扫描前调用一次
    bool BuildIndex(bool includeActiveFiles = false, bool showProgress = true);

    // 根据 LCN 查找匹配的 MFT 记录
    // 返回包含该 LCN 的所有 MFT 记录
    vector<LCNMappingInfo> FindByLCN(ULONGLONG lcn);

    // 根据 LCN 范围查找
    vector<LCNMappingInfo> FindByLCNRange(ULONGLONG startLCN, ULONGLONG endLCN);

    // 获取索引大小
    size_t GetIndexSize() const { return lcnIndex.size(); }

    // 检查索引是否已构建
    bool IsIndexBuilt() const { return indexBuilt; }

    // 清除索引
    void ClearIndex();
};
