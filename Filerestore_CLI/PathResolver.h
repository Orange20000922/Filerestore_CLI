#pragma once
#include <Windows.h>
#include <vector>
#include <string>
#include <map>
#include "MFTStructures.h"
#include "MFTReader.h"
#include "MFTParser.h"

using namespace std;

// 路径解析器类 - 负责路径解析和重建
class PathResolver
{
private:
    MFTReader* reader;
    MFTParser* parser;

    // 路径缓存：记录号 -> 完整路径
    map<ULONGLONG, wstring> pathCache;

    // 缓存大小限制(防止内存耗尽)
    static const size_t MAX_CACHE_SIZE = 50000;  // 最多缓存5万条路径

    // 缓存统计
    ULONGLONG cacheHits;
    ULONGLONG cacheMisses;

    // 内部方法:缓存大小管理
    void TrimCacheIfNeeded();

public:
    PathResolver(MFTReader* mftReader, MFTParser* mftParser);
    ~PathResolver();

    // 路径解析
    vector<wstring> ParsePath(const string& path);

    // 路径查找
    ULONGLONG FindFileRecordByPath(const string& filePath);
    ULONGLONG FindInDirectory(ULONGLONG dirRecordNumber, const wstring& name);

    // 路径重建（用于已删除文件）
    wstring ReconstructPath(ULONGLONG recordNumber);

    // 缓存管理
    void ClearCache();
    void GetCacheStats(ULONGLONG& hits, ULONGLONG& misses);

    // 路径缓存持久化
    static bool SavePathCache(const map<ULONGLONG, wstring>& cache, char driveLetter);
    static bool LoadPathCache(map<ULONGLONG, wstring>& cache, char driveLetter);
    static string GetPathCachePath(char driveLetter);

    // 加载/保存当前缓存
    bool SaveCache(char driveLetter);
    bool LoadCache(char driveLetter);

    // 获取缓存引用（用于预加载）
    map<ULONGLONG, wstring>& GetCacheRef() { return pathCache; }
    size_t GetCacheSize() const { return pathCache.size(); }
};
