#include "PathResolver.h"
#include <iostream>
#include <fstream>
#include <set>
#include "Logger.h"
using namespace std;

PathResolver::PathResolver(MFTReader* mftReader, MFTParser* mftParser)
    : reader(mftReader), parser(mftParser), cacheHits(0), cacheMisses(0) {
}

PathResolver::~PathResolver() {
}

vector<wstring> PathResolver::ParsePath(const string& path) {
    vector<wstring> components;

    // 跳过驱动器字母 (例如 "C:")
    size_t start = 0;
    if (path.length() >= 2 && path[1] == ':') {
        start = 2;
    }

    // 跳过前导斜杠
    while (start < path.length() && (path[start] == '\\' || path[start] == '/')) {
        start++;
    }

    // 分割路径
    string currentPath = path.substr(start);
    size_t pos = 0;

    while ((pos = currentPath.find_first_of("\\/")) != string::npos) {
        if (pos > 0) {
            string component = currentPath.substr(0, pos);
            // 转换为宽字符
            int len = MultiByteToWideChar(CP_ACP, 0, component.c_str(), -1, NULL, 0);
            if (len > 0) {
                wstring wComponent(len - 1, 0);
                MultiByteToWideChar(CP_ACP, 0, component.c_str(), -1, &wComponent[0], len);
                components.push_back(wComponent);
            }
        }
        currentPath = currentPath.substr(pos + 1);
    }

    // 添加最后一个组件
    if (!currentPath.empty()) {
        int len = MultiByteToWideChar(CP_ACP, 0, currentPath.c_str(), -1, NULL, 0);
        if (len > 0) {
            wstring wComponent(len - 1, 0);
            MultiByteToWideChar(CP_ACP, 0, currentPath.c_str(), -1, &wComponent[0], len);
            components.push_back(wComponent);
        }
    }

    return components;
}

ULONGLONG PathResolver::FindInDirectory(ULONGLONG dirRecordNumber, const wstring& name) {
    vector<BYTE> dirRecord;
    if (!reader->ReadMFT(dirRecordNumber, dirRecord)) {
        cout << "读取目录记录失败: " << dirRecordNumber << endl;
        return 0;
    }

    // 获取索引根
    vector<BYTE> indexData;
    if (!parser->GetIndexRoot(dirRecord, indexData)) {
        cout << "目录中未找到索引根。" << endl;
        return 0;
    }

    PINDEX_ROOT indexRoot = (PINDEX_ROOT)indexData.data();
    BYTE* entryPtr = indexData.data() + sizeof(INDEX_ROOT);
    BYTE* endPtr = indexData.data() + indexRoot->TotalEntrySize + sizeof(INDEX_ROOT);

    // 遍历索引条目
    while (entryPtr < endPtr) {
        PINDEX_ENTRY entry = (PINDEX_ENTRY)entryPtr;

        // 检查是否为最后一个条目
        if (entry->Flags & INDEX_ENTRY_FLAG_LAST) {
            break;
        }

        // 获取文件名属性
        if (entry->StreamLength > 0) {
            PFILE_NAME_ATTRIBUTE fileNameAttr = (PFILE_NAME_ATTRIBUTE)(entryPtr + sizeof(INDEX_ENTRY));
            wstring fileName(fileNameAttr->FileName, fileNameAttr->FileNameLength);

            // 不区分大小写比较
            if (_wcsicmp(fileName.c_str(), name.c_str()) == 0) {
                // 提取MFT记录号（低48位）
                ULONGLONG recordNumber = entry->FileReference & 0xFFFFFFFFFFFF;
                return recordNumber;
            }
        }

        entryPtr += entry->EntryLength;
    }

    return 0;
}

ULONGLONG PathResolver::FindFileRecordByPath(const string& filePath) {
    vector<wstring> pathComponents = ParsePath(filePath);

    if (pathComponents.empty()) {
        cout << "无效路径或根目录。" << endl;
        return 5; // 根目录的记录号是5
    }

    // 从根目录开始
    ULONGLONG currentRecord = 5;

    cout << "正在路径中搜索文件..." << endl;

    // 逐级查找
    for (size_t i = 0; i < pathComponents.size(); i++) {
        wcout << L"正在查找: " << pathComponents[i] << endl;

        ULONGLONG nextRecord = FindInDirectory(currentRecord, pathComponents[i]);

        if (nextRecord == 0) {
            wcout << L"未找到: " << pathComponents[i] << endl;
            return 0;
        }

        wcout << L"在MFT记录中找到: " << nextRecord << endl;
        currentRecord = nextRecord;
    }

    cout << "最终MFT记录号: " << currentRecord << endl;
    return currentRecord;
}

wstring PathResolver::ReconstructPath(ULONGLONG recordNumber) {
    if (recordNumber == 5) {
        return L"\\"; // 根目录
    }

    // 检查缓存
    auto it = pathCache.find(recordNumber);
    if (it != pathCache.end()) {
        cacheHits++;
        // LOG_DEBUG_FMT("Cache hit for record #%llu", recordNumber);
        return it->second;
    }

    cacheMisses++;

    vector<wstring> pathComponents;
    ULONGLONG currentRecord = recordNumber;
    set<ULONGLONG> visitedRecords; // 循环检测
    const int MAX_DEPTH = 100; // 最大深度限制

    // 向上遍历到根目录
    while (currentRecord != 5 && currentRecord > 0 && pathComponents.size() < MAX_DEPTH) {
        // 检查当前记录是否已在缓存中
        auto cachedIt = pathCache.find(currentRecord);
        if (cachedIt != pathCache.end()) {
            // 找到缓存的父路径，直接使用
            wstring cachedPath = cachedIt->second;

            // 反向构建剩余路径
            wstring path = cachedPath;
            for (auto it = pathComponents.rbegin(); it != pathComponents.rend(); ++it) {
                path += L"\\" + *it;
            }

            // 缓存最终路径
            pathCache[recordNumber] = path;
            TrimCacheIfNeeded();  // 防止缓存无限增长
            return path;
        }

        // 循环检测
        if (visitedRecords.count(currentRecord) > 0) {
            // cout << "[DEBUG] ReconstructPath: Loop detected at record " << currentRecord << endl;
            return L""; // 检测到循环引用
        }
        visitedRecords.insert(currentRecord);

        vector<BYTE> record;
        if (!reader->ReadMFT(currentRecord, record)) {
            // cout << "[DEBUG] ReconstructPath: Failed to read record " << currentRecord << endl;
            break;
        }

        ULONGLONG parentDir;
        wstring fileName = parser->GetFileNameFromRecord(record, parentDir);

        if (fileName.empty()) {
            // cout << "[DEBUG] ReconstructPath: No filename at record " << currentRecord << endl;
            break;
        }

        pathComponents.push_back(fileName);

        // 缓存中间节点（父目录）
        // 这样下次遇到同一父目录时可以直接使用

        currentRecord = parentDir;
    }

    if (pathComponents.size() >= MAX_DEPTH) {
        // cout << "[DEBUG] ReconstructPath: Max depth reached" << endl;
        return L""; // 达到最大深度
    }

    // 反向构建路径
    wstring path = L"\\";
    for (auto it = pathComponents.rbegin(); it != pathComponents.rend(); ++it) {
        path += *it;
        if (next(it) != pathComponents.rend()) {
            path += L"\\";
        }
    }

    // 缓存结果
    pathCache[recordNumber] = path;
    TrimCacheIfNeeded();  // 防止缓存无限增长

    // 同时缓存路径中的所有父目录
    wstring partialPath = L"\\";
    ULONGLONG currentRec = recordNumber;
    for (auto it = pathComponents.rbegin(); it != pathComponents.rend(); ++it) {
        partialPath += *it;
        // 这里可以进一步优化，缓存每个中间路径
        if (next(it) != pathComponents.rend()) {
            partialPath += L"\\";
        }
    }

    return path;
}

// 缓存大小管理:当缓存超过限制时,清理一半旧缓存
void PathResolver::TrimCacheIfNeeded() {
    if (pathCache.size() > MAX_CACHE_SIZE) {
        // 清理一半的缓存(简单策略:清理前面的一半)
        auto it = pathCache.begin();
        size_t toRemove = pathCache.size() / 2;
        for (size_t i = 0; i < toRemove && it != pathCache.end(); ) {
            it = pathCache.erase(it);
            i++;
        }
        LOG_INFO_FMT("Path cache trimmed: removed %zu entries, current size: %zu",
                     toRemove, pathCache.size());
    }
}

void PathResolver::ClearCache() {
    pathCache.clear();
    cacheHits = 0;
    cacheMisses = 0;
}

void PathResolver::GetCacheStats(ULONGLONG& hits, ULONGLONG& misses) {
    hits = cacheHits;
    misses = cacheMisses;
}

// ==================== 路径缓存持久化 ====================

string PathResolver::GetPathCachePath(char driveLetter) {
    char cachePath[MAX_PATH];
    GetTempPathA(MAX_PATH, cachePath);
    string cacheFile = string(cachePath) + "path_cache_" + driveLetter + ".cache";
    return cacheFile;
}

bool PathResolver::SavePathCache(const map<ULONGLONG, wstring>& cache, char driveLetter) {
    string cachePath = GetPathCachePath(driveLetter);
    
    LOG_INFO_FMT("Saving %zu path entries to cache: %s", cache.size(), cachePath.c_str());
    
    ofstream ofs(cachePath, ios::binary);
    if (!ofs.is_open()) {
        LOG_ERROR_FMT("Failed to open path cache file for writing: %s", cachePath.c_str());
        return false;
    }

    try {
        // 写入版本号和条目数量
        DWORD version = 1;
        ULONGLONG entryCount = cache.size();
        ofs.write((char*)&version, sizeof(version));
        ofs.write((char*)&entryCount, sizeof(entryCount));

        // 写入每个路径缓存条目
        for (const auto& entry : cache) {
            // 写入记录号
            ofs.write((char*)&entry.first, sizeof(entry.first));
            
            // 写入路径长度和内容
            DWORD pathLen = (DWORD)entry.second.length();
            ofs.write((char*)&pathLen, sizeof(pathLen));
            ofs.write((char*)entry.second.c_str(), pathLen * sizeof(wchar_t));
        }

        ofs.close();
        LOG_INFO_FMT("Successfully saved %zu path entries to cache", cache.size());
        return true;
    }
    catch (const exception& e) {
        LOG_ERROR_FMT("Exception while saving path cache: %s", e.what());
        ofs.close();
        return false;
    }
}

bool PathResolver::LoadPathCache(map<ULONGLONG, wstring>& cache, char driveLetter) {
    string cachePath = GetPathCachePath(driveLetter);
    
    LOG_INFO_FMT("Loading path cache from: %s", cachePath.c_str());
    
    ifstream ifs(cachePath, ios::binary);
    if (!ifs.is_open()) {
        LOG_DEBUG_FMT("Path cache file not found: %s", cachePath.c_str());
        return false;
    }

    try {
        // 读取版本号
        DWORD version;
        ifs.read((char*)&version, sizeof(version));
        if (version != 1) {
            LOG_ERROR_FMT("Unsupported path cache version: %d", version);
            ifs.close();
            return false;
        }

        // 读取条目数量
        ULONGLONG entryCount;
        ifs.read((char*)&entryCount, sizeof(entryCount));
        
        LOG_INFO_FMT("Loading %llu path entries from cache", entryCount);
        cache.clear();

        // 读取每个路径缓存条目
        for (ULONGLONG i = 0; i < entryCount; i++) {
            ULONGLONG recordNumber;
            ifs.read((char*)&recordNumber, sizeof(recordNumber));
            
            DWORD pathLen;
            ifs.read((char*)&pathLen, sizeof(pathLen));
            
            wstring path;
            path.resize(pathLen);
            ifs.read((char*)path.data(), pathLen * sizeof(wchar_t));
            
            cache[recordNumber] = path;
        }

        ifs.close();
        LOG_INFO_FMT("Successfully loaded %zu path entries from cache", cache.size());
        return true;
    }
    catch (const exception& e) {
        LOG_ERROR_FMT("Exception while loading path cache: %s", e.what());
        ifs.close();
        return false;
    }
}

bool PathResolver::SaveCache(char driveLetter) {
    return SavePathCache(pathCache, driveLetter);
}

bool PathResolver::LoadCache(char driveLetter) {
    bool result = LoadPathCache(pathCache, driveLetter);
    if (result) {
        LOG_INFO_FMT("Loaded path cache with %zu entries", pathCache.size());
        // 重置统计（因为缓存是从文件加载的）
        cacheHits = 0;
        cacheMisses = 0;
    }
    return result;
}
