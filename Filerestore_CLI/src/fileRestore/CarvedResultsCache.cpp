#include "CarvedResultsCache.h"
#include "Logger.h"
#include <shlobj.h>
#include <iostream>

using namespace std;

// ============================================================================
// 构造和析构
// ============================================================================
CarvedResultsCache::CarvedResultsCache()
    : isValid(false), driveLetter(0), totalResults(0) {
}

CarvedResultsCache::~CarvedResultsCache() {
}

// ============================================================================
// 生成缓存文件路径
// ============================================================================
string CarvedResultsCache::GenerateCachePath(char drive) {
    char tempPath[MAX_PATH];
    GetTempPathA(MAX_PATH, tempPath);

    string cachePath = tempPath;
    cachePath += "filerestore_carved_";
    cachePath += drive;
    cachePath += ".cache";

    return cachePath;
}

// ============================================================================
// 序列化单个 CarvedFileInfo
// ============================================================================
void CarvedResultsCache::SerializeFileInfo(ofstream& out, const CarvedFileInfo& info) {
    // 写入基本字段
    out.write((const char*)&info.startLCN, sizeof(info.startLCN));
    out.write((const char*)&info.startOffset, sizeof(info.startOffset));
    out.write((const char*)&info.fileSize, sizeof(info.fileSize));
    out.write((const char*)&info.hasValidFooter, sizeof(info.hasValidFooter));
    out.write((const char*)&info.confidence, sizeof(info.confidence));

    // 写入字符串（长度 + 数据）
    DWORD extLen = (DWORD)info.extension.length();
    out.write((const char*)&extLen, sizeof(extLen));
    if (extLen > 0) {
        out.write(info.extension.c_str(), extLen);
    }

    DWORD descLen = (DWORD)info.description.length();
    out.write((const char*)&descLen, sizeof(descLen));
    if (descLen > 0) {
        out.write(info.description.c_str(), descLen);
    }

    // 写入时间戳信息
    out.write((const char*)&info.creationTime, sizeof(info.creationTime));
    out.write((const char*)&info.modificationTime, sizeof(info.modificationTime));
    out.write((const char*)&info.accessTime, sizeof(info.accessTime));
    out.write((const char*)&info.tsSource, sizeof(info.tsSource));
    out.write((const char*)&info.matchedMftRecord, sizeof(info.matchedMftRecord));

    DWORD embeddedLen = (DWORD)info.embeddedInfo.length();
    out.write((const char*)&embeddedLen, sizeof(embeddedLen));
    if (embeddedLen > 0) {
        out.write(info.embeddedInfo.c_str(), embeddedLen);
    }

    // 写入完整性信息
    out.write((const char*)&info.integrityScore, sizeof(info.integrityScore));
    out.write((const char*)&info.integrityValidated, sizeof(info.integrityValidated));

    DWORD diagLen = (DWORD)info.integrityDiagnosis.length();
    out.write((const char*)&diagLen, sizeof(diagLen));
    if (diagLen > 0) {
        out.write(info.integrityDiagnosis.c_str(), diagLen);
    }
}

// ============================================================================
// 反序列化单个 CarvedFileInfo
// ============================================================================
bool CarvedResultsCache::DeserializeFileInfo(ifstream& in, CarvedFileInfo& info) {
    // 读取基本字段
    in.read((char*)&info.startLCN, sizeof(info.startLCN));
    in.read((char*)&info.startOffset, sizeof(info.startOffset));
    in.read((char*)&info.fileSize, sizeof(info.fileSize));
    in.read((char*)&info.hasValidFooter, sizeof(info.hasValidFooter));
    in.read((char*)&info.confidence, sizeof(info.confidence));

    if (in.fail()) return false;

    // 读取扩展名
    DWORD extLen = 0;
    in.read((char*)&extLen, sizeof(extLen));
    if (extLen > 0 && extLen < 256) {
        char buffer[256];
        in.read(buffer, extLen);
        info.extension = string(buffer, extLen);
    }

    // 读取描述
    DWORD descLen = 0;
    in.read((char*)&descLen, sizeof(descLen));
    if (descLen > 0 && descLen < 1024) {
        char buffer[1024];
        in.read(buffer, descLen);
        info.description = string(buffer, descLen);
    }

    // 读取时间戳信息
    in.read((char*)&info.creationTime, sizeof(info.creationTime));
    in.read((char*)&info.modificationTime, sizeof(info.modificationTime));
    in.read((char*)&info.accessTime, sizeof(info.accessTime));
    in.read((char*)&info.tsSource, sizeof(info.tsSource));
    in.read((char*)&info.matchedMftRecord, sizeof(info.matchedMftRecord));

    DWORD embeddedLen = 0;
    in.read((char*)&embeddedLen, sizeof(embeddedLen));
    if (embeddedLen > 0 && embeddedLen < 2048) {
        char buffer[2048];
        in.read(buffer, embeddedLen);
        info.embeddedInfo = string(buffer, embeddedLen);
    }

    // 读取完整性信息
    in.read((char*)&info.integrityScore, sizeof(info.integrityScore));
    in.read((char*)&info.integrityValidated, sizeof(info.integrityValidated));

    DWORD diagLen = 0;
    in.read((char*)&diagLen, sizeof(diagLen));
    if (diagLen > 0 && diagLen < 1024) {
        char buffer[1024];
        in.read(buffer, diagLen);
        info.integrityDiagnosis = string(buffer, diagLen);
    }

    return !in.fail();
}

// ============================================================================
// 保存结果到缓存文件
// ============================================================================
bool CarvedResultsCache::SaveResults(const vector<CarvedFileInfo>& results, char drive) {
    cacheFilePath = GenerateCachePath(drive);
    driveLetter = drive;
    totalResults = results.size();

    LOG_INFO_FMT("Saving %llu carved results to cache: %s", totalResults, cacheFilePath.c_str());

    ofstream out(cacheFilePath, ios::binary);
    if (!out.is_open()) {
        LOG_ERROR_FMT("Failed to create cache file: %s", cacheFilePath.c_str());
        return false;
    }

    try {
        // 写入文件头
        CarvedCacheHeader header;
        header.magic = CARVED_CACHE_MAGIC;
        header.version = CARVED_CACHE_VERSION;
        header.fileCount = totalResults;
        header.totalSize = 0;  // 稍后计算
        header.driveLetter = drive;
        memset(header.padding, 0, sizeof(header.padding));

        out.write((const char*)&header, sizeof(header));

        // 写入所有结果
        for (const auto& info : results) {
            SerializeFileInfo(out, info);
        }

        // 更新总大小
        header.totalSize = out.tellp();
        out.seekp(0);
        out.write((const char*)&header, sizeof(header));

        out.close();

        isValid = true;
        LOG_INFO_FMT("Cache saved successfully: %llu bytes, %llu files",
                     header.totalSize, totalResults);
        return true;
    }
    catch (const exception& e) {
        LOG_ERROR_FMT("Exception while saving cache: %s", e.what());
        out.close();
        DeleteFileA(cacheFilePath.c_str());
        return false;
    }
}

// ============================================================================
// 加载所有结果
// ============================================================================
bool CarvedResultsCache::LoadAllResults(vector<CarvedFileInfo>& results, char& outDrive) {
    if (cacheFilePath.empty()) {
        LOG_ERROR("Cache file path not set");
        return false;
    }

    ifstream in(cacheFilePath, ios::binary);
    if (!in.is_open()) {
        LOG_ERROR_FMT("Failed to open cache file: %s", cacheFilePath.c_str());
        return false;
    }

    try {
        // 读取文件头
        CarvedCacheHeader header;
        in.read((char*)&header, sizeof(header));

        if (header.magic != CARVED_CACHE_MAGIC) {
            LOG_ERROR("Invalid cache file magic");
            in.close();
            return false;
        }

        if (header.version != CARVED_CACHE_VERSION) {
            LOG_WARNING_FMT("Cache version mismatch: %u (expected %u)",
                           header.version, CARVED_CACHE_VERSION);
        }

        outDrive = header.driveLetter;
        driveLetter = header.driveLetter;
        totalResults = header.fileCount;

        LOG_INFO_FMT("Loading %llu results from cache", totalResults);

        results.clear();
        results.reserve((size_t)totalResults);

        // 读取所有结果
        for (ULONGLONG i = 0; i < totalResults; i++) {
            CarvedFileInfo info;
            if (DeserializeFileInfo(in, info)) {
                results.push_back(info);
            } else {
                LOG_WARNING_FMT("Failed to deserialize file info #%llu", i);
                break;
            }
        }

        in.close();

        isValid = true;
        LOG_INFO_FMT("Loaded %zu files from cache", results.size());
        return true;
    }
    catch (const exception& e) {
        LOG_ERROR_FMT("Exception while loading cache: %s", e.what());
        in.close();
        return false;
    }
}

// ============================================================================
// 加载指定范围的结果（分页）
// ============================================================================
bool CarvedResultsCache::LoadResultRange(vector<CarvedFileInfo>& results,
                                          size_t startIndex, size_t count,
                                          char& outDrive) {
    if (cacheFilePath.empty()) {
        LOG_ERROR("Cache file path not set");
        return false;
    }

    ifstream in(cacheFilePath, ios::binary);
    if (!in.is_open()) {
        LOG_ERROR_FMT("Failed to open cache file: %s", cacheFilePath.c_str());
        return false;
    }

    try {
        // 读取文件头
        CarvedCacheHeader header;
        in.read((char*)&header, sizeof(header));

        if (header.magic != CARVED_CACHE_MAGIC) {
            LOG_ERROR("Invalid cache file magic");
            in.close();
            return false;
        }

        outDrive = header.driveLetter;
        driveLetter = header.driveLetter;
        totalResults = header.fileCount;

        if (startIndex >= totalResults) {
            LOG_WARNING_FMT("Start index %zu exceeds total results %llu",
                           startIndex, totalResults);
            in.close();
            return false;
        }

        // 跳过前面的记录（需要逐个读取，因为记录大小不固定）
        for (size_t i = 0; i < startIndex; i++) {
            CarvedFileInfo dummy;
            if (!DeserializeFileInfo(in, dummy)) {
                LOG_ERROR_FMT("Failed to skip record #%zu", i);
                in.close();
                return false;
            }
        }

        // 读取指定数量的记录
        results.clear();
        results.reserve(count);

        size_t endIndex = min(startIndex + count, (size_t)totalResults);
        for (size_t i = startIndex; i < endIndex; i++) {
            CarvedFileInfo info;
            if (DeserializeFileInfo(in, info)) {
                results.push_back(info);
            } else {
                LOG_WARNING_FMT("Failed to deserialize file info #%zu", i);
                break;
            }
        }

        in.close();

        LOG_DEBUG_FMT("Loaded page: %zu files (range %zu-%zu)",
                     results.size(), startIndex, startIndex + results.size());
        return true;
    }
    catch (const exception& e) {
        LOG_ERROR_FMT("Exception while loading cache range: %s", e.what());
        in.close();
        return false;
    }
}

// ============================================================================
// 获取缓存文件信息
// ============================================================================
bool CarvedResultsCache::GetCacheInfo(ULONGLONG& outCount, char& outDrive) {
    if (cacheFilePath.empty()) {
        return false;
    }

    ifstream in(cacheFilePath, ios::binary);
    if (!in.is_open()) {
        return false;
    }

    CarvedCacheHeader header;
    in.read((char*)&header, sizeof(header));
    in.close();

    if (header.magic != CARVED_CACHE_MAGIC) {
        return false;
    }

    outCount = header.fileCount;
    outDrive = header.driveLetter;
    totalResults = header.fileCount;
    driveLetter = header.driveLetter;
    isValid = true;

    return true;
}

// ============================================================================
// 清除缓存
// ============================================================================
void CarvedResultsCache::ClearCache() {
    if (!cacheFilePath.empty()) {
        DeleteFileA(cacheFilePath.c_str());
        LOG_INFO_FMT("Cache file deleted: %s", cacheFilePath.c_str());
    }
    isValid = false;
    totalResults = 0;
}

// ============================================================================
// 静态方法：检查是否存在有效缓存
// ============================================================================
bool CarvedResultsCache::HasValidCache(char drive) {
    string cachePath = GenerateCachePath(drive);

    ifstream in(cachePath, ios::binary);
    if (!in.is_open()) {
        return false;
    }

    CarvedCacheHeader header;
    in.read((char*)&header, sizeof(header));
    in.close();

    return (header.magic == CARVED_CACHE_MAGIC);
}

// ============================================================================
// 静态方法：获取缓存大小
// ============================================================================
ULONGLONG CarvedResultsCache::GetCacheSize(char drive) {
    string cachePath = GenerateCachePath(drive);

    WIN32_FILE_ATTRIBUTE_DATA fileInfo;
    if (!GetFileAttributesExA(cachePath.c_str(), GetFileExInfoStandard, &fileInfo)) {
        return 0;
    }

    LARGE_INTEGER size;
    size.LowPart = fileInfo.nFileSizeLow;
    size.HighPart = fileInfo.nFileSizeHigh;

    return size.QuadPart;
}
