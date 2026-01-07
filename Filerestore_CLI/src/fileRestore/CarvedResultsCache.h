#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <fstream>
#include "CarvedFileTypes.h"

using namespace std;

// 序列化的魔术字节标识
constexpr DWORD CARVED_CACHE_MAGIC = 0x43525645;  // "CRVE"
constexpr DWORD CARVED_CACHE_VERSION = 1;

// 缓存文件头
struct CarvedCacheHeader {
    DWORD magic;            // 魔术字节
    DWORD version;          // 版本号
    ULONGLONG fileCount;    // 文件数量
    ULONGLONG totalSize;    // 总大小（字节）
    char driveLetter;       // 驱动器字母
    char padding[7];        // 对齐
};

// Carved 文件结果持久化缓存
// 功能：将扫描结果保存到磁盘，避免内存占用过高
class CarvedResultsCache {
private:
    string cacheFilePath;
    bool isValid;
    char driveLetter;
    ULONGLONG totalResults;

    // 生成缓存文件路径
    static string GenerateCachePath(char drive);

    // 序列化单个 CarvedFileInfo
    void SerializeFileInfo(ofstream& out, const CarvedFileInfo& info);

    // 反序列化单个 CarvedFileInfo
    bool DeserializeFileInfo(ifstream& in, CarvedFileInfo& info);

public:
    CarvedResultsCache();
    ~CarvedResultsCache();

    // 保存结果到缓存文件
    bool SaveResults(const vector<CarvedFileInfo>& results, char drive);

    // 加载所有结果（用于小数据集）
    bool LoadAllResults(vector<CarvedFileInfo>& results, char& outDrive);

    // 加载指定范围的结果（分页）
    bool LoadResultRange(vector<CarvedFileInfo>& results,
                         size_t startIndex, size_t count,
                         char& outDrive);

    // 获取缓存文件信息
    bool GetCacheInfo(ULONGLONG& outCount, char& outDrive);

    // 检查缓存是否有效
    bool IsValid() const { return isValid; }

    // 获取缓存文件路径
    const string& GetCachePath() const { return cacheFilePath; }

    // 清除缓存
    void ClearCache();

    // 静态方法：检查是否存在有效缓存
    static bool HasValidCache(char drive);

    // 静态方法：获取缓存大小
    static ULONGLONG GetCacheSize(char drive);

    // 从指定驱动器的缓存文件初始化（用于程序重启后恢复）
    bool InitFromDrive(char drive);

    // 查找并初始化任何存在的缓存文件
    bool InitFromAnyExistingCache();
};
