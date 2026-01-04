#pragma once
#include <Windows.h>
#include <vector>
#include <string>
#include "MFTStructures.h"
#include "MFTReader.h"
#include "MFTParser.h"
#include "PathResolver.h"
#include "MFTBatchReader.h"
#include "ProgressBar.h"

using namespace std;

// 低价值文件过滤级别
enum FilterLevel {
    FILTER_NONE,        // 不过滤，全部重建路径（最慢，最完整）
    FILTER_SKIP_PATH,   // 跳过低价值文件的路径重建，使用占位符（默认，平衡性能和完整性）
    FILTER_EXCLUDE      // 完全排除低价值文件（最快，结果最少）
};

// 已删除文件扫描器类 - 负责扫描和定位已删除文件
class DeletedFileScanner
{
private:
    MFTReader* reader;
    MFTParser* parser;
    PathResolver* pathResolver;
    MFTBatchReader* batchReader;    // 批量读取器（可选，用于高性能扫描）
    bool useBatchReading;           // 是否使用批量读取
    FilterLevel filterLevel;        // 低价值文件过滤级别

public:
    DeletedFileScanner(MFTReader* mftReader, MFTParser* mftParser, PathResolver* resolver);
    ~DeletedFileScanner();

    // 扫描已删除文件（自动选择最优方式）
    vector<DeletedFileInfo> ScanDeletedFiles(ULONGLONG maxRecords = 10000);

    // 扫描已删除文件（使用批量缓冲读取，高性能）
    vector<DeletedFileInfo> ScanDeletedFilesBatch(ULONGLONG maxRecords = 0);

    // 启用/禁用批量读取
    void SetUseBatchReading(bool enable) { useBatchReading = enable; }

    // 设置/获取过滤级别
    void SetFilterLevel(FilterLevel level) { filterLevel = level; }
    FilterLevel GetFilterLevel() const { return filterLevel; }

    // 静态辅助函数：筛选特定类型的文件
    static vector<DeletedFileInfo> FilterByExtension(const vector<DeletedFileInfo>& files, const wstring& extension);
    static vector<DeletedFileInfo> FilterBySize(const vector<DeletedFileInfo>& files, ULONGLONG minSize, ULONGLONG maxSize);
    static vector<DeletedFileInfo> FilterByName(const vector<DeletedFileInfo>& files, const wstring& namePattern);
    static vector<DeletedFileInfo> FilterUserFiles(const vector<DeletedFileInfo>& files);  // 过滤出用户文件

    // 缓存管理函数
    static bool SaveToCache(const vector<DeletedFileInfo>& files, char driveLetter);
    static bool LoadFromCache(vector<DeletedFileInfo>& files, char driveLetter);
    static bool IsCacheValid(char driveLetter, int maxAgeMinutes = 60);
    static string GetCachePath(char driveLetter);

    // 路径重建过滤（性能优化）
    static bool ShouldSkipPathReconstruction(const wstring& fileName);
    static wstring GetFileExtension(const wstring& fileName);
};
