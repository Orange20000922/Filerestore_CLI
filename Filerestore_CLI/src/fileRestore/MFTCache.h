#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <fstream>
#include "CarvedFileTypes.h"

using namespace std;

// MFT 缓存魔术字节
constexpr DWORD MFT_CACHE_MAGIC = 0x4D465443;  // "MFTC"
constexpr DWORD MFT_CACHE_VERSION = 2;

// ============================================================================
// MFT 缓存条目 - 轻量级，只保留恢复必需的信息
// ============================================================================
struct MFTCacheEntry {
    ULONGLONG recordNumber;         // MFT 记录号
    ULONGLONG fileSize;             // 文件大小
    ULONGLONG startLCN;             // 起始 LCN（第一个数据运行）
    ULONGLONG totalClusters;        // 总簇数
    FILETIME creationTime;          // 创建时间
    FILETIME modificationTime;      // 修改时间
    wstring fileName;               // 文件名（不含路径）
    wstring extension;              // 扩展名
    ULONGLONG parentRecord;         // 父目录记录号
    bool isDeleted;                 // 是否已删除
    bool isDirectory;               // 是否是目录
    bool isResident;                // 数据是否驻留在 MFT 中
    WORD sequenceNumber;            // 序列号（用于验证）

    MFTCacheEntry() :
        recordNumber(0), fileSize(0), startLCN(0), totalClusters(0),
        parentRecord(0), isDeleted(false), isDirectory(false),
        isResident(false), sequenceNumber(0) {
        ZeroMemory(&creationTime, sizeof(FILETIME));
        ZeroMemory(&modificationTime, sizeof(FILETIME));
    }
};

// ============================================================================
// MFT 缓存文件头
// ============================================================================
struct MFTCacheHeader {
    DWORD magic;                    // 魔术字节
    DWORD version;                  // 版本号
    ULONGLONG totalRecords;         // 总记录数
    ULONGLONG deletedCount;         // 已删除文件数
    ULONGLONG activeCount;          // 活动文件数
    char driveLetter;               // 驱动器字母
    char padding[7];                // 对齐
    FILETIME buildTime;             // 构建时间
    ULONGLONG bytesPerCluster;      // 每簇字节数
};

// ============================================================================
// MFT 缓存管理器
// 功能：
// 1. 一次性解析 MFT，构建轻量级缓存
// 2. 持久化到磁盘，避免重复解析
// 3. 提供快速的 LCN -> 记录 和 记录号 -> 信息 查找
// ============================================================================
class MFTCache {
private:
    // 缓存数据
    unordered_map<ULONGLONG, MFTCacheEntry> entriesByRecord;  // 按记录号索引
    map<ULONGLONG, ULONGLONG> lcnToRecord;                     // LCN -> 记录号映射

    // 元信息
    char driveLetter;
    ULONGLONG bytesPerCluster;
    FILETIME buildTime;
    bool isValid;
    string cacheFilePath;

    // 序列化
    void SerializeEntry(ofstream& out, const MFTCacheEntry& entry);
    bool DeserializeEntry(ifstream& in, MFTCacheEntry& entry);

    // 生成缓存路径
    static string GenerateCachePath(char drive);

public:
    MFTCache();
    ~MFTCache();

    // ========== 构建缓存 ==========

    // 从 MFT 构建缓存（核心方法）
    // includeActive: 是否包含活动文件（用于删除状态检查）
    bool BuildFromMFT(char drive, bool includeActive = true, bool showProgress = true);

    // ========== 持久化 ==========

    // 保存到磁盘
    bool SaveToFile(const string& path = "");

    // 从磁盘加载
    bool LoadFromFile(char drive);
    bool LoadFromFile(const string& path);

    // ========== 查询接口 ==========

    // 按记录号查找
    const MFTCacheEntry* GetByRecordNumber(ULONGLONG recordNum) const;

    // 按 LCN 查找（支持范围匹配）
    const MFTCacheEntry* GetByLCN(ULONGLONG lcn) const;
    vector<const MFTCacheEntry*> GetByLCNRange(ULONGLONG startLCN, ULONGLONG endLCN) const;

    // 获取所有已删除文件
    vector<const MFTCacheEntry*> GetDeletedFiles() const;

    // 按扩展名筛选
    vector<const MFTCacheEntry*> FilterByExtension(const wstring& ext) const;

    // 按文件名模糊匹配
    vector<const MFTCacheEntry*> SearchByName(const wstring& pattern) const;

    // ========== 与签名扫描结果关联 ==========

    // 为 CarvedFileInfo 填充 MFT 信息
    bool EnrichCarvedInfo(CarvedFileInfo& carved) const;

    // 批量填充
    size_t EnrichCarvedInfoBatch(vector<CarvedFileInfo>& carved) const;

    // 检查 LCN 是否属于活动文件
    bool IsLCNActive(ULONGLONG lcn) const;

    // ========== 状态查询 ==========

    bool IsValid() const { return isValid; }
    char GetDriveLetter() const { return driveLetter; }
    size_t GetTotalCount() const { return entriesByRecord.size(); }
    size_t GetDeletedCount() const;
    size_t GetActiveCount() const;
    ULONGLONG GetBytesPerCluster() const { return bytesPerCluster; }
    const string& GetCachePath() const { return cacheFilePath; }

    // ========== 缓存有效性检查 ==========

    // 检查指定驱动器是否有有效缓存
    static bool HasValidCache(char drive, int maxAgeMinutes = 60);

    // 获取缓存文件大小
    static ULONGLONG GetCacheFileSize(char drive);

    // 清除缓存
    void Clear();
};

// ============================================================================
// 全局 MFT 缓存单例（可选，用于跨命令共享）
// ============================================================================
class MFTCacheManager {
private:
    static unique_ptr<MFTCache> globalCache;
    static char cachedDrive;

public:
    // 获取或构建缓存
    static MFTCache* GetCache(char drive, bool forceRebuild = false);

    // 释放缓存
    static void ReleaseCache();

    // 检查缓存状态
    static bool IsCacheReady(char drive);
};
