#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <algorithm>

using namespace std;

// ============================================================================
// MFT 快照魔术字节和版本
// ============================================================================
constexpr DWORD MFT_SNAPSHOT_MAGIC = 0x4D465453;  // "MFTS"
constexpr DWORD MFT_SNAPSHOT_VERSION = 1;

// ============================================================================
// MFT 快照条目 - 记录文件删除时刻的完整 MFT 元数据
// ============================================================================
struct MFTSnapshot {
    ULONGLONG recordNumber;                           // MFT 记录号
    WORD sequenceNumber;                              // 删除时的序列号
    wstring fileName;                                 // 文件名
    ULONGLONG fileSize;                               // 文件大小
    vector<pair<ULONGLONG, ULONGLONG>> dataRuns;      // 完整 LCN 映射 (LCN, clusterCount)
    FILETIME captureTime;                             // 快照捕获时间
    FILETIME deleteTime;                              // 删除时间（来自 USN）
    bool isResident;                                  // 是否常驻数据
    vector<BYTE> residentData;                        // 常驻数据内容
    ULONGLONG parentRecord;                           // 父目录记录号

    MFTSnapshot() :
        recordNumber(0), sequenceNumber(0), fileSize(0),
        isResident(false), parentRecord(0) {
        ZeroMemory(&captureTime, sizeof(FILETIME));
        ZeroMemory(&deleteTime, sizeof(FILETIME));
    }
};

// ============================================================================
// MFT 快照存储文件头
// ============================================================================
struct MFTSnapshotHeader {
    DWORD magic;
    DWORD version;
    ULONGLONG snapshotCount;
    char driveLetter;
    char padding[7];
    FILETIME createTime;       // 存储创建时间
    FILETIME lastUpdateTime;   // 最后更新时间
};

// ============================================================================
// MFT 快照存储 - 线程安全的增量式快照管理
// ============================================================================
class MFTSnapshotStore {
public:
    MFTSnapshotStore();
    ~MFTSnapshotStore();

    // ========== 快照管理 ==========

    // 添加快照（线程安全）
    void AddSnapshot(const MFTSnapshot& snapshot);

    // 按 MFT 记录号 + 序列号精确查询
    // 返回 nullptr 如果未找到
    const MFTSnapshot* FindByRecord(ULONGLONG recordNum, WORD seqNum) const;

    // 按 MFT 记录号查找最近的快照（不限序列号）
    const MFTSnapshot* FindLatestByRecord(ULONGLONG recordNum) const;

    // 按文件名模糊查询
    vector<const MFTSnapshot*> SearchByName(const wstring& pattern) const;

    // ========== 持久化 ==========

    // 保存到文件
    bool SaveToFile(const string& path);

    // 从文件加载
    bool LoadFromFile(const string& path);

    // 生成默认存储路径
    static string GenerateStorePath(char drive);

    // ========== 清理和统计 ==========

    // 清理超过指定天数的快照，返回删除数量
    size_t PurgeOlderThan(int days);

    // 获取快照总数
    size_t GetCount() const;

    // 清空所有快照
    void Clear();

    // ========== 驱动器信息 ==========

    void SetDriveLetter(char drive) { driveLetter = drive; }
    char GetDriveLetter() const { return driveLetter; }

private:
    // 按记录号索引（key = recordNumber, value = 该记录的快照列表，按序列号排序）
    unordered_map<ULONGLONG, vector<MFTSnapshot>> snapshots;
    mutable mutex mtx;
    char driveLetter;

    // 序列化单条快照
    void SerializeSnapshot(ofstream& out, const MFTSnapshot& snap);
    bool DeserializeSnapshot(ifstream& in, MFTSnapshot& snap);
};
