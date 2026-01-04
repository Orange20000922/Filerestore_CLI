#pragma once
#include <Windows.h>
#include <vector>
#include "MFTStructures.h"

using namespace std;

// NTFS引导扇区结构
#pragma pack(push, 1)
typedef struct _NTFS_BOOT_SECTOR {
    BYTE Jump[3];                       // 0x00: 跳转指令
    BYTE OemId[8];                      // 0x03: "NTFS    "
    WORD BytesPerSector;                // 0x0B: 每扇区字节数
    BYTE SectorsPerCluster;             // 0x0D: 每簇扇区数
    BYTE Reserved1[7];                  // 0x0E: 保留
    BYTE MediaDescriptor;               // 0x15: 媒体描述符
    WORD Reserved2;                     // 0x16: 保留
    WORD SectorsPerTrack;               // 0x18: 每磁道扇区数
    WORD NumberOfHeads;                 // 0x1A: 磁头数
    DWORD HiddenSectors;                // 0x1C: 隐藏扇区数
    DWORD Reserved3;                    // 0x20: 保留
    DWORD Reserved4;                    // 0x24: 保留
    ULONGLONG TotalSectors;             // 0x28: 总扇区数
    ULONGLONG MftStartLCN;              // 0x30: MFT起始簇号
    ULONGLONG Mft2StartLCN;             // 0x38: MFT镜像起始簇号
    CHAR ClustersPerFileRecord;         // 0x40: 每个文件记录的簇数（有符号字节！）
    BYTE Reserved5[3];                  // 0x41: 保留（填充）
    CHAR ClustersPerIndexBlock;         // 0x44: 每个索引块的簇数（有符号字节！）
    BYTE Reserved6[3];                  // 0x45: 保留（填充）
    ULONGLONG VolumeSerialNumber;       // 0x48: 卷序列号
    DWORD Checksum;                     // 0x50: 校验和
    BYTE BootCode[426];                 // 0x54: 引导代码
    WORD EndMarker;                     // 0x1FE: 结束标记 (0xAA55)
} NTFS_BOOT_SECTOR, *PNTFS_BOOT_SECTOR;
#pragma pack(pop)

// MFT 读取器类 - 负责底层磁盘和 MFT 访问
class MFTReader
{
private:
    HANDLE hVolume;
    DWORD bytesPerSector;
    DWORD sectorsPerCluster;
    ULONGLONG mftStartLCN;
    DWORD bytesPerFileRecord;
    ULONGLONG totalClusters;            // 卷总簇数

    // MFT的data runs（MFT在磁盘上的实际分布）
    vector<pair<ULONGLONG, ULONGLONG>> mftDataRuns;  // pair<LCN, clusterCount>
    bool mftDataRunsLoaded;

    // 根据MFT记录号查找对应的实际LCN
    bool GetMFTRecordLCN(ULONGLONG recordNumber, ULONGLONG& lcn, ULONGLONG& offsetInCluster);

    // 简单的data runs解析（仅用于MFT记录#0）
    bool ParseMFTDataRuns(BYTE* dataRun);

public:
    MFTReader();
    ~MFTReader();

    // 卷操作
    bool OpenVolume(char driveLetter);
    void CloseVolume();
    bool IsVolumeOpen() const { return hVolume != INVALID_HANDLE_VALUE; }

    // 基础读取操作
    bool ReadClusters(ULONGLONG startLCN, ULONGLONG clusterCount, vector<BYTE>& buffer);
    bool ReadMFT(ULONGLONG fileRecordNumber, vector<BYTE>& record);

    // 批量读取 MFT 记录（性能优化）
    bool ReadMFTBatch(ULONGLONG startRecordNumber, ULONGLONG recordCount, vector<vector<BYTE>>& records);

    // MFT 信息
    ULONGLONG GetTotalMFTRecords();

    // 诊断：检测MFT是否碎片化
    void DiagnoseMFTFragmentation();

    // 获取参数
    DWORD GetBytesPerSector() const { return bytesPerSector; }
    DWORD GetSectorsPerCluster() const { return sectorsPerCluster; }
    DWORD GetBytesPerFileRecord() const { return bytesPerFileRecord; }
    ULONGLONG GetMftStartLCN() const { return mftStartLCN; }
    ULONGLONG GetTotalClusters() const { return totalClusters; }
    ULONGLONG GetBytesPerCluster() const { return (ULONGLONG)bytesPerSector * sectorsPerCluster; }
    HANDLE GetVolumeHandle() const { return hVolume; }
};
