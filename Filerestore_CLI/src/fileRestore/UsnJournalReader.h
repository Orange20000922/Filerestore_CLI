#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <functional>

// FileReferenceNumber 格式说明:
// ┌──────────────────┬────────────────────────────────────────────┐
// │  Sequence (16位) │           MFT Record Index (48位)          │
// └──────────────────┴────────────────────────────────────────────┘
// 使用 ExtractMftRecordNumber() 提取真实的 MFT 记录号

// 从 FileReferenceNumber 提取 MFT 记录号 (低48位)
inline ULONGLONG ExtractMftRecordNumber(DWORDLONG fileRefNumber) {
    return fileRefNumber & 0x0000FFFFFFFFFFFFULL;
}

// 从 FileReferenceNumber 提取序列号 (高16位)
inline WORD ExtractSequenceNumber(DWORDLONG fileRefNumber) {
    return (WORD)(fileRefNumber >> 48);
}

// USN 日志删除文件信息结构
struct UsnDeletedFileInfo {
    DWORDLONG FileReferenceNumber;      // 文件引用号 (包含序列号+MFT记录号)
    DWORDLONG ParentFileReferenceNumber; // 父目录引用号
    std::wstring FileName;               // 文件名
    LARGE_INTEGER TimeStamp;             // 删除时间戳
    DWORD Reason;                        // USN 原因标志
    DWORD FileAttributes;                // 文件属性
    USN Usn;                             // USN 号

    // ==================== MFT 增强信息（可选，需要调用 EnrichWithMFT 填充）====================
    ULONGLONG FileSize;                  // 文件大小（从 MFT 获取，0 表示未查询或不可用）
    FILETIME MftCreationTime;            // 创建时间（从 MFT 获取）
    FILETIME MftModificationTime;        // 修改时间（从 MFT 获取）
    bool MftInfoValid;                   // MFT 信息是否有效
    bool MftRecordReused;                // MFT 记录是否已被复用（序列号不匹配）

    // 获取真实的 MFT 记录号
    ULONGLONG GetMftRecordNumber() const {
        return ExtractMftRecordNumber(FileReferenceNumber);
    }

    // 获取期望的序列号（来自 USN）
    WORD GetExpectedSequence() const {
        return ExtractSequenceNumber(FileReferenceNumber);
    }

    // 获取父目录的 MFT 记录号
    ULONGLONG GetParentMftRecordNumber() const {
        return ExtractMftRecordNumber(ParentFileReferenceNumber);
    }

    // 默认构造函数
    UsnDeletedFileInfo() : FileReferenceNumber(0), ParentFileReferenceNumber(0),
                           Reason(0), FileAttributes(0), Usn(0),
                           FileSize(0), MftInfoValid(false), MftRecordReused(false) {
        TimeStamp.QuadPart = 0;
        MftCreationTime.dwLowDateTime = 0;
        MftCreationTime.dwHighDateTime = 0;
        MftModificationTime.dwLowDateTime = 0;
        MftModificationTime.dwHighDateTime = 0;
    }
};

// USN 日志统计信息结构
struct UsnJournalStats {
    USN FirstUsn;
    USN NextUsn;
    DWORDLONG UsnJournalID;
    DWORDLONG MaximumSize;
    DWORDLONG AllocationDelta;
};

class UsnJournalReader {
public:
    UsnJournalReader();
    ~UsnJournalReader();

    // 打开指定驱动器的 USN 日志
    bool Open(char driveLetter);

    // 关闭
    void Close();

    // 检查是否已打开
    bool IsOpen() const { return hVolume != INVALID_HANDLE_VALUE; }

    // 获取 USN 日志统计信息
    // forceRefresh: true = 强制刷新获取实时数据, false = 使用缓存数据（如果可用）
    bool GetJournalStats(UsnJournalStats& stats, bool forceRefresh = false);

    // 扫描最近删除的文件
    // maxTimeSeconds: 最大回溯时间（秒），0表示不限制
    // maxRecords: 最大返回记录数，0表示不限制
    std::vector<UsnDeletedFileInfo> ScanRecentlyDeletedFiles(
        int maxTimeSeconds = 3600,  // 默认1小时
        size_t maxRecords = 10000   // 默认最多10000条
    );

    // 使用回调函数扫描（适合大量数据）
    bool ScanWithCallback(
        std::function<bool(const UsnDeletedFileInfo&)> callback,
        int maxTimeSeconds = 3600
    );

    // 根据文件名搜索删除记录
    std::vector<UsnDeletedFileInfo> SearchDeletedByName(
        const std::wstring& fileName,
        bool exactMatch = false
    );

    // 获取驱动器字母
    char GetDriveLetter() const { return driveLetter; }

    // 获取错误信息
    std::string GetLastError() const { return lastError; }

private:
    HANDLE hVolume;
    char driveLetter;
    std::string lastError;
    USN_JOURNAL_DATA_V0 journalData;
    bool journalDataValid;

    // 内部方法
    bool QueryJournalData();
    bool IsDeleteReason(DWORD reason) const;
    FILETIME GetCurrentFileTime() const;
    bool IsWithinTimeRange(const LARGE_INTEGER& timestamp, int maxSeconds) const;
};
