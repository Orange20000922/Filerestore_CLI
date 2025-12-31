#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <functional>

// USN Journal 删除文件信息
struct UsnDeletedFileInfo {
    DWORDLONG FileReferenceNumber;      // 文件的 MFT 记录号
    DWORDLONG ParentFileReferenceNumber; // 父目录的 MFT 记录号
    std::wstring FileName;               // 文件名
    LARGE_INTEGER TimeStamp;             // 删除时间戳
    DWORD Reason;                        // USN 原因标志
    DWORD FileAttributes;                // 文件属性
    USN Usn;                             // USN 号
};

// USN Journal 统计信息
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

    // 打开指定驱动器的 USN Journal
    bool Open(char driveLetter);

    // 关闭
    void Close();

    // 检查是否已打开
    bool IsOpen() const { return hVolume != INVALID_HANDLE_VALUE; }

    // 获取 USN Journal 统计信息
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
