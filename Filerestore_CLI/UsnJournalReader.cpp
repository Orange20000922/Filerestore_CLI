#include "UsnJournalReader.h"
#include "Logger.h"
#include <iostream>
#include <sstream>
#include <algorithm>

using namespace std;

// USN_REASON 标志定义（如果未定义）
#ifndef USN_REASON_FILE_DELETE
#define USN_REASON_FILE_DELETE 0x00000200
#endif

#ifndef USN_REASON_RENAME_OLD_NAME
#define USN_REASON_RENAME_OLD_NAME 0x00001000
#endif

#ifndef USN_REASON_CLOSE
#define USN_REASON_CLOSE 0x80000000
#endif

UsnJournalReader::UsnJournalReader()
    : hVolume(INVALID_HANDLE_VALUE), driveLetter(0), journalDataValid(false) {
    ZeroMemory(&journalData, sizeof(journalData));
}

UsnJournalReader::~UsnJournalReader() {
    Close();
}

bool UsnJournalReader::Open(char driveLetter) {
    Close();  // 先关闭之前的

    this->driveLetter = toupper(driveLetter);

    // 构建卷路径
    wchar_t volumePath[16];
    swprintf_s(volumePath, L"\\\\.\\%c:", this->driveLetter);

    LOG_INFO_FMT("Opening volume for USN Journal: %c:", this->driveLetter);

    // 打开卷（需要管理员权限）
    hVolume = CreateFileW(
        volumePath,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hVolume == INVALID_HANDLE_VALUE) {
        DWORD error = ::GetLastError();
        stringstream ss;
        ss << "Failed to open volume " << this->driveLetter << ": (Error code: " << error << ")";
        lastError = ss.str();
        LOG_ERROR(lastError);

        if (error == ERROR_ACCESS_DENIED) {
            lastError += " - Administrator privileges required";
        }
        return false;
    }

    LOG_INFO("Volume opened successfully");

    // 查询 USN Journal 数据
    if (!QueryJournalData()) {
        Close();
        return false;
    }

    return true;
}

void UsnJournalReader::Close() {
    if (hVolume != INVALID_HANDLE_VALUE) {
        CloseHandle(hVolume);
        hVolume = INVALID_HANDLE_VALUE;
    }
    driveLetter = 0;
    journalDataValid = false;
}

bool UsnJournalReader::QueryJournalData() {
    DWORD bytesReturned;

    BOOL result = DeviceIoControl(
        hVolume,
        FSCTL_QUERY_USN_JOURNAL,
        NULL,
        0,
        &journalData,
        sizeof(journalData),
        &bytesReturned,
        NULL
    );

    if (!result) {
        DWORD error = ::GetLastError();
        stringstream ss;
        ss << "Failed to query USN Journal (Error code: " << error << ")";
        lastError = ss.str();
        LOG_ERROR(lastError);

        if (error == ERROR_JOURNAL_NOT_ACTIVE) {
            lastError += " - USN Journal is not active on this volume";
        }
        return false;
    }

    journalDataValid = true;

    LOG_INFO_FMT("USN Journal queried: ID=%llu, FirstUsn=%llu, NextUsn=%llu",
                 journalData.UsnJournalID, journalData.FirstUsn, journalData.NextUsn);

    return true;
}

bool UsnJournalReader::GetJournalStats(UsnJournalStats& stats, bool forceRefresh) {
    if (!IsOpen()) {
        lastError = "Volume not opened";
        LOG_ERROR(lastError);
        return false;
    }

    // 如果强制刷新或缓存无效，则重新查询
    if (forceRefresh || !journalDataValid) {
        if (!QueryJournalData()) {
            return false;
        }
    }

    stats.FirstUsn = journalData.FirstUsn;
    stats.NextUsn = journalData.NextUsn;
    stats.UsnJournalID = journalData.UsnJournalID;
    stats.MaximumSize = journalData.MaximumSize;
    stats.AllocationDelta = journalData.AllocationDelta;
    return true;
}

bool UsnJournalReader::IsDeleteReason(DWORD reason) const {
    // 检查是否包含删除相关的原因
    return (reason & USN_REASON_FILE_DELETE) != 0;
}

FILETIME UsnJournalReader::GetCurrentFileTime() const {
    FILETIME ft;
    SYSTEMTIME st;
    GetSystemTime(&st);
    SystemTimeToFileTime(&st, &ft);
    return ft;
}

bool UsnJournalReader::IsWithinTimeRange(const LARGE_INTEGER& timestamp, int maxSeconds) const {
    if (maxSeconds <= 0) {
        return true;  // 不限制时间
    }

    FILETIME currentFt = GetCurrentFileTime();
    ULARGE_INTEGER current, fileTime;
    current.LowPart = currentFt.dwLowDateTime;
    current.HighPart = currentFt.dwHighDateTime;
    fileTime.QuadPart = timestamp.QuadPart;

    // 计算时间差（100纳秒为单位）
    if (current.QuadPart < fileTime.QuadPart) {
        return true;  // 时间戳在未来？
    }

    ULONGLONG diffSeconds = (current.QuadPart - fileTime.QuadPart) / 10000000ULL;
    return diffSeconds <= (ULONGLONG)maxSeconds;
}

vector<UsnDeletedFileInfo> UsnJournalReader::ScanRecentlyDeletedFiles(
    int maxTimeSeconds, size_t maxRecords) {

    vector<UsnDeletedFileInfo> deletedFiles;

    if (!IsOpen()) {
        lastError = "Volume not opened";
        LOG_ERROR(lastError);
        return deletedFiles;
    }

    if (!journalDataValid) {
        lastError = "USN Journal data not available";
        LOG_ERROR(lastError);
        return deletedFiles;
    }

    LOG_INFO_FMT("Scanning USN Journal for deleted files (max time: %d seconds, max records: %zu)",
                 maxTimeSeconds, maxRecords);

    // 准备读取 USN Journal
    const DWORD bufferSize = 64 * 1024;  // 64KB 缓冲区
    vector<BYTE> buffer(bufferSize);

    READ_USN_JOURNAL_DATA_V0 readData;
    ZeroMemory(&readData, sizeof(readData));
    readData.StartUsn = journalData.FirstUsn;
    readData.ReasonMask = USN_REASON_FILE_DELETE | USN_REASON_RENAME_OLD_NAME;
    readData.ReturnOnlyOnClose = FALSE;
    readData.Timeout = 0;
    readData.BytesToWaitFor = 0;
    readData.UsnJournalID = journalData.UsnJournalID;

    DWORD bytesReturned;
    size_t totalRecordsScanned = 0;
    size_t deleteRecordsFound = 0;

    cout << "Scanning USN Journal..." << endl;

    while (true) {
        BOOL result = DeviceIoControl(
            hVolume,
            FSCTL_READ_USN_JOURNAL,
            &readData,
            sizeof(readData),
            buffer.data(),
            bufferSize,
            &bytesReturned,
            NULL
        );

        if (!result) {
            DWORD error = ::GetLastError();
            if (error == ERROR_HANDLE_EOF || error == ERROR_NO_MORE_ITEMS) {
                // 正常结束
                break;
            }
            LOG_WARNING_FMT("DeviceIoControl failed: %d", error);
            break;
        }

        if (bytesReturned <= sizeof(USN)) {
            break;  // 没有更多数据
        }

        // 更新下一个读取位置
        USN nextUsn = *(USN*)buffer.data();
        readData.StartUsn = nextUsn;

        // 解析 USN 记录
        BYTE* recordPtr = buffer.data() + sizeof(USN);
        BYTE* bufferEnd = buffer.data() + bytesReturned;

        while (recordPtr < bufferEnd) {
            USN_RECORD* record = (USN_RECORD*)recordPtr;

            if (record->RecordLength == 0) {
                break;
            }

            totalRecordsScanned++;

            // 检查是否是删除操作
            if (IsDeleteReason(record->Reason)) {
                // 检查时间范围
                if (IsWithinTimeRange(record->TimeStamp, maxTimeSeconds)) {
                    UsnDeletedFileInfo info;
                    info.FileReferenceNumber = record->FileReferenceNumber;
                    info.ParentFileReferenceNumber = record->ParentFileReferenceNumber;
                    info.TimeStamp = record->TimeStamp;
                    info.Reason = record->Reason;
                    info.FileAttributes = record->FileAttributes;
                    info.Usn = record->Usn;

                    // 提取文件名
                    wchar_t* fileName = (wchar_t*)((BYTE*)record + record->FileNameOffset);
                    int nameLength = record->FileNameLength / sizeof(wchar_t);
                    info.FileName = wstring(fileName, nameLength);

                    deletedFiles.push_back(info);
                    deleteRecordsFound++;

                    // 检查是否达到最大记录数
                    if (maxRecords > 0 && deletedFiles.size() >= maxRecords) {
                        LOG_INFO_FMT("Reached max records limit: %zu", maxRecords);
                        goto done;
                    }
                }
            }

            // 移动到下一条记录
            recordPtr += record->RecordLength;
        }

        // 显示进度
        if (totalRecordsScanned % 10000 == 0) {
            cout << "\r  Scanned: " << totalRecordsScanned << " records, found: "
                 << deleteRecordsFound << " deletions" << flush;
        }
    }

done:
    cout << "\r  Scan complete: " << totalRecordsScanned << " records scanned, "
         << deleteRecordsFound << " deletions found" << endl;

    LOG_INFO_FMT("USN Journal scan complete: %zu records scanned, %zu deletions found",
                 totalRecordsScanned, deleteRecordsFound);

    return deletedFiles;
}

bool UsnJournalReader::ScanWithCallback(
    function<bool(const UsnDeletedFileInfo&)> callback,
    int maxTimeSeconds) {

    if (!IsOpen() || !journalDataValid) {
        return false;
    }

    const DWORD bufferSize = 64 * 1024;
    vector<BYTE> buffer(bufferSize);

    READ_USN_JOURNAL_DATA_V0 readData;
    ZeroMemory(&readData, sizeof(readData));
    readData.StartUsn = journalData.FirstUsn;
    readData.ReasonMask = USN_REASON_FILE_DELETE | USN_REASON_RENAME_OLD_NAME;
    readData.ReturnOnlyOnClose = FALSE;
    readData.Timeout = 0;
    readData.BytesToWaitFor = 0;
    readData.UsnJournalID = journalData.UsnJournalID;

    DWORD bytesReturned;

    while (true) {
        BOOL result = DeviceIoControl(
            hVolume,
            FSCTL_READ_USN_JOURNAL,
            &readData,
            sizeof(readData),
            buffer.data(),
            bufferSize,
            &bytesReturned,
            NULL
        );

        if (!result || bytesReturned <= sizeof(USN)) {
            break;
        }

        USN nextUsn = *(USN*)buffer.data();
        readData.StartUsn = nextUsn;

        BYTE* recordPtr = buffer.data() + sizeof(USN);
        BYTE* bufferEnd = buffer.data() + bytesReturned;

        while (recordPtr < bufferEnd) {
            USN_RECORD* record = (USN_RECORD*)recordPtr;

            if (record->RecordLength == 0) {
                break;
            }

            if (IsDeleteReason(record->Reason)) {
                if (IsWithinTimeRange(record->TimeStamp, maxTimeSeconds)) {
                    UsnDeletedFileInfo info;
                    info.FileReferenceNumber = record->FileReferenceNumber;
                    info.ParentFileReferenceNumber = record->ParentFileReferenceNumber;
                    info.TimeStamp = record->TimeStamp;
                    info.Reason = record->Reason;
                    info.FileAttributes = record->FileAttributes;
                    info.Usn = record->Usn;

                    wchar_t* fileName = (wchar_t*)((BYTE*)record + record->FileNameOffset);
                    int nameLength = record->FileNameLength / sizeof(wchar_t);
                    info.FileName = wstring(fileName, nameLength);

                    // 调用回调，如果返回 false 则停止扫描
                    if (!callback(info)) {
                        return true;
                    }
                }
            }

            recordPtr += record->RecordLength;
        }
    }

    return true;
}

vector<UsnDeletedFileInfo> UsnJournalReader::SearchDeletedByName(
    const wstring& fileName, bool exactMatch) {

    vector<UsnDeletedFileInfo> results;

    // 转换为小写进行比较
    wstring searchName = fileName;
    transform(searchName.begin(), searchName.end(), searchName.begin(), ::towlower);

    ScanWithCallback([&](const UsnDeletedFileInfo& info) {
        wstring infoName = info.FileName;
        transform(infoName.begin(), infoName.end(), infoName.begin(), ::towlower);

        bool match = false;
        if (exactMatch) {
            match = (infoName == searchName);
        } else {
            match = (infoName.find(searchName) != wstring::npos);
        }

        if (match) {
            results.push_back(info);
        }

        return true;  // 继续扫描
    }, 0);  // 不限制时间

    return results;
}
