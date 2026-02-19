#include "UsnDeleteMonitor.h"
#include "Logger.h"
#include <iostream>

using namespace std;

// ============================================================================
// 构造和析构
// ============================================================================
UsnDeleteMonitor::UsnDeleteMonitor(char driveLetter)
    : driveLetter(driveLetter), running(false),
      lastProcessedUsn(0), pollIntervalMs(1000), autoSaveIntervalSec(60) {
    snapshotStore.SetDriveLetter(driveLetter);
}

UsnDeleteMonitor::~UsnDeleteMonitor() {
    Stop();
}

// ============================================================================
// 一次性扫描：为现有 USN 删除记录捕获快照
// ============================================================================
size_t UsnDeleteMonitor::CaptureExistingDeleted(int maxTimeHours) {
    MFTReader reader;
    if (!reader.OpenVolume(driveLetter)) {
        LOG_ERROR_FMT("CaptureExistingDeleted: 无法打开卷 %c:", driveLetter);
        return 0;
    }
    reader.GetTotalMFTRecords();

    MFTParser parser(&reader);

    UsnJournalReader usnReader;
    if (!usnReader.Open(driveLetter)) {
        LOG_ERROR_FMT("CaptureExistingDeleted: 无法打开 USN 日志 %c:", driveLetter);
        return 0;
    }

    int maxTimeSeconds = maxTimeHours * 3600;
    auto deletedFiles = usnReader.ScanRecentlyDeletedFiles(maxTimeSeconds, 0);

    size_t captured = 0;
    size_t missed = 0;
    size_t skipped = 0;

    for (const auto& info : deletedFiles) {
        // 跳过目录
        if (info.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            skipped++;
            continue;
        }

        ULONGLONG recordNum = info.GetMftRecordNumber();
        WORD expectedSeq = info.GetExpectedSequence();

        // 读取 MFT 记录
        vector<BYTE> recordData;
        if (!reader.ReadMFT(recordNum, recordData)) {
            missed++;
            continue;
        }

        // 提取文件数据信息
        auto dataInfo = parser.ExtractFileDataInfo(recordData.data(), recordData.size());
        if (!dataInfo) {
            missed++;
            continue;
        }

        // 检查序列号：如果已经被复用（序列号递增），仍然尝试捕获
        // 因为 data runs 可能仍然指向原文件数据（尤其是刚删除的情况下）
        WORD actualSeq = dataInfo->sequenceNumber;

        MFTSnapshot snapshot;
        snapshot.recordNumber = recordNum;
        snapshot.sequenceNumber = expectedSeq;  // 使用 USN 中的序列号
        snapshot.fileName = info.FileName;
        snapshot.fileSize = dataInfo->fileSize;
        snapshot.dataRuns = dataInfo->dataRuns;
        snapshot.isResident = dataInfo->isResident;
        snapshot.residentData = dataInfo->residentData;
        snapshot.parentRecord = info.GetParentMftRecordNumber();

        // 时间戳
        GetSystemTimeAsFileTime(&snapshot.captureTime);
        snapshot.deleteTime.dwLowDateTime = info.TimeStamp.LowPart;
        snapshot.deleteTime.dwHighDateTime = info.TimeStamp.HighPart;

        snapshotStore.AddSnapshot(snapshot);
        captured++;
    }

    stats.capturedCount += captured;
    stats.missedCount += missed;
    stats.skippedCount += skipped;
    stats.totalEvents += deletedFiles.size();

    LOG_INFO_FMT("CaptureExistingDeleted: %zu 个删除记录, 捕获 %zu, 失败 %zu, 跳过 %zu",
                 deletedFiles.size(), captured, missed, skipped);

    return captured;
}

// ============================================================================
// 处理单个删除事件
// ============================================================================
bool UsnDeleteMonitor::HandleDeleteEvent(const UsnDeletedFileInfo& info) {
    // 跳过目录
    if (info.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        stats.skippedCount++;
        return false;
    }

    stats.totalEvents++;

    ULONGLONG recordNum = info.GetMftRecordNumber();

    // 读取 MFT 记录（需要独立的 reader，因为在后台线程中）
    MFTReader reader;
    if (!reader.OpenVolume(driveLetter)) {
        stats.missedCount++;
        return false;
    }
    reader.GetTotalMFTRecords();

    MFTParser parser(&reader);

    vector<BYTE> recordData;
    if (!reader.ReadMFT(recordNum, recordData)) {
        stats.missedCount++;
        return false;
    }

    auto dataInfo = parser.ExtractFileDataInfo(recordData.data(), recordData.size());
    if (!dataInfo) {
        stats.missedCount++;
        if (eventCallback) {
            FILETIME ft;
            ft.dwLowDateTime = info.TimeStamp.LowPart;
            ft.dwHighDateTime = info.TimeStamp.HighPart;
            eventCallback(recordNum, 0, ft, info.FileName, false);
        }
        return false;
    }

    MFTSnapshot snapshot;
    snapshot.recordNumber = recordNum;
    snapshot.sequenceNumber = info.GetExpectedSequence();
    snapshot.fileName = info.FileName;
    snapshot.fileSize = dataInfo->fileSize;
    snapshot.dataRuns = dataInfo->dataRuns;
    snapshot.isResident = dataInfo->isResident;
    snapshot.residentData = dataInfo->residentData;
    snapshot.parentRecord = info.GetParentMftRecordNumber();

    GetSystemTimeAsFileTime(&snapshot.captureTime);
    snapshot.deleteTime.dwLowDateTime = info.TimeStamp.LowPart;
    snapshot.deleteTime.dwHighDateTime = info.TimeStamp.HighPart;

    snapshotStore.AddSnapshot(snapshot);
    stats.capturedCount++;

    if (eventCallback) {
        FILETIME ft;
        ft.dwLowDateTime = info.TimeStamp.LowPart;
        ft.dwHighDateTime = info.TimeStamp.HighPart;
        eventCallback(recordNum, dataInfo->fileSize, ft, info.FileName, true);
    }

    LOG_DEBUG_FMT("捕获快照: MFT#%llu %S (%.1fKB, %zu runs)",
                  recordNum, info.FileName.c_str(),
                  (double)dataInfo->fileSize / 1024.0,
                  dataInfo->dataRuns.size());

    return true;
}

// ============================================================================
// 后台监控线程
// ============================================================================
void UsnDeleteMonitor::MonitorThread() {
    LOG_INFO_FMT("USN 删除监控已启动: 驱动器 %c:, 轮询间隔 %ums", driveLetter, pollIntervalMs);

    UsnJournalReader usnReader;
    if (!usnReader.Open(driveLetter)) {
        LOG_ERROR_FMT("MonitorThread: 无法打开 USN 日志 %c:", driveLetter);
        running = false;
        return;
    }

    // 获取当前 USN 位置作为起点
    UsnJournalStats journalStats;
    if (usnReader.GetJournalStats(journalStats)) {
        lastProcessedUsn = journalStats.NextUsn;
    }

    DWORD lastSaveTime = GetTickCount();

    while (running.load()) {
        // 扫描新的删除事件
        // 使用短时间窗口（30秒）避免重复处理
        auto deleted = usnReader.ScanRecentlyDeletedFiles(30, 1000);

        for (const auto& info : deleted) {
            // 只处理 USN 位置在上次处理之后的记录
            if (info.Usn > lastProcessedUsn) {
                HandleDeleteEvent(info);
                if (info.Usn > lastProcessedUsn) {
                    lastProcessedUsn = info.Usn;
                }
            }
        }

        // 自动保存
        if (autoSaveIntervalSec > 0) {
            DWORD now = GetTickCount();
            if (now - lastSaveTime >= autoSaveIntervalSec * 1000) {
                string path = MFTSnapshotStore::GenerateStorePath(driveLetter);
                snapshotStore.SaveToFile(path);
                lastSaveTime = now;
            }
        }

        // 等待下一次轮询
        for (DWORD waited = 0; waited < pollIntervalMs && running.load(); waited += 100) {
            Sleep(100);
        }
    }

    // 退出前保存
    string path = MFTSnapshotStore::GenerateStorePath(driveLetter);
    snapshotStore.SaveToFile(path);

    LOG_INFO("USN 删除监控已停止");
}

// ============================================================================
// 启动和停止
// ============================================================================
bool UsnDeleteMonitor::Start() {
    if (running.load()) {
        return true;  // 已经在运行
    }

    // 尝试加载已有的快照
    string path = MFTSnapshotStore::GenerateStorePath(driveLetter);
    snapshotStore.LoadFromFile(path);

    running = true;
    monitorThread = thread(&UsnDeleteMonitor::MonitorThread, this);

    return true;
}

void UsnDeleteMonitor::Stop() {
    if (!running.load()) return;

    running = false;
    if (monitorThread.joinable()) {
        monitorThread.join();
    }
}
