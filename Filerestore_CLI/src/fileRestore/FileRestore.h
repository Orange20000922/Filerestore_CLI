#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <memory>
#include "MFTStructures.h"
#include "MFTReader.h"
#include "MFTParser.h"
#include "PathResolver.h"
#include "DeletedFileScanner.h"
#include "OverwriteDetector.h"

using namespace std;

// 文件恢复主类 - 协调各个组件完成文件恢复任务
class FileRestore
{
private:
    unique_ptr<MFTReader> reader;
    unique_ptr<MFTParser> parser;
    unique_ptr<PathResolver> pathResolver;
    unique_ptr<DeletedFileScanner> scanner;
    unique_ptr<OverwriteDetector> overwriteDetector;

    char currentDrive;
    bool volumeOpened;

public:
    FileRestore();
    ~FileRestore();

    // 打开/关闭卷
    bool OpenDrive(char driveLetter);
    void CloseDrive();

    // 扫描已删除文件
    vector<DeletedFileInfo> ScanDeletedFiles(char driveLetter, ULONGLONG maxRecords = 10000);

    // 通过MFT记录号恢复文件（核心功能）
    bool RestoreFileByRecordNumber(char driveLetter, ULONGLONG recordNumber, string restoreFilePath);

    // 检测文件数据覆盖情况（新增）
    OverwriteDetectionResult DetectFileOverwrite(char driveLetter, ULONGLONG recordNumber);

    // 获取覆盖检测器（用于高级用法）
    OverwriteDetector* GetOverwriteDetector() { return overwriteDetector.get(); }

    // 设置过滤级别
    void SetFilterLevel(FilterLevel level) { if (scanner) scanner->SetFilterLevel(level); }
    FilterLevel GetFilterLevel() const { return scanner ? scanner->GetFilterLevel() : FILTER_SKIP_PATH; }

    // 静态辅助函数：转发到 DeletedFileScanner 的筛选方法
    static vector<DeletedFileInfo> FilterByExtension(const vector<DeletedFileInfo>& files, const wstring& extension) {
        return DeletedFileScanner::FilterByExtension(files, extension);
    }
    static vector<DeletedFileInfo> FilterByName(const vector<DeletedFileInfo>& files, const wstring& namePattern) {
        return DeletedFileScanner::FilterByName(files, namePattern);
    }
};
