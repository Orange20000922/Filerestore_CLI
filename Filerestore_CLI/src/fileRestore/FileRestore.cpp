#include "FileRestore.h"
#include "Logger.h"
#include <iostream>

using namespace std;

FileRestore::FileRestore() : reader(nullptr), parser(nullptr),
    pathResolver(nullptr), scanner(nullptr), overwriteDetector(nullptr),
    currentDrive(0), volumeOpened(false) {

    LOG_DEBUG("FileRestore constructor started");

    // 初始化组件
    LOG_DEBUG("Creating MFTReader...");
    reader = new MFTReader();
    LOG_DEBUG("MFTReader created");

    LOG_DEBUG("Creating MFTParser...");
    parser = new MFTParser(reader);
    LOG_DEBUG("MFTParser created");

    LOG_DEBUG("Creating PathResolver...");
    pathResolver = new PathResolver(reader, parser);
    LOG_DEBUG("PathResolver created");

    LOG_DEBUG("Creating DeletedFileScanner...");
    scanner = new DeletedFileScanner(reader, parser, pathResolver);
    LOG_DEBUG("DeletedFileScanner created");

    LOG_DEBUG("Creating OverwriteDetector...");
    overwriteDetector = new OverwriteDetector(reader);
    LOG_DEBUG("OverwriteDetector created");

    LOG_DEBUG("FileRestore constructor completed");
}

FileRestore::~FileRestore() {
    CloseDrive();

    // 清理组件
    if (overwriteDetector) delete overwriteDetector;
    if (scanner) delete scanner;
    if (pathResolver) delete pathResolver;
    if (parser) delete parser;
    if (reader) delete reader;
}

bool FileRestore::OpenDrive(char driveLetter) {
    LOG_DEBUG_FMT("OpenDrive called for drive %c:", driveLetter);

    if (volumeOpened && currentDrive == driveLetter) {
        LOG_DEBUG("Volume already opened for this drive");
        return true; // 已经打开
    }

    // 关闭旧的卷
    if (volumeOpened) {
        LOG_DEBUG("Closing previously opened volume");
        CloseDrive();
    }

    LOG_DEBUG("Calling reader->OpenVolume...");
    if (reader->OpenVolume(driveLetter)) {
        currentDrive = driveLetter;
        volumeOpened = true;
        LOG_INFO_FMT("Successfully opened drive %c:", driveLetter);
        return true;
    }

    LOG_ERROR_FMT("Failed to open drive %c:", driveLetter);
    return false;
}

void FileRestore::CloseDrive() {
    if (volumeOpened) {
        reader->CloseVolume();
        volumeOpened = false;
        currentDrive = 0;
    }
}

vector<DeletedFileInfo> FileRestore::ScanDeletedFiles(char driveLetter, ULONGLONG maxRecords) {
    LOG_DEBUG_FMT("ScanDeletedFiles called for drive %c:, maxRecords=%llu", driveLetter, maxRecords);
    vector<DeletedFileInfo> emptyList;

    LOG_DEBUG("Calling OpenDrive...");
    if (!OpenDrive(driveLetter)) {
        LOG_ERROR("OpenDrive failed, returning empty list");
        return emptyList;
    }

    // 尝试加载路径缓存
    LOG_INFO("Attempting to load path cache...");
    if (pathResolver->LoadCache(driveLetter)) {
        cout << "Path cache loaded: " << pathResolver->GetCacheSize() << " entries" << endl;
    } else {
        cout << "No path cache available, will build from scratch" << endl;
    }

    LOG_DEBUG("Drive opened successfully, calling scanner->ScanDeletedFiles...");
    vector<DeletedFileInfo> results = scanner->ScanDeletedFiles(maxRecords);

    // 保存路径缓存（用于下次加速）
    if (pathResolver->GetCacheSize() > 0) {
        LOG_INFO("Saving path cache...");
        if (pathResolver->SaveCache(driveLetter)) {
            cout << "Path cache saved: " << pathResolver->GetCacheSize() << " entries" << endl;
        }
    }

    return results;
}

bool FileRestore::RestoreFileByRecordNumber(char driveLetter, ULONGLONG recordNumber, string restoreFilePath) {
    cout << "=== File Recovery Started ===" << endl;
    cout << "Drive: " << driveLetter << ":" << endl;
    cout << "MFT Record Number: " << recordNumber << endl;
    cout << "Output Path: " << restoreFilePath << endl;
    cout << endl;

    // 打开卷
    if (!OpenDrive(driveLetter)) {
        cout << "Failed to open volume." << endl;
        return false;
    }

    // 读取 MFT 记录
    vector<BYTE> mftRecord;
    if (!reader->ReadMFT(recordNumber, mftRecord)) {
        cout << "Failed to read MFT record." << endl;
        return false;
    }

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();

    // 显示文件状态
    cout << "File Status: ";
    if ((header->Flags & 0x01) == 0) {
        cout << "DELETED" << endl;
    } else {
        cout << "ACTIVE" << endl;
    }

    // 获取文件名
    ULONGLONG parentDir;
    wstring fileName = parser->GetFileNameFromRecord(mftRecord, parentDir);
    if (!fileName.empty()) {
        wcout << L"File Name: " << fileName << endl;
    }

    // 检查数据可用性
    bool dataAvailable = parser->CheckDataAvailable(mftRecord);
    cout << "Data Available: " << (dataAvailable ? "YES" : "NO (may be overwritten)") << endl;
    cout << endl;

    if (!dataAvailable) {
        cout << "Warning: Data appears to be overwritten or cleared." << endl;
        cout << "Recovery may fail or produce corrupted data." << endl;
        cout << endl;
    }

    // 提取文件数据
    cout << "Extracting file data..." << endl;
    vector<BYTE> fileData;
    if (!parser->ExtractFileData(mftRecord, fileData)) {
        cout << "Failed to extract file data." << endl;
        cout << "The file data may have been overwritten." << endl;
        return false;
    }

    cout << "Data extracted successfully: " << fileData.size() << " bytes" << endl;
    cout << endl;

    // 保存恢复的文件
    cout << "Writing recovered data to file..." << endl;
    HANDLE hFile = CreateFileA(restoreFilePath.c_str(), GENERIC_WRITE,
        0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE) {
        cout << "Failed to create output file. Error: " << GetLastError() << endl;
        return false;
    }

    DWORD bytesWritten;
    BOOL result = WriteFile(hFile, fileData.data(), (DWORD)fileData.size(), &bytesWritten, NULL);
    CloseHandle(hFile);

    if (result && bytesWritten == fileData.size()) {
        cout << "=== File Recovery Completed Successfully ===" << endl;
        cout << "Recovered " << bytesWritten << " bytes" << endl;
        cout << "Saved to: " << restoreFilePath << endl;
        return true;
    } else {
        cout << "Failed to write recovered data." << endl;
        return false;
    }
}

// 检测文件数据覆盖情况
OverwriteDetectionResult FileRestore::DetectFileOverwrite(char driveLetter, ULONGLONG recordNumber) {
    OverwriteDetectionResult emptyResult;

    cout << "=== Overwrite Detection Started ===" << endl;
    cout << "Drive: " << driveLetter << ":" << endl;
    cout << "MFT Record Number: " << recordNumber << endl;
    cout << endl;

    // 打开卷
    if (!OpenDrive(driveLetter)) {
        cout << "Failed to open volume." << endl;
        return emptyResult;
    }

    // 读取 MFT 记录
    vector<BYTE> mftRecord;
    if (!reader->ReadMFT(recordNumber, mftRecord)) {
        cout << "Failed to read MFT record." << endl;
        return emptyResult;
    }

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();

    // 显示文件状态
    cout << "File Status: ";
    if ((header->Flags & 0x01) == 0) {
        cout << "DELETED" << endl;
    } else {
        cout << "ACTIVE" << endl;
    }

    // 获取文件名
    ULONGLONG parentDir;
    wstring fileName = parser->GetFileNameFromRecord(mftRecord, parentDir);
    if (!fileName.empty()) {
        wcout << L"File Name: " << fileName << endl;
    }

    cout << endl;
    cout << "Analyzing data clusters..." << endl;
    cout << "This may take a while for large files..." << endl;
    cout << endl;

    // 执行覆盖检测
    OverwriteDetectionResult result = overwriteDetector->DetectOverwrite(mftRecord);

    // 显示检测报告
    string report = overwriteDetector->GetDetectionReport(result);
    cout << report << endl;

    return result;
}
