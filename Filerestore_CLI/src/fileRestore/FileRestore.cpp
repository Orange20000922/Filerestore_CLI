#include "FileRestore.h"
#include "Logger.h"
#include <iostream>

using namespace std;

FileRestore::FileRestore() : currentDrive(0), volumeOpened(false) {

    LOG_DEBUG("FileRestore 构造函数开始");

    // 初始化组件
    LOG_DEBUG("正在创建 MFTReader...");
    reader = make_unique<MFTReader>();
    LOG_DEBUG("MFTReader 创建完成");

    LOG_DEBUG("正在创建 MFTParser...");
    parser = make_unique<MFTParser>(reader.get());
    LOG_DEBUG("MFTParser 创建完成");

    LOG_DEBUG("正在创建 PathResolver...");
    pathResolver = make_unique<PathResolver>(reader.get(), parser.get());
    LOG_DEBUG("PathResolver 创建完成");

    LOG_DEBUG("正在创建 DeletedFileScanner...");
    scanner = make_unique<DeletedFileScanner>(reader.get(), parser.get(), pathResolver.get());
    LOG_DEBUG("DeletedFileScanner 创建完成");

    LOG_DEBUG("正在创建 OverwriteDetector...");
    overwriteDetector = make_unique<OverwriteDetector>(reader.get());
    LOG_DEBUG("OverwriteDetector 创建完成");

    LOG_DEBUG("FileRestore 构造函数完成");
}

FileRestore::~FileRestore() {
    CloseDrive();
    // unique_ptr 自动释放资源，无需手动 delete
}

bool FileRestore::OpenDrive(char driveLetter) {
    LOG_DEBUG_FMT("调用 OpenDrive，驱动器 %c:", driveLetter);

    if (volumeOpened && currentDrive == driveLetter) {
        LOG_DEBUG("该驱动器的卷已经打开");
        return true; // 已经打开
    }

    // 关闭旧的卷
    if (volumeOpened) {
        LOG_DEBUG("正在关闭之前打开的卷");
        CloseDrive();
    }

    LOG_DEBUG("调用 reader->OpenVolume...");
    if (reader->OpenVolume(driveLetter)) {
        currentDrive = driveLetter;
        volumeOpened = true;
        LOG_INFO_FMT("成功打开驱动器 %c:", driveLetter);
        return true;
    }

    LOG_ERROR_FMT("打开驱动器 %c: 失败", driveLetter);
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
    LOG_DEBUG_FMT("调用 ScanDeletedFiles，驱动器 %c:，最大记录数=%llu", driveLetter, maxRecords);
    vector<DeletedFileInfo> emptyList;

    LOG_DEBUG("调用 OpenDrive...");
    if (!OpenDrive(driveLetter)) {
        LOG_ERROR("OpenDrive 失败，返回空列表");
        return emptyList;
    }

    // 尝试加载路径缓存
    LOG_INFO("正在尝试加载路径缓存...");
    if (pathResolver->LoadCache(driveLetter)) {
        cout << "路径缓存已加载: " << pathResolver->GetCacheSize() << " 条记录" << endl;
    } else {
        cout << "无可用路径缓存，将从头构建" << endl;
    }

    LOG_DEBUG("驱动器打开成功，调用 scanner->ScanDeletedFiles...");
    vector<DeletedFileInfo> results = scanner->ScanDeletedFiles(maxRecords);

    // 保存路径缓存（用于下次加速）
    if (pathResolver->GetCacheSize() > 0) {
        LOG_INFO("正在保存路径缓存...");
        if (pathResolver->SaveCache(driveLetter)) {
            cout << "路径缓存已保存: " << pathResolver->GetCacheSize() << " 条记录" << endl;
        }
    }

    return results;
}

bool FileRestore::RestoreFileByRecordNumber(char driveLetter, ULONGLONG recordNumber, string restoreFilePath) {
    cout << "=== 文件恢复开始 ===" << endl;
    cout << "驱动器: " << driveLetter << ":" << endl;
    cout << "MFT 记录号: " << recordNumber << endl;
    cout << "输出路径: " << restoreFilePath << endl;
    cout << endl;

    // 打开卷
    if (!OpenDrive(driveLetter)) {
        cout << "打开卷失败。" << endl;
        return false;
    }

    // 读取 MFT 记录
    vector<BYTE> mftRecord;
    if (!reader->ReadMFT(recordNumber, mftRecord)) {
        cout << "读取 MFT 记录失败。" << endl;
        return false;
    }

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();

    // 显示文件状态
    cout << "文件状态: ";
    if ((header->Flags & 0x01) == 0) {
        cout << "已删除" << endl;
    } else {
        cout << "活动" << endl;
    }

    // 获取文件名
    ULONGLONG parentDir;
    wstring fileName = parser->GetFileNameFromRecord(mftRecord, parentDir);
    if (!fileName.empty()) {
        wcout << L"文件名: " << fileName << endl;
    }

    // 检查数据可用性
    bool dataAvailable = parser->CheckDataAvailable(mftRecord);
    cout << "数据可用: " << (dataAvailable ? "是" : "否 (可能已被覆盖)") << endl;
    cout << endl;

    if (!dataAvailable) {
        cout << "警告: 数据似乎已被覆盖或清除。" << endl;
        cout << "恢复可能失败或产生损坏的数据。" << endl;
        cout << endl;
    }

    // 提取文件数据
    cout << "正在提取文件数据..." << endl;
    vector<BYTE> fileData;
    if (!parser->ExtractFileData(mftRecord, fileData)) {
        cout << "提取文件数据失败。" << endl;
        cout << "文件数据可能已被覆盖。" << endl;
        return false;
    }

    cout << "数据提取成功: " << fileData.size() << " 字节" << endl;
    cout << endl;

    // 保存恢复的文件
    cout << "正在将恢复的数据写入文件..." << endl;

    // 确保父目录存在
    string parentPath = restoreFilePath;
    size_t lastSlash = parentPath.find_last_of("\\/");
    if (lastSlash != string::npos) {
        parentPath = parentPath.substr(0, lastSlash);
        // 递归创建目录
        string currentPath;
        for (size_t i = 0; i < parentPath.length(); i++) {
            char c = parentPath[i];
            currentPath += c;
            if (c == '\\' || c == '/' || i == parentPath.length() - 1) {
                if (currentPath.length() > 2) {  // 跳过驱动器号如 "C:"
                    CreateDirectoryA(currentPath.c_str(), NULL);
                }
            }
        }
    }

    HANDLE hFile = CreateFileA(restoreFilePath.c_str(), GENERIC_WRITE,
        0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE) {
        cout << "创建输出文件失败。错误码: " << GetLastError() << endl;
        return false;
    }

    DWORD bytesWritten;
    BOOL result = WriteFile(hFile, fileData.data(), (DWORD)fileData.size(), &bytesWritten, NULL);
    CloseHandle(hFile);

    if (result && bytesWritten == fileData.size()) {
        cout << "=== 文件恢复成功完成 ===" << endl;
        cout << "已恢复 " << bytesWritten << " 字节" << endl;
        cout << "已保存到: " << restoreFilePath << endl;
        return true;
    } else {
        cout << "写入恢复数据失败。" << endl;
        return false;
    }
}

// 检测文件数据覆盖情况
OverwriteDetectionResult FileRestore::DetectFileOverwrite(char driveLetter, ULONGLONG recordNumber) {
    OverwriteDetectionResult emptyResult;

    cout << "=== 覆盖检测开始 ===" << endl;
    cout << "驱动器: " << driveLetter << ":" << endl;
    cout << "MFT 记录号: " << recordNumber << endl;
    cout << endl;

    // 打开卷
    if (!OpenDrive(driveLetter)) {
        cout << "打开卷失败。" << endl;
        return emptyResult;
    }

    // 读取 MFT 记录
    vector<BYTE> mftRecord;
    if (!reader->ReadMFT(recordNumber, mftRecord)) {
        cout << "读取 MFT 记录失败。" << endl;
        return emptyResult;
    }

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();

    // 显示文件状态
    cout << "文件状态: ";
    if ((header->Flags & 0x01) == 0) {
        cout << "已删除" << endl;
    } else {
        cout << "活动" << endl;
    }

    // 获取文件名
    ULONGLONG parentDir;
    wstring fileName = parser->GetFileNameFromRecord(mftRecord, parentDir);
    if (!fileName.empty()) {
        wcout << L"文件名: " << fileName << endl;
    }

    cout << endl;
    cout << "正在分析数据簇..." << endl;
    cout << "对于大文件这可能需要一些时间..." << endl;
    cout << endl;

    // 执行覆盖检测
    OverwriteDetectionResult result = overwriteDetector->DetectOverwrite(mftRecord);

    // 显示检测报告
    string report = overwriteDetector->GetDetectionReport(result);
    cout << report << endl;

    return result;
}
