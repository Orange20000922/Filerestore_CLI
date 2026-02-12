#include "FileRestoreAPI.h"
#include "MFTReader.h"
#include "MFTParser.h"
#include "PathResolver.h"
#include "DeletedFileScanner.h"
#include "FileRestore.h"
#include "FileCarver.h"
#include "OverwriteDetector.h"
#include "CarvedFileTypes.h"
#include <Windows.h>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>

namespace FR {

// ============================================================================
// 内部实现结构
// ============================================================================
struct FileRestoreAPI::Impl {
    std::unique_ptr<MFTReader> reader;
    std::unique_ptr<MFTParser> parser;
    std::unique_ptr<PathResolver> pathResolver;
    std::unique_ptr<DeletedFileScanner> scanner;
    std::unique_ptr<FileCarver> carver;
    std::unique_ptr<OverwriteDetector> overwriteDetector;

    char currentDrive = 0;
    bool volumeOpen = false;

    Impl() = default;

    void CleanupComponents() {
        scanner.reset();
        pathResolver.reset();
        parser.reset();
        carver.reset();
        overwriteDetector.reset();

        if (reader) {
            reader->CloseVolume();
            reader.reset();
        }

        volumeOpen = false;
        currentDrive = 0;
    }

    bool InitializeComponents() {
        if (!reader || !reader->IsVolumeOpen()) {
            return false;
        }

        parser = std::make_unique<MFTParser>(reader.get());
        pathResolver = std::make_unique<PathResolver>(reader.get(), parser.get());
        scanner = std::make_unique<DeletedFileScanner>(reader.get(), parser.get(), pathResolver.get());
        carver = std::make_unique<FileCarver>(reader.get());
        overwriteDetector = std::make_unique<OverwriteDetector>(reader.get());

        return true;
    }
};

// ============================================================================
// 构造函数和析构函数
// ============================================================================
FileRestoreAPI::FileRestoreAPI()
    : pImpl(std::make_unique<Impl>())
{
}

FileRestoreAPI::~FileRestoreAPI() {
    CloseVolume();
}

// ============================================================================
// 单例实例
// ============================================================================
FileRestoreAPI& FileRestoreAPI::Instance() {
    static FileRestoreAPI instance;
    return instance;
}

// ============================================================================
// 卷操作
// ============================================================================
Result<void> FileRestoreAPI::OpenVolume(char driveLetter) {
    // 验证驱动器字母
    if (!isalpha(driveLetter)) {
        return Result<void>::Failure(
            ErrorCode::SystemInvalidDriveLetter,
            "Invalid drive letter"
        );
    }

    driveLetter = (char)toupper(driveLetter);

    // 如果已打开同一卷，直接返回成功
    if (pImpl->volumeOpen && pImpl->currentDrive == driveLetter) {
        return Result<void>::Success();
    }

    // 关闭之前打开的卷
    CloseVolume();

    // 创建新的 MFTReader
    pImpl->reader = std::make_unique<MFTReader>();

    // 尝试打开卷
    if (!pImpl->reader->OpenVolume(driveLetter)) {
        pImpl->reader.reset();
        return Result<void>::Failure(
            MakeSystemError(
                ErrorCode::SystemDiskAccessDenied,
                "Failed to open volume",
                std::string("Drive: ") + driveLetter
            )
        );
    }

    // 加载 MFT data runs（支持碎片化 MFT 的正确记录定位）
    pImpl->reader->GetTotalMFTRecords();

    // 初始化其他组件
    if (!pImpl->InitializeComponents()) {
        pImpl->CleanupComponents();
        return Result<void>::Failure(
            ErrorCode::MemoryAllocationFailed,
            "Failed to initialize components"
        );
    }

    pImpl->currentDrive = driveLetter;
    pImpl->volumeOpen = true;

    return Result<void>::Success();
}

void FileRestoreAPI::CloseVolume() {
    pImpl->CleanupComponents();
}

bool FileRestoreAPI::IsVolumeOpen() const {
    return pImpl->volumeOpen;
}

char FileRestoreAPI::GetCurrentDrive() const {
    return pImpl->currentDrive;
}

// ============================================================================
// 辅助函数：将内部 DeletedFileInfo 转换为 API 的 DeletedFile
// ============================================================================
static DeletedFile ConvertDeletedFileInfo(const DeletedFileInfo& info) {
    DeletedFile file;
    file.recordNumber = info.recordNumber;
    file.fileName = info.fileName;
    file.fullPath = info.filePath;
    file.fileSize = info.fileSize;
    file.parentDirectory = info.parentDirectory;
    file.isDirectory = info.isDirectory;
    file.dataAvailable = info.dataAvailable;
    file.overwritePercentage = info.overwritePercentage;

    // 转换 FILETIME 到 Timestamp
    ULARGE_INTEGER uli;
    uli.LowPart = info.deletionTime.dwLowDateTime;
    uli.HighPart = info.deletionTime.dwHighDateTime;
    file.deletionTime = FileRestoreAPI::FileTimeToTimestamp(uli.QuadPart);

    // 计算可恢复性
    if (info.overwriteDetected) {
        file.recoverability = 1.0 - (info.overwritePercentage / 100.0);
    } else {
        file.recoverability = info.dataAvailable ? 1.0 : 0.0;
    }

    return file;
}

// ============================================================================
// 已删除文件扫描
// ============================================================================
Result<std::vector<DeletedFile>> FileRestoreAPI::ScanDeletedFiles(
    char driveLetter,
    const ScanOptions& options
) {
    // 打开卷
    auto openResult = OpenVolume(driveLetter);
    if (openResult.IsFailure()) {
        return Result<std::vector<DeletedFile>>::Failure(openResult.Error());
    }

    if (!pImpl->scanner) {
        return Result<std::vector<DeletedFile>>::Failure(
            ErrorCode::LogicInvalidArgument,
            "Scanner not initialized"
        );
    }

    // 执行扫描
    std::vector<DeletedFileInfo> rawResults = pImpl->scanner->ScanDeletedFiles(options.maxResults);

    // 转换结果
    std::vector<DeletedFile> results;
    results.reserve(rawResults.size());

    for (const auto& info : rawResults) {
        DeletedFile file = ConvertDeletedFileInfo(info);

        // 应用过滤器
        if (options.includeDirectories == false && file.isDirectory) {
            continue;
        }

        if (file.fileSize < options.minSize || file.fileSize > options.maxSize) {
            continue;
        }

        if (!options.extensionFilter.empty()) {
            // 检查扩展名
            size_t dotPos = file.fileName.rfind(L'.');
            if (dotPos == std::wstring::npos) {
                continue;
            }
            std::wstring ext = file.fileName.substr(dotPos);
            // 不区分大小写比较
            std::wstring filterLower = options.extensionFilter;
            std::wstring extLower = ext;
            std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::towlower);
            std::transform(extLower.begin(), extLower.end(), extLower.begin(), ::towlower);
            if (extLower != filterLower) {
                continue;
            }
        }

        results.push_back(std::move(file));
    }

    return Result<std::vector<DeletedFile>>::Success(std::move(results));
}

// ============================================================================
// 过滤函数
// ============================================================================
std::vector<DeletedFile> FileRestoreAPI::FilterByExtension(
    const std::vector<DeletedFile>& files,
    const std::wstring& extension
) {
    std::vector<DeletedFile> result;
    std::wstring extLower = extension;
    std::transform(extLower.begin(), extLower.end(), extLower.begin(), ::towlower);

    for (const auto& file : files) {
        size_t dotPos = file.fileName.rfind(L'.');
        if (dotPos != std::wstring::npos) {
            std::wstring fileExt = file.fileName.substr(dotPos);
            std::transform(fileExt.begin(), fileExt.end(), fileExt.begin(), ::towlower);
            if (fileExt == extLower) {
                result.push_back(file);
            }
        }
    }
    return result;
}

std::vector<DeletedFile> FileRestoreAPI::FilterByName(
    const std::vector<DeletedFile>& files,
    const std::wstring& pattern
) {
    std::vector<DeletedFile> result;
    std::wstring patternLower = pattern;
    std::transform(patternLower.begin(), patternLower.end(), patternLower.begin(), ::towlower);

    for (const auto& file : files) {
        std::wstring nameLower = file.fileName;
        std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::towlower);
        if (nameLower.find(patternLower) != std::wstring::npos) {
            result.push_back(file);
        }
    }
    return result;
}

std::vector<DeletedFile> FileRestoreAPI::FilterBySize(
    const std::vector<DeletedFile>& files,
    FileSize minSize,
    FileSize maxSize
) {
    std::vector<DeletedFile> result;
    for (const auto& file : files) {
        if (file.fileSize >= minSize && file.fileSize <= maxSize) {
            result.push_back(file);
        }
    }
    return result;
}

// ============================================================================
// 文件恢复
// ============================================================================
Result<RecoveryResult> FileRestoreAPI::RecoverFile(
    char driveLetter,
    FileId recordNumber,
    const std::string& outputPath,
    const RecoveryOptions& options
) {
    // 打开卷
    auto openResult = OpenVolume(driveLetter);
    if (openResult.IsFailure()) {
        return Result<RecoveryResult>::Failure(openResult.Error());
    }

    // 使用现有的 FileRestore 类进行恢复
    FileRestore fileRestore;

    bool success = fileRestore.RestoreFileByRecordNumber(
        driveLetter,
        recordNumber,
        outputPath
    );

    RecoveryResult result;
    result.success = success;
    result.outputPath = outputPath;

    if (success) {
        // 尝试获取恢复的文件大小
        HANDLE hFile = CreateFileA(
            outputPath.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );

        if (hFile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER fileSize;
            if (GetFileSizeEx(hFile, &fileSize)) {
                result.bytesRecovered = fileSize.QuadPart;
            }
            CloseHandle(hFile);
        }

        return Result<RecoveryResult>::Success(std::move(result));
    } else {
        return Result<RecoveryResult>::Failure(
            ErrorCode::RecoveryFileOverwritten,
            "Failed to recover file"
        );
    }
}

Result<std::vector<RecoveryResult>> FileRestoreAPI::RecoverFiles(
    char driveLetter,
    const std::vector<FileId>& recordNumbers,
    const std::string& outputDirectory,
    const RecoveryOptions& options,
    ProgressCallback progress
) {
    std::vector<RecoveryResult> results;
    results.reserve(recordNumbers.size());

    size_t total = recordNumbers.size();
    size_t processed = 0;

    for (FileId recordNumber : recordNumbers) {
        std::string outputPath = outputDirectory + "\\" + std::to_string(recordNumber) + "_recovered";

        auto recoverResult = RecoverFile(driveLetter, recordNumber, outputPath, options);

        RecoveryResult result;
        if (recoverResult.IsSuccess()) {
            result = recoverResult.Value();
        } else {
            result.success = false;
            result.outputPath = outputPath;
        }

        results.push_back(std::move(result));

        processed++;
        if (progress) {
            double percentage = (double)processed / total * 100.0;
            progress(percentage, "Recovering file " + std::to_string(processed) + "/" + std::to_string(total));
        }
    }

    return Result<std::vector<RecoveryResult>>::Success(std::move(results));
}

// ============================================================================
// 覆盖检测
// ============================================================================
Result<double> FileRestoreAPI::CheckOverwriteStatus(
    char driveLetter,
    FileId recordNumber
) {
    // 打开卷
    auto openResult = OpenVolume(driveLetter);
    if (openResult.IsFailure()) {
        return Result<double>::Failure(openResult.Error());
    }

    // 使用现有的 FileRestore 类进行检测
    FileRestore fileRestore;
    OverwriteDetectionResult detResult = fileRestore.DetectFileOverwrite(driveLetter, recordNumber);

    return Result<double>::Success(detResult.overwritePercentage);
}

// ============================================================================
// 签名扫描（File Carving）
// ============================================================================
std::vector<std::string> FileRestoreAPI::GetSupportedFileTypes() const {
    // 返回支持的文件类型列表
    return {
        "jpg", "jpeg", "png", "gif", "bmp", "webp",
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        "zip", "rar", "7z", "tar", "gz",
        "mp3", "mp4", "avi", "mkv", "mov", "wav", "flac",
        "exe", "dll", "msi",
        "sqlite", "psd", "ai", "eps"
    };
}

// 辅助函数：将内部 CarvedFileInfo 转换为 API 的 CarvedFile
static CarvedFile ConvertCarvedFileInfo(const CarvedFileInfo& info) {
    CarvedFile file;
    file.extension = info.extension;
    file.description = info.description;
    file.fileSize = info.fileSize;
    file.startCluster = info.startLCN;
    file.offsetInCluster = info.startOffset;
    file.confidence = info.confidence;
    file.integrityScore = info.integrityScore;
    file.headerValid = (info.confidence > 0.5);  // 从置信度推断
    file.footerValid = info.hasValidFooter;

    // 转换时间戳
    if (info.tsSource != TS_NONE_1) {
        file.hasTimestamp = true;
        ULARGE_INTEGER uli;
        uli.LowPart = info.creationTime.dwLowDateTime;
        uli.HighPart = info.creationTime.dwHighDateTime;
        file.creationTime = FileRestoreAPI::FileTimeToTimestamp(uli.QuadPart);

        uli.LowPart = info.modificationTime.dwLowDateTime;
        uli.HighPart = info.modificationTime.dwHighDateTime;
        file.modificationTime = FileRestoreAPI::FileTimeToTimestamp(uli.QuadPart);
    }

    return file;
}

Result<std::vector<CarvedFile>> FileRestoreAPI::PerformCarving(
    char driveLetter,
    const CarvingOptions& options,
    ProgressCallback progress
) {
    // 打开卷
    auto openResult = OpenVolume(driveLetter);
    if (openResult.IsFailure()) {
        return Result<std::vector<CarvedFile>>::Failure(openResult.Error());
    }

    if (!pImpl->carver) {
        return Result<std::vector<CarvedFile>>::Failure(
            ErrorCode::LogicInvalidArgument,
            "Carver not initialized"
        );
    }

    // 执行签名扫描
    std::vector<CarvedFileInfo> rawResults;

    if (options.fileTypes.empty()) {
        // 扫描所有支持的类型
        rawResults = pImpl->carver->ScanForFileTypes(
            GetSupportedFileTypes(),
            CARVE_SMART,
            options.maxResults
        );
    } else {
        rawResults = pImpl->carver->ScanForFileTypes(
            options.fileTypes,
            CARVE_SMART,
            options.maxResults
        );
    }

    // 转换结果
    std::vector<CarvedFile> results;
    results.reserve(rawResults.size());

    for (const auto& info : rawResults) {
        results.push_back(ConvertCarvedFileInfo(info));
    }

    return Result<std::vector<CarvedFile>>::Success(std::move(results));
}

Result<RecoveryResult> FileRestoreAPI::RecoverCarvedFile(
    char driveLetter,
    const CarvedFile& file,
    const std::string& outputPath
) {
    // 打开卷
    auto openResult = OpenVolume(driveLetter);
    if (openResult.IsFailure()) {
        return Result<RecoveryResult>::Failure(openResult.Error());
    }

    if (!pImpl->reader) {
        return Result<RecoveryResult>::Failure(
            ErrorCode::LogicInvalidArgument,
            "Reader not initialized"
        );
    }

    // 读取文件数据
    ULONGLONG bytesPerCluster = pImpl->reader->GetBytesPerCluster();
    ULONGLONG clustersNeeded = (file.fileSize + file.offsetInCluster + bytesPerCluster - 1) / bytesPerCluster;

    std::vector<BYTE> buffer;
    if (!pImpl->reader->ReadClusters(file.startCluster, clustersNeeded, buffer)) {
        return Result<RecoveryResult>::Failure(
            ErrorCode::IOReadFailed,
            "Failed to read file data"
        );
    }

    // 写入文件
    HANDLE hFile = CreateFileA(
        outputPath.c_str(),
        GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        return Result<RecoveryResult>::Failure(
            MakeSystemError(
                ErrorCode::IOWriteFailed,
                "Failed to create output file",
                outputPath
            )
        );
    }

    DWORD bytesWritten;
    BOOL writeSuccess = WriteFile(
        hFile,
        buffer.data() + file.offsetInCluster,
        (DWORD)file.fileSize,
        &bytesWritten,
        NULL
    );

    CloseHandle(hFile);

    if (!writeSuccess || bytesWritten != file.fileSize) {
        return Result<RecoveryResult>::Failure(
            ErrorCode::IOWriteFailed,
            "Failed to write file data"
        );
    }

    RecoveryResult result;
    result.success = true;
    result.outputPath = outputPath;
    result.bytesRecovered = file.fileSize;
    result.overwritePercentage = 0.0;

    return Result<RecoveryResult>::Success(std::move(result));
}

// ============================================================================
// 实用工具
// ============================================================================
Timestamp FileRestoreAPI::FileTimeToTimestamp(uint64_t fileTime) {
    // Windows FILETIME: 100纳秒间隔，自1601-01-01
    // Unix timestamp: 微秒，自1970-01-01
    // 差值: 11644473600 秒

    if (fileTime == 0) return 0;

    // 转换为微秒
    uint64_t microseconds = fileTime / 10;

    // 减去1601到1970的差值（微秒）
    const uint64_t EPOCH_DIFF_MICROSECONDS = 11644473600ULL * 1000000ULL;

    if (microseconds < EPOCH_DIFF_MICROSECONDS) {
        return 0;
    }

    return microseconds - EPOCH_DIFF_MICROSECONDS;
}

std::string FileRestoreAPI::TimestampToString(Timestamp ts) {
    if (ts == 0) return "N/A";

    // 转换为 time_t（秒）
    time_t seconds = (time_t)(ts / 1000000);

    struct tm timeinfo;
    localtime_s(&timeinfo, &seconds);

    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);

    return std::string(buffer);
}

std::string FileRestoreAPI::FormatFileSize(FileSize size) {
    std::ostringstream oss;

    if (size < 1024) {
        oss << size << " B";
    } else if (size < 1024 * 1024) {
        oss << std::fixed << std::setprecision(1) << (double)size / 1024 << " KB";
    } else if (size < 1024 * 1024 * 1024) {
        oss << std::fixed << std::setprecision(1) << (double)size / (1024 * 1024) << " MB";
    } else {
        oss << std::fixed << std::setprecision(2) << (double)size / (1024 * 1024 * 1024) << " GB";
    }

    return oss.str();
}

} // namespace FR
