#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>
#include "Result.h"
#include "ErrorCodes.h"

// 前向声明（避免头文件依赖）
class MFTReader;
class MFTParser;
class PathResolver;
class DeletedFileScanner;
class FileCarver;
class OverwriteDetector;

namespace FR {

// ============================================================================
// 跨平台友好的数据类型
// ============================================================================

// 时间戳（微秒，自1970-01-01）
using Timestamp = uint64_t;

// 文件ID（MFT记录号）
using FileId = uint64_t;

// 文件大小
using FileSize = uint64_t;

// ============================================================================
// 已删除文件信息（简化版，跨平台友好）
// ============================================================================
struct DeletedFile {
    FileId recordNumber;            // MFT记录号
    std::wstring fileName;          // 文件名
    std::wstring fullPath;          // 完整路径
    FileSize fileSize;              // 文件大小（字节）
    FileId parentDirectory;         // 父目录记录号
    bool isDirectory;               // 是否为目录
    Timestamp deletionTime;         // 删除时间（微秒）

    // 恢复可能性评估
    double recoverability;          // 可恢复性 (0.0-1.0)
    double overwritePercentage;     // 覆盖百分比 (0.0-100.0)
    bool dataAvailable;             // 数据是否可用

    DeletedFile()
        : recordNumber(0), fileSize(0), parentDirectory(0)
        , isDirectory(false), deletionTime(0)
        , recoverability(1.0), overwritePercentage(0.0), dataAvailable(true)
    {}
};

// ============================================================================
// 签名扫描（Carving）结果
// ============================================================================
struct CarvedFile {
    std::string extension;          // 文件扩展名
    std::string description;        // 文件类型描述
    FileSize fileSize;              // 文件大小
    uint64_t startCluster;          // 起始簇号
    uint64_t offsetInCluster;       // 簇内偏移

    // 可信度评估
    double confidence;              // 置信度 (0.0-1.0)
    double integrityScore;          // 完整性分数 (0.0-1.0)
    bool headerValid;               // 文件头有效
    bool footerValid;               // 文件尾有效

    // 时间戳（如果可用）
    Timestamp creationTime;
    Timestamp modificationTime;
    bool hasTimestamp;

    CarvedFile()
        : fileSize(0), startCluster(0), offsetInCluster(0)
        , confidence(0.0), integrityScore(0.0)
        , headerValid(false), footerValid(false)
        , creationTime(0), modificationTime(0), hasTimestamp(false)
    {}
};

// ============================================================================
// 扫描选项
// ============================================================================
struct ScanOptions {
    uint64_t maxResults = 10000;        // 最大结果数
    std::wstring extensionFilter;       // 扩展名过滤（如 L".jpg"）
    std::wstring namePattern;           // 文件名模式
    FileSize minSize = 0;               // 最小文件大小
    FileSize maxSize = UINT64_MAX;      // 最大文件大小
    bool includeDirectories = false;    // 是否包含目录
    bool skipSystemFiles = true;        // 跳过系统文件

    ScanOptions() = default;
};

// ============================================================================
// 恢复选项
// ============================================================================
struct RecoveryOptions {
    bool skipOverwriteCheck = false;    // 跳过覆盖检查
    bool preserveTimestamps = true;     // 保留时间戳
    bool forceRecover = false;          // 强制恢复（即使部分覆盖）
    bool createParentDirs = true;       // 自动创建父目录

    RecoveryOptions() = default;
};

// ============================================================================
// 签名扫描选项
// ============================================================================
struct CarvingOptions {
    std::vector<std::string> fileTypes; // 要扫描的文件类型（如 {"jpg", "pdf"}）
    uint64_t maxResults = 1000;         // 最大结果数
    bool useMultithreading = true;      // 使用多线程
    bool extractTimestamps = false;     // 提取时间戳（较慢）
    bool validateIntegrity = true;      // 验证文件完整性

    CarvingOptions() = default;
};

// ============================================================================
// 恢复结果
// ============================================================================
struct RecoveryResult {
    bool success;                       // 是否成功
    std::string outputPath;             // 输出文件路径
    FileSize bytesRecovered;            // 恢复的字节数
    double overwritePercentage;         // 覆盖百分比

    RecoveryResult()
        : success(false), bytesRecovered(0), overwritePercentage(0.0)
    {}
};

// ============================================================================
// 进度回调
// ============================================================================
using ProgressCallback = std::function<void(double percentage, const std::string& status)>;

// ============================================================================
// 统一文件恢复 API
// ============================================================================
class FileRestoreAPI {
private:
    // 内部实现（PIMPL模式）
    struct Impl;
    std::unique_ptr<Impl> pImpl;

    // 私有构造函数（单例）
    FileRestoreAPI();

public:
    ~FileRestoreAPI();

    // 禁止拷贝
    FileRestoreAPI(const FileRestoreAPI&) = delete;
    FileRestoreAPI& operator=(const FileRestoreAPI&) = delete;

    // 获取单例实例
    static FileRestoreAPI& Instance();

    // ========== 卷操作 ==========

    // 打开卷
    Result<void> OpenVolume(char driveLetter);

    // 关闭卷
    void CloseVolume();

    // 检查卷是否已打开
    bool IsVolumeOpen() const;

    // 获取当前驱动器字母
    char GetCurrentDrive() const;

    // ========== 已删除文件扫描 ==========

    // 扫描已删除文件
    Result<std::vector<DeletedFile>> ScanDeletedFiles(
        char driveLetter,
        const ScanOptions& options = ScanOptions()
    );

    // 按扩展名过滤
    static std::vector<DeletedFile> FilterByExtension(
        const std::vector<DeletedFile>& files,
        const std::wstring& extension
    );

    // 按文件名过滤
    static std::vector<DeletedFile> FilterByName(
        const std::vector<DeletedFile>& files,
        const std::wstring& pattern
    );

    // 按大小过滤
    static std::vector<DeletedFile> FilterBySize(
        const std::vector<DeletedFile>& files,
        FileSize minSize,
        FileSize maxSize
    );

    // ========== 文件恢复 ==========

    // 通过记录号恢复单个文件
    Result<RecoveryResult> RecoverFile(
        char driveLetter,
        FileId recordNumber,
        const std::string& outputPath,
        const RecoveryOptions& options = RecoveryOptions()
    );

    // 批量恢复
    Result<std::vector<RecoveryResult>> RecoverFiles(
        char driveLetter,
        const std::vector<FileId>& recordNumbers,
        const std::string& outputDirectory,
        const RecoveryOptions& options = RecoveryOptions(),
        ProgressCallback progress = nullptr
    );

    // ========== 覆盖检测 ==========

    // 检测文件覆盖状态
    Result<double> CheckOverwriteStatus(
        char driveLetter,
        FileId recordNumber
    );

    // ========== 签名扫描（File Carving）==========

    // 获取支持的文件类型列表
    std::vector<std::string> GetSupportedFileTypes() const;

    // 执行签名扫描
    Result<std::vector<CarvedFile>> PerformCarving(
        char driveLetter,
        const CarvingOptions& options = CarvingOptions(),
        ProgressCallback progress = nullptr
    );

    // 恢复签名扫描找到的文件
    Result<RecoveryResult> RecoverCarvedFile(
        char driveLetter,
        const CarvedFile& file,
        const std::string& outputPath
    );

    // ========== 实用工具 ==========

    // 将 FILETIME 转换为 Timestamp
    static Timestamp FileTimeToTimestamp(uint64_t fileTime);

    // 将 Timestamp 转换为字符串
    static std::string TimestampToString(Timestamp ts);

    // 格式化文件大小
    static std::string FormatFileSize(FileSize size);
};

} // namespace FR
