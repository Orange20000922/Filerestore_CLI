#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <map>
#include <atomic>
#include "FileFormatUtils.h"
#include "CarvedFileTypes.h"
#include "FileIntegrityValidator.h"

using namespace std;

class MFTReader;

// ============================================================================
// FileCarverRecovery — 文件恢复引擎
// 从磁盘读取和恢复已雕刻的文件
// ============================================================================
class FileCarverRecovery {
public:
    FileCarverRecovery(MFTReader* reader, const map<string, FileSignature>& signatures);

    // ==================== ZIP 恢复配置/结果 ====================

    struct ZipRecoveryConfig {
        ULONGLONG maxSize = 50ULL * 1024 * 1024 * 1024;  // 最大搜索大小 (默认 50GB)
        ULONGLONG expectedSize = 0;                      // 用户预期大小 (0 = 不限制)
        ULONGLONG expectedSizeTolerance = 0;             // 大小容差 (0 = 自动 10%)
        bool verifyCRC = true;                           // 恢复后验证 CRC
        bool stopOnFirstEOCD = true;                     // 找到第一个 EOCD 就停止
        bool allowFragmented = false;                    // 允许碎片化文件
    };

    struct ZipRecoveryResult {
        bool success = false;
        ULONGLONG actualSize = 0;
        ULONGLONG bytesWritten = 0;
        bool crcValid = false;
        int totalFiles = 0;
        int corruptedFiles = 0;
        bool isFragmented = false;
        string diagnosis;
    };

    // ==================== 核心恢复方法 ====================

    // 恢复 carved 文件到指定路径
    bool RecoverCarvedFile(const CarvedFileInfo& info, const string& outputPath);

    // 恢复前精细化：精确计算文件大小、完整性验证、置信度重估
    bool RefineCarvedFileInfo(CarvedFileInfo& info, bool verbose = true);

    // ZIP EOCD 扫描恢复（推荐方法）
    ZipRecoveryResult RecoverZipWithEOCDScan(
        ULONGLONG startLCN,
        const string& outputPath,
        const ZipRecoveryConfig& config = ZipRecoveryConfig()
    );

    // 验证 ZIP 文件完整性
    static ZipRecoveryResult ValidateZipFile(const string& filePath);

    // 验证 ZIP 数据完整性（从内存）
    static ZipRecoveryResult ValidateZipData(const BYTE* data, size_t dataSize);

    // 停止恢复操作
    void StopRecovery() { shouldStop_ = true; }

private:
    MFTReader* reader_;
    const map<string, FileSignature>& signatures_;
    atomic<bool> shouldStop_{false};

    static constexpr ULONGLONG MEMORY_BUFFER_THRESHOLD = 256ULL * 1024 * 1024;
    static constexpr ULONGLONG MAX_STREAMING_LIMIT = 512ULL * 1024 * 1024;

    // 提取文件数据到内存
    bool ExtractFile(ULONGLONG startLCN, ULONGLONG startOffset,
                     ULONGLONG fileSize, vector<BYTE>& fileData);

    // 验证文件有效性（基于内容）
    double ValidateFileOptimized(const BYTE* data, size_t dataSize,
                                 const FileSignature& sig,
                                 bool signatureAlreadyMatched,
                                 ULONGLONG footerPos);

    // ZIP 恢复内部实现
    ZipRecoveryResult RecoverZipWithEOCDScan_MemoryBuffer(
        ULONGLONG startLCN, const string& outputPath,
        const ZipRecoveryConfig& config, ULONGLONG estimatedSize);

    ZipRecoveryResult RecoverZipWithEOCDScan_Streaming(
        ULONGLONG startLCN, const string& outputPath,
        const ZipRecoveryConfig& config, ULONGLONG estimatedSize);
};
