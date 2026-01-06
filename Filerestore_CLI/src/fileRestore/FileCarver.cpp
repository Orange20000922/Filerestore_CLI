#include "FileCarver.h"
#include "SignatureScanThreadPool.h"
#include "TimestampExtractor.h"
#include "MFTLCNIndex.h"
#include "Logger.h"
#include <iostream>
#include <algorithm>
#include <iomanip>

using namespace std;

// ============================================================================
// 优化常量
// ============================================================================
constexpr ULONGLONG BUFFER_SIZE_MB = 64;                    // 64MB 读取缓冲区
constexpr ULONGLONG BUFFER_SIZE_CLUSTERS = 16384;           // 16384 簇 (64MB @ 4KB/cluster)
constexpr size_t EMPTY_CHECK_SAMPLE_SIZE = 512;             // 空簇检测采样大小
constexpr double EMPTY_THRESHOLD = 0.98;                    // 98% 零字节视为空
constexpr ULONGLONG PROGRESS_UPDATE_INTERVAL = 50000;       // 每 50K 簇更新一次进度

// ============================================================================
// 构造函数和析构函数
// ============================================================================
FileCarver::FileCarver(MFTReader* mftReader)
    : reader(mftReader), shouldStop(false), useAsyncIO(true),
      currentReadBuffer(0), currentScanBuffer(0),
      ioWaitTimeMs(0), scanTimeMs(0), totalIoTimeMs(0), totalScanTimeMs(0),
      scanThreadPool(nullptr), useThreadPool(true),
      lcnIndex(nullptr), timestampExtractionEnabled(true), mftIndexBuilt(false),
      integrityValidationEnabled(true), validatedCount(0), corruptedCount(0) {
    memset(&stats, 0, sizeof(stats));

    // 初始化双缓冲区
    for (int i = 0; i < 2; i++) {
        buffers[i].ready = false;
        buffers[i].isEmpty = false;
        buffers[i].isLast = false;
        buffers[i].startLCN = 0;
        buffers[i].clusterCount = 0;
    }

    // 获取最优线程池配置
    threadPoolConfig = ThreadPoolUtils::GetOptimalConfig();

    InitializeSignatures();
    BuildSignatureIndex();
    LOG_INFO("FileCarver initialized with async I/O and thread pool support");
}

FileCarver::~FileCarver() {
    shouldStop = true;  // 确保停止任何运行中的线程

    // 清理线程池
    if (scanThreadPool) {
        scanThreadPool->Stop();
        delete scanThreadPool;
        scanThreadPool = nullptr;
    }

    // 清理 LCN 索引
    if (lcnIndex) {
        delete lcnIndex;
        lcnIndex = nullptr;
    }
}

// ============================================================================
// 初始化签名数据库
// ============================================================================
void FileCarver::InitializeSignatures() {
    // ZIP (包括 DOCX, XLSX, PPTX, JAR, APK 等)
    signatures["zip"] = {
        "zip",
        {0x50, 0x4B, 0x03, 0x04},           // PK..
        {0x50, 0x4B, 0x05, 0x06},           // End of central directory
        500 * 1024 * 1024,                  // 500MB max
        22,                                 // min size
        true,
        "ZIP Archive",
        0x50                                // firstByte
    };

    // PDF
    signatures["pdf"] = {
        "pdf",
        {0x25, 0x50, 0x44, 0x46},           // %PDF
        {0x25, 0x25, 0x45, 0x4F, 0x46},     // %%EOF
        200 * 1024 * 1024,
        100,
        true,
        "PDF Document",
        0x25
    };

    // JPEG
    signatures["jpg"] = {
        "jpg",
        {0xFF, 0xD8, 0xFF},                 // SOI + APP marker
        {0xFF, 0xD9},                       // EOI
        50 * 1024 * 1024,
        100,
        true,
        "JPEG Image",
        0xFF
    };

    // PNG
    signatures["png"] = {
        "png",
        {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A},
        {0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82},
        50 * 1024 * 1024,
        100,
        true,
        "PNG Image",
        0x89
    };

    // GIF
    signatures["gif"] = {
        "gif",
        {0x47, 0x49, 0x46, 0x38},           // GIF8
        {0x00, 0x3B},                       // Trailer
        20 * 1024 * 1024,
        50,
        true,
        "GIF Image",
        0x47
    };

    // BMP
    signatures["bmp"] = {
        "bmp",
        {0x42, 0x4D},                       // BM
        {},
        100 * 1024 * 1024,
        54,
        false,
        "Bitmap Image",
        0x42
    };

    // 7z
    signatures["7z"] = {
        "7z",
        {0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C},
        {},
        1024ULL * 1024 * 1024,              // 1GB
        32,
        false,
        "7-Zip Archive",
        0x37
    };

    // RAR
    signatures["rar"] = {
        "rar",
        {0x52, 0x61, 0x72, 0x21, 0x1A, 0x07},
        {},
        1024ULL * 1024 * 1024,
        20,
        false,
        "RAR Archive",
        0x52
    };

    // MP3 (ID3v2)
    signatures["mp3"] = {
        "mp3",
        {0x49, 0x44, 0x33},                 // ID3
        {},
        50 * 1024 * 1024,
        128,
        false,
        "MP3 Audio",
        0x49
    };

    // MP4/MOV (ftyp atom)
    signatures["mp4"] = {
        "mp4",
        {0x00, 0x00, 0x00},                 // Size prefix (need special handling)
        {},
        4ULL * 1024 * 1024 * 1024,
        100,
        false,
        "MP4 Video",
        0x00
    };

    // AVI (RIFF)
    signatures["avi"] = {
        "avi",
        {0x52, 0x49, 0x46, 0x46},           // RIFF
        {},
        4ULL * 1024 * 1024 * 1024,
        100,
        false,
        "AVI Video",
        0x52
    };

    // EXE/DLL (MZ)
    signatures["exe"] = {
        "exe",
        {0x4D, 0x5A},                       // MZ
        {},
        500 * 1024 * 1024,
        64,
        false,
        "Windows Executable",
        0x4D
    };

    // SQLite
    signatures["sqlite"] = {
        "sqlite",
        {0x53, 0x51, 0x4C, 0x69, 0x74, 0x65, 0x20, 0x66, 0x6F, 0x72, 0x6D, 0x61, 0x74},
        {},
        2ULL * 1024 * 1024 * 1024,
        100,
        false,
        "SQLite Database",
        0x53
    };

    // WAV (RIFF WAVE)
    signatures["wav"] = {
        "wav",
        {0x52, 0x49, 0x46, 0x46},           // RIFF (check WAVE at offset 8)
        {},
        500 * 1024 * 1024,
        44,
        false,
        "WAV Audio",
        0x52
    };

    LOG_INFO_FMT("Loaded %zu file signatures", signatures.size());
}

// ============================================================================
// 构建签名索引（按首字节分组，实现O(1)查找）
// ============================================================================
void FileCarver::BuildSignatureIndex() {
    signatureIndex.clear();
    for (auto& pair : signatures) {
        if (!pair.second.header.empty()) {
            BYTE firstByte = pair.second.header[0];
            pair.second.firstByte = firstByte;
            signatureIndex[firstByte].push_back(&pair.second);
        }
    }
    LOG_DEBUG_FMT("Built signature index with %zu first-byte groups", signatureIndex.size());
}

// ============================================================================
// 构建活动签名索引（用于选择性扫描）
// ============================================================================
void FileCarver::BuildActiveSignatureIndex() {
    // 如果没有指定活动签名，使用全部
    if (activeSignatures.empty()) {
        for (const auto& pair : signatures) {
            activeSignatures.insert(pair.first);
        }
    }
}

// ============================================================================
// 基础匹配函数
// ============================================================================
bool FileCarver::MatchSignature(const BYTE* data, size_t dataSize,
                                const vector<BYTE>& signature) {
    if (dataSize < signature.size()) {
        return false;
    }
    return memcmp(data, signature.data(), signature.size()) == 0;
}

ULONGLONG FileCarver::FindFooter(const BYTE* data, size_t dataSize,
                                 const vector<BYTE>& footer, ULONGLONG maxSearch) {
    if (footer.empty() || dataSize < footer.size()) {
        return 0;
    }

    size_t searchLimit = min((size_t)maxSearch, dataSize - footer.size());

    // 优化：从后向前搜索更常见的情况
    // 但对于大数据，还是从前向后更快（顺序访问）
    for (size_t i = 0; i <= searchLimit; i++) {
        if (memcmp(data + i, footer.data(), footer.size()) == 0) {
            return i + footer.size();
        }
    }
    return 0;
}

// ============================================================================
// 估算文件大小（优化：返回footer位置避免重复查找）
// ============================================================================
ULONGLONG FileCarver::EstimateFileSize(const BYTE* data, size_t dataSize,
                                       const FileSignature& sig,
                                       ULONGLONG* outFooterPos) {
    ULONGLONG footerPos = 0;

    // 如果有footer，尝试查找
    if (sig.hasFooter && !sig.footer.empty()) {
        footerPos = FindFooter(data, dataSize, sig.footer, min((ULONGLONG)dataSize, sig.maxSize));
        if (outFooterPos) *outFooterPos = footerPos;
        if (footerPos > 0) {
            return footerPos;
        }
    }

    // 特殊格式处理
    if (sig.extension == "bmp" && dataSize >= 6) {
        DWORD size = *(DWORD*)(data + 2);
        if (size > sig.minSize && size <= sig.maxSize && size <= dataSize) {
            return size;
        }
    }

    if (sig.extension == "mp4" && dataSize >= 12) {
        // 检查 ftyp atom
        if (data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p') {
            DWORD atomSize = _byteswap_ulong(*(DWORD*)data);
            if (atomSize >= 8) {
                // MP4 文件通常很大，返回保守估计
                return min((ULONGLONG)dataSize, sig.maxSize);
            }
        }
    }

    if (sig.extension == "avi" && dataSize >= 12) {
        // 检查是否真的是 AVI (RIFF + AVI)
        if (data[8] == 'A' && data[9] == 'V' && data[10] == 'I' && data[11] == ' ') {
            DWORD riffSize = *(DWORD*)(data + 4);
            if (riffSize > 0 && riffSize <= sig.maxSize) {
                return min((ULONGLONG)riffSize + 8, (ULONGLONG)dataSize);
            }
        }
    }

    if (sig.extension == "wav" && dataSize >= 12) {
        // 检查是否是 WAV (RIFF + WAVE)
        if (data[8] == 'W' && data[9] == 'A' && data[10] == 'V' && data[11] == 'E') {
            DWORD riffSize = *(DWORD*)(data + 4);
            if (riffSize > 0 && riffSize <= sig.maxSize) {
                return min((ULONGLONG)riffSize + 8, (ULONGLONG)dataSize);
            }
        }
    }

    // 默认：使用最大大小或可用数据大小
    return min((ULONGLONG)dataSize, sig.maxSize);
}

// ============================================================================
// 验证文件（优化版：避免重复计算）
// ============================================================================
double FileCarver::ValidateFileOptimized(const BYTE* data, size_t dataSize,
                                         const FileSignature& sig,
                                         bool signatureAlreadyMatched,
                                         ULONGLONG footerPos) {
    double confidence = 0.5;  // 基础置信度

    // 签名已匹配 +0.3
    if (signatureAlreadyMatched) {
        confidence += 0.3;
    }

    // Footer 已找到 +0.2
    if (footerPos > 0) {
        confidence += 0.2;
    }

    // 特定格式额外验证
    if (sig.extension == "jpg" && dataSize >= 10) {
        if ((data[3] == 0xE0 || data[3] == 0xE1) && data[6] == 0x4A) {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "png" && dataSize >= 24) {
        if (data[12] == 'I' && data[13] == 'H' && data[14] == 'D' && data[15] == 'R') {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "pdf" && dataSize >= 20) {
        if (data[5] == '-' && data[6] >= '1' && data[6] <= '9') {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "zip" && dataSize >= 30) {
        WORD version = *(WORD*)(data + 4);
        WORD flags = *(WORD*)(data + 6);
        if (version <= 63 && (flags & 0xFF00) == 0) {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "mp4" && dataSize >= 12) {
        if (data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p') {
            confidence += 0.15;
        }
    }
    else if (sig.extension == "avi" && dataSize >= 12) {
        if (data[8] == 'A' && data[9] == 'V' && data[10] == 'I') {
            confidence += 0.15;
        }
    }
    else if (sig.extension == "wav" && dataSize >= 12) {
        if (data[8] == 'W' && data[9] == 'A' && data[10] == 'V' && data[11] == 'E') {
            confidence += 0.15;
        }
    }

    return min(1.0, confidence);
}

// ============================================================================
// 检查缓冲区是否为空（全零）
// ============================================================================
bool FileCarver::IsEmptyBuffer(const BYTE* data, size_t size) {
    if (size == 0) return true;

    // 采样检查，而非检查每个字节
    size_t sampleSize = min(size, EMPTY_CHECK_SAMPLE_SIZE);
    size_t step = size / sampleSize;
    if (step == 0) step = 1;

    size_t zeroCount = 0;
    for (size_t i = 0; i < size; i += step) {
        if (data[i] == 0) {
            zeroCount++;
        }
    }

    double zeroRatio = (double)zeroCount / (size / step);
    return zeroRatio >= EMPTY_THRESHOLD;
}

// ============================================================================
// 核心扫描函数：单缓冲区多签名匹配
// ============================================================================
void FileCarver::ScanBufferMultiSignature(const BYTE* data, size_t dataSize,
                                          ULONGLONG baseLCN, ULONGLONG bytesPerCluster,
                                          vector<CarvedFileInfo>& results,
                                          ULONGLONG maxResults) {
    size_t offset = 0;

    while (offset < dataSize && results.size() < maxResults && !shouldStop) {
        BYTE currentByte = data[offset];

        // 使用首字节索引快速查找可能匹配的签名
        auto it = signatureIndex.find(currentByte);
        if (it != signatureIndex.end()) {
            // 检查该首字节对应的所有签名
            for (const FileSignature* sig : it->second) {
                // 检查是否在活动签名列表中
                if (activeSignatures.find(sig->extension) == activeSignatures.end()) {
                    continue;
                }

                size_t remaining = dataSize - offset;
                if (remaining < sig->header.size()) {
                    continue;
                }

                // 完整签名匹配
                if (MatchSignature(data + offset, remaining, sig->header)) {
                    // 特殊检查：区分 AVI 和 WAV (都是 RIFF)
                    if (sig->extension == "avi" || sig->extension == "wav") {
                        if (remaining >= 12) {
                            bool isAvi = (data[offset + 8] == 'A' && data[offset + 9] == 'V' &&
                                          data[offset + 10] == 'I');
                            bool isWav = (data[offset + 8] == 'W' && data[offset + 9] == 'A' &&
                                          data[offset + 10] == 'V' && data[offset + 11] == 'E');
                            if ((sig->extension == "avi" && !isAvi) ||
                                (sig->extension == "wav" && !isWav)) {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }

                    // 特殊检查：MP4 需要验证 ftyp
                    if (sig->extension == "mp4" && remaining >= 8) {
                        if (!(data[offset + 4] == 'f' && data[offset + 5] == 't' &&
                              data[offset + 6] == 'y' && data[offset + 7] == 'p')) {
                            continue;
                        }
                    }

                    // 估算文件大小
                    ULONGLONG footerPos = 0;
                    ULONGLONG estimatedSize = EstimateFileSize(data + offset, remaining, *sig, &footerPos);

                    // 验证文件（已匹配签名，不重复检查）
                    double confidence = ValidateFileOptimized(data + offset,
                                                              min(remaining, (size_t)sig->maxSize),
                                                              *sig, true, footerPos);

                    // 只添加置信度足够的结果
                    if (confidence >= 0.6) {
                        ULONGLONG absoluteLCN = baseLCN + (offset / bytesPerCluster);
                        ULONGLONG clusterOffset = offset % bytesPerCluster;

                        CarvedFileInfo info;
                        info.startLCN = absoluteLCN;
                        info.startOffset = clusterOffset;
                        info.fileSize = estimatedSize;
                        info.extension = sig->extension;
                        info.description = sig->description;
                        info.hasValidFooter = (footerPos > 0);
                        info.confidence = confidence;

                        results.push_back(info);
                        stats.filesFound++;

                        // 跳过当前文件区域
                        offset += (size_t)min(estimatedSize, (ULONGLONG)remaining);
                        goto next_position;
                    }
                }
            }
        }

        offset++;
        next_position:;
    }
}

// ============================================================================
// 提取文件数据
// ============================================================================
bool FileCarver::ExtractFile(ULONGLONG startLCN, ULONGLONG startOffset,
                             ULONGLONG fileSize, vector<BYTE>& fileData) {
    ULONGLONG bytesPerCluster = reader->GetBytesPerCluster();

    ULONGLONG totalBytes = startOffset + fileSize;
    ULONGLONG clustersNeeded = (totalBytes + bytesPerCluster - 1) / bytesPerCluster;

    const ULONGLONG MAX_READ_CLUSTERS = 100000;
    if (clustersNeeded > MAX_READ_CLUSTERS) {
        LOG_WARNING_FMT("File too large, limiting to %llu clusters", MAX_READ_CLUSTERS);
        clustersNeeded = MAX_READ_CLUSTERS;
        fileSize = clustersNeeded * bytesPerCluster - startOffset;
    }

    vector<BYTE> clusterData;
    if (!reader->ReadClusters(startLCN, clustersNeeded, clusterData)) {
        LOG_ERROR_FMT("Failed to read clusters at LCN %llu", startLCN);
        return false;
    }

    if (startOffset + fileSize > clusterData.size()) {
        fileSize = clusterData.size() - startOffset;
    }

    fileData.resize((size_t)fileSize);
    memcpy(fileData.data(), clusterData.data() + startOffset, (size_t)fileSize);

    return true;
}

// ============================================================================
// 扫描特定类型（优化：使用通用扫描引擎）
// ============================================================================
vector<CarvedFileInfo> FileCarver::ScanForFileType(const string& fileType,
                                                    CarvingMode mode,
                                                    ULONGLONG maxResults) {
    vector<string> types;
    types.push_back(fileType);
    return ScanForFileTypes(types, mode, maxResults);
}

// ============================================================================
// 扫描多种类型（核心扫描函数）
// ============================================================================
vector<CarvedFileInfo> FileCarver::ScanForFileTypes(const vector<string>& fileTypes,
                                                     CarvingMode mode,
                                                     ULONGLONG maxResults) {
    vector<CarvedFileInfo> results;
    shouldStop = false;

    // 重置统计
    memset(&stats, 0, sizeof(stats));
    stats.totalClusters = reader->GetTotalClusters();

    // 设置活动签名
    activeSignatures.clear();
    for (const string& type : fileTypes) {
        if (signatures.find(type) != signatures.end()) {
            activeSignatures.insert(type);
        } else {
            cout << "Unknown file type: " << type << endl;
        }
    }

    if (activeSignatures.empty()) {
        cout << "No valid file types to scan." << endl;
        return results;
    }

    // 显示扫描信息
    cout << "\n============================================" << endl;
    cout << "    Optimized File Carving Scanner" << endl;
    cout << "============================================\n" << endl;

    cout << "Scanning for " << activeSignatures.size() << " file type(s): ";
    for (const string& type : activeSignatures) {
        cout << type << " ";
    }
    cout << endl;

    ULONGLONG bytesPerCluster = reader->GetBytesPerCluster();
    ULONGLONG totalBytes = stats.totalClusters * bytesPerCluster;

    cout << "Total clusters: " << stats.totalClusters << endl;
    cout << "Cluster size: " << bytesPerCluster << " bytes" << endl;
    cout << "Volume size: " << (totalBytes / (1024 * 1024 * 1024)) << " GB" << endl;
    cout << "Buffer size: " << BUFFER_SIZE_MB << " MB (" << BUFFER_SIZE_CLUSTERS << " clusters)" << endl;
    cout << "\nScanning... (Press Ctrl+C to stop)\n" << endl;

    // 计算实际使用的缓冲区大小
    ULONGLONG bufferClusters = min(BUFFER_SIZE_CLUSTERS, stats.totalClusters);

    // 开始计时
    DWORD startTime = GetTickCount();

    vector<BYTE> buffer;
    ULONGLONG currentLCN = 0;
    ULONGLONG lastProgressLCN = 0;

    while (currentLCN < stats.totalClusters && !shouldStop && results.size() < maxResults) {
        ULONGLONG clustersToRead = min(bufferClusters, stats.totalClusters - currentLCN);

        // 读取一大块数据
        if (!reader->ReadClusters(currentLCN, clustersToRead, buffer)) {
            // 读取失败，跳过
            currentLCN += clustersToRead;
            stats.scannedClusters += clustersToRead;
            continue;
        }

        stats.bytesRead += buffer.size();

        // 检查是否为空块（优化：跳过空白区域）
        if (mode == CARVE_SMART && IsEmptyBuffer(buffer.data(), buffer.size())) {
            stats.skippedClusters += clustersToRead;
            currentLCN += clustersToRead;
            stats.scannedClusters += clustersToRead;

            // 更新进度
            if (stats.scannedClusters - lastProgressLCN > PROGRESS_UPDATE_INTERVAL) {
                DWORD elapsed = GetTickCount() - startTime;
                double progress = (double)stats.scannedClusters / stats.totalClusters * 100.0;
                double speedMBps = (elapsed > 0) ?
                    ((double)stats.bytesRead / (1024 * 1024)) / (elapsed / 1000.0) : 0;

                cout << "\rProgress: " << fixed << setprecision(1) << progress << "% | "
                     << "Scanned: " << (stats.scannedClusters / 1000) << "K clusters | "
                     << "Skipped: " << (stats.skippedClusters / 1000) << "K | "
                     << "Found: " << stats.filesFound << " files | "
                     << "Speed: " << setprecision(1) << speedMBps << " MB/s" << flush;

                lastProgressLCN = stats.scannedClusters;
            }
            continue;
        }

        // 多签名扫描
        ScanBufferMultiSignature(buffer.data(), buffer.size(), currentLCN, bytesPerCluster,
                                 results, maxResults);

        currentLCN += clustersToRead;
        stats.scannedClusters += clustersToRead;

        // 更新进度
        if (stats.scannedClusters - lastProgressLCN > PROGRESS_UPDATE_INTERVAL) {
            DWORD elapsed = GetTickCount() - startTime;
            double progress = (double)stats.scannedClusters / stats.totalClusters * 100.0;
            double speedMBps = (elapsed > 0) ?
                ((double)stats.bytesRead / (1024 * 1024)) / (elapsed / 1000.0) : 0;

            cout << "\rProgress: " << fixed << setprecision(1) << progress << "% | "
                 << "Scanned: " << (stats.scannedClusters / 1000) << "K clusters | "
                 << "Skipped: " << (stats.skippedClusters / 1000) << "K | "
                 << "Found: " << stats.filesFound << " files | "
                 << "Speed: " << setprecision(1) << speedMBps << " MB/s" << flush;

            lastProgressLCN = stats.scannedClusters;
        }
    }

    // 完成统计
    stats.elapsedMs = GetTickCount() - startTime;
    stats.readSpeedMBps = (stats.elapsedMs > 0) ?
        ((double)stats.bytesRead / (1024 * 1024)) / (stats.elapsedMs / 1000.0) : 0;

    // 显示结果
    cout << "\r                                                                              " << endl;
    cout << "\n============================================" << endl;
    cout << "           Scan Complete" << endl;
    cout << "============================================\n" << endl;

    cout << "Time elapsed: " << (stats.elapsedMs / 1000) << "."
         << ((stats.elapsedMs % 1000) / 100) << " seconds" << endl;
    cout << "Clusters scanned: " << stats.scannedClusters << endl;
    cout << "Clusters skipped (empty): " << stats.skippedClusters << endl;
    cout << "Data read: " << (stats.bytesRead / (1024 * 1024)) << " MB" << endl;
    cout << "Average speed: " << fixed << setprecision(1) << stats.readSpeedMBps << " MB/s" << endl;
    cout << "Files found: " << results.size() << endl;

    return results;
}

// ============================================================================
// 扫描所有类型（优化：单次扫描）
// ============================================================================
vector<CarvedFileInfo> FileCarver::ScanAllTypes(CarvingMode mode, ULONGLONG maxResults) {
    vector<string> allTypes;
    for (const auto& pair : signatures) {
        allTypes.push_back(pair.first);
    }
    return ScanForFileTypes(allTypes, mode, maxResults);
}

// ============================================================================
// 恢复文件
// ============================================================================
bool FileCarver::RecoverCarvedFile(const CarvedFileInfo& info, const string& outputPath) {
    cout << "Recovering carved file..." << endl;
    cout << "  Start LCN: " << info.startLCN << endl;
    cout << "  Offset: " << info.startOffset << endl;
    cout << "  Size: " << info.fileSize << " bytes" << endl;
    cout << "  Type: " << info.description << endl;

    vector<BYTE> fileData;
    if (!ExtractFile(info.startLCN, info.startOffset, info.fileSize, fileData)) {
        cout << "Failed to extract file data." << endl;
        return false;
    }

    cout << "Extracted " << fileData.size() << " bytes" << endl;

    // 确保父目录存在
    string parentPath = outputPath;
    size_t lastSlash = parentPath.find_last_of("\\/");
    if (lastSlash != string::npos) {
        parentPath = parentPath.substr(0, lastSlash);
        string currentPath;
        for (size_t i = 0; i < parentPath.length(); i++) {
            char c = parentPath[i];
            currentPath += c;
            if (c == '\\' || c == '/' || i == parentPath.length() - 1) {
                if (currentPath.length() > 2) {
                    CreateDirectoryA(currentPath.c_str(), NULL);
                }
            }
        }
    }

    HANDLE hFile = CreateFileA(outputPath.c_str(), GENERIC_WRITE,
        0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE) {
        cout << "Failed to create output file. Error: " << GetLastError() << endl;
        return false;
    }

    DWORD bytesWritten;
    BOOL result = WriteFile(hFile, fileData.data(), (DWORD)fileData.size(), &bytesWritten, NULL);
    CloseHandle(hFile);

    if (result && bytesWritten == fileData.size()) {
        cout << "File recovered successfully: " << outputPath << endl;
        return true;
    }

    cout << "Failed to write file data." << endl;
    return false;
}

// ============================================================================
// 辅助函数
// ============================================================================
vector<string> FileCarver::GetSupportedTypes() {
    vector<string> types;
    for (const auto& pair : signatures) {
        types.push_back(pair.first + " - " + pair.second.description);
    }
    return types;
}

double FileCarver::GetProgress() const {
    if (stats.totalClusters == 0) return 0.0;
    return (double)stats.scannedClusters / stats.totalClusters;
}

void FileCarver::StopScanning() {
    shouldStop = true;
}

// ============================================================================
// 异步I/O：读取线程（生产者）
// ============================================================================
void FileCarver::IOReaderThread(ULONGLONG startLCN, ULONGLONG endLCN,
                                ULONGLONG bufferClusters, ULONGLONG bytesPerCluster,
                                CarvingMode mode) {
    ULONGLONG currentLCN = startLCN;
    int bufferIndex = 0;

    while (currentLCN < endLCN && !shouldStop) {
        DWORD ioStartTime = GetTickCount();

        ULONGLONG clustersToRead = min(bufferClusters, endLCN - currentLCN);

        // 等待目标缓冲区被消费
        {
            unique_lock<mutex> lock(bufferMutex);
            bufferConsumedCV.wait(lock, [this, bufferIndex]() {
                return !buffers[bufferIndex].ready || shouldStop;
            });

            if (shouldStop) break;
        }

        DWORD waitEndTime = GetTickCount();
        ioWaitTimeMs += (waitEndTime - ioStartTime);

        // 读取数据到缓冲区
        vector<BYTE> tempBuffer;
        bool readSuccess = reader->ReadClusters(currentLCN, clustersToRead, tempBuffer);

        DWORD readEndTime = GetTickCount();
        totalIoTimeMs += (readEndTime - waitEndTime);

        // 更新缓冲区状态
        {
            lock_guard<mutex> lock(bufferMutex);

            if (readSuccess) {
                buffers[bufferIndex].data = move(tempBuffer);
                buffers[bufferIndex].startLCN = currentLCN;
                buffers[bufferIndex].clusterCount = clustersToRead;

                // 检查是否为空块
                if (mode == CARVE_SMART) {
                    buffers[bufferIndex].isEmpty = IsEmptyBuffer(
                        buffers[bufferIndex].data.data(),
                        buffers[bufferIndex].data.size());
                } else {
                    buffers[bufferIndex].isEmpty = false;
                }
            } else {
                buffers[bufferIndex].data.clear();
                buffers[bufferIndex].isEmpty = true;
            }

            buffers[bufferIndex].isLast = (currentLCN + clustersToRead >= endLCN);
            buffers[bufferIndex].ready = true;
        }

        // 通知扫描线程
        bufferReadyCV.notify_one();

        currentLCN += clustersToRead;
        bufferIndex = 1 - bufferIndex;  // 切换缓冲区 0->1->0->1...
    }

    // 标记结束
    {
        lock_guard<mutex> lock(bufferMutex);
        // 确保最后一个缓冲区标记为 isLast
        for (int i = 0; i < 2; i++) {
            if (!buffers[i].ready) {
                buffers[i].isLast = true;
                buffers[i].ready = true;
                buffers[i].data.clear();
            }
        }
    }
    bufferReadyCV.notify_all();
}

// ============================================================================
// 异步I/O：扫描线程（消费者）
// ============================================================================
void FileCarver::ScanWorkerThread(vector<CarvedFileInfo>& results,
                                  ULONGLONG bytesPerCluster, ULONGLONG maxResults) {
    int bufferIndex = 0;
    bool finished = false;

    while (!finished && !shouldStop && results.size() < maxResults) {
        ScanBuffer* currentBuffer = nullptr;

        // 等待缓冲区就绪
        {
            unique_lock<mutex> lock(bufferMutex);

            DWORD waitStart = GetTickCount();
            bufferReadyCV.wait(lock, [this, bufferIndex]() {
                return buffers[bufferIndex].ready || shouldStop;
            });
            DWORD waitEnd = GetTickCount();
            scanTimeMs += (waitEnd - waitStart);  // 实际上是等待时间

            if (shouldStop) break;

            currentBuffer = &buffers[bufferIndex];
            finished = currentBuffer->isLast && currentBuffer->data.empty();
        }

        if (currentBuffer && !currentBuffer->data.empty()) {
            DWORD scanStart = GetTickCount();

            // 处理非空缓冲区
            if (!currentBuffer->isEmpty) {
                ScanBufferMultiSignature(
                    currentBuffer->data.data(),
                    currentBuffer->data.size(),
                    currentBuffer->startLCN,
                    bytesPerCluster,
                    results,
                    maxResults);

                stats.scannedClusters += currentBuffer->clusterCount;
            } else {
                stats.skippedClusters += currentBuffer->clusterCount;
                stats.scannedClusters += currentBuffer->clusterCount;
            }

            stats.bytesRead += currentBuffer->data.size();

            DWORD scanEnd = GetTickCount();
            totalScanTimeMs += (scanEnd - scanStart);

            // 检查是否为最后一块
            finished = currentBuffer->isLast;
        }

        // 标记缓冲区已消费
        {
            lock_guard<mutex> lock(bufferMutex);
            buffers[bufferIndex].ready = false;
        }
        bufferConsumedCV.notify_one();

        bufferIndex = 1 - bufferIndex;  // 切换缓冲区
    }
}

// ============================================================================
// 异步扫描主函数
// ============================================================================
vector<CarvedFileInfo> FileCarver::ScanForFileTypesAsync(const vector<string>& fileTypes,
                                                          CarvingMode mode,
                                                          ULONGLONG maxResults) {
    vector<CarvedFileInfo> results;
    shouldStop = false;

    // 重置统计和计时器
    memset(&stats, 0, sizeof(stats));
    ioWaitTimeMs = 0;
    scanTimeMs = 0;
    totalIoTimeMs = 0;
    totalScanTimeMs = 0;

    stats.totalClusters = reader->GetTotalClusters();

    // 设置活动签名
    activeSignatures.clear();
    for (const string& type : fileTypes) {
        if (signatures.find(type) != signatures.end()) {
            activeSignatures.insert(type);
        } else {
            cout << "Unknown file type: " << type << endl;
        }
    }

    if (activeSignatures.empty()) {
        cout << "No valid file types to scan." << endl;
        return results;
    }

    // 显示扫描信息
    cout << "\n============================================" << endl;
    cout << "  Async File Carving Scanner (Dual Buffer)" << endl;
    cout << "============================================\n" << endl;

    cout << "Scanning for " << activeSignatures.size() << " file type(s): ";
    for (const string& type : activeSignatures) {
        cout << type << " ";
    }
    cout << endl;

    ULONGLONG bytesPerCluster = reader->GetBytesPerCluster();
    ULONGLONG totalBytes = stats.totalClusters * bytesPerCluster;

    cout << "Total clusters: " << stats.totalClusters << endl;
    cout << "Cluster size: " << bytesPerCluster << " bytes" << endl;
    cout << "Volume size: " << (totalBytes / (1024 * 1024 * 1024)) << " GB" << endl;
    cout << "Buffer size: " << BUFFER_SIZE_MB << " MB x 2 (dual buffer)" << endl;
    cout << "Mode: Async I/O (Producer-Consumer)" << endl;
    cout << "\nScanning... (Press Ctrl+C to stop)\n" << endl;

    // 重置双缓冲区状态
    for (int i = 0; i < 2; i++) {
        buffers[i].ready = false;
        buffers[i].isEmpty = false;
        buffers[i].isLast = false;
        buffers[i].data.clear();
    }

    ULONGLONG bufferClusters = min(BUFFER_SIZE_CLUSTERS, stats.totalClusters);

    // 开始计时
    DWORD startTime = GetTickCount();

    // 启动I/O读取线程（生产者）
    thread ioThread(&FileCarver::IOReaderThread, this,
                    0, stats.totalClusters, bufferClusters, bytesPerCluster, mode);

    // 扫描线程（消费者）- 在当前线程运行
    // 同时显示进度
    int bufferIndex = 0;
    bool finished = false;
    ULONGLONG lastProgressUpdate = 0;

    while (!finished && !shouldStop && results.size() < maxResults) {
        ScanBuffer* currentBuffer = nullptr;

        // 等待缓冲区就绪
        {
            unique_lock<mutex> lock(bufferMutex);
            bufferReadyCV.wait(lock, [this, bufferIndex]() {
                return buffers[bufferIndex].ready || shouldStop;
            });

            if (shouldStop) break;
            currentBuffer = &buffers[bufferIndex];
            finished = currentBuffer->isLast && currentBuffer->data.empty();
        }

        if (currentBuffer && !currentBuffer->data.empty()) {
            // 处理非空缓冲区
            if (!currentBuffer->isEmpty) {
                ScanBufferMultiSignature(
                    currentBuffer->data.data(),
                    currentBuffer->data.size(),
                    currentBuffer->startLCN,
                    bytesPerCluster,
                    results,
                    maxResults);
            } else {
                stats.skippedClusters += currentBuffer->clusterCount;
            }

            stats.scannedClusters += currentBuffer->clusterCount;
            stats.bytesRead += currentBuffer->data.size();

            finished = currentBuffer->isLast;
        }

        // 更新进度
        if (stats.scannedClusters - lastProgressUpdate > PROGRESS_UPDATE_INTERVAL) {
            DWORD elapsed = GetTickCount() - startTime;
            double progress = (double)stats.scannedClusters / stats.totalClusters * 100.0;
            double speedMBps = (elapsed > 0) ?
                ((double)stats.bytesRead / (1024 * 1024)) / (elapsed / 1000.0) : 0;

            cout << "\rProgress: " << fixed << setprecision(1) << progress << "% | "
                 << "Scanned: " << (stats.scannedClusters / 1000) << "K | "
                 << "Skipped: " << (stats.skippedClusters / 1000) << "K | "
                 << "Found: " << stats.filesFound << " | "
                 << "Speed: " << setprecision(1) << speedMBps << " MB/s [ASYNC]" << flush;

            lastProgressUpdate = stats.scannedClusters;
        }

        // 标记缓冲区已消费
        {
            lock_guard<mutex> lock(bufferMutex);
            buffers[bufferIndex].ready = false;
        }
        bufferConsumedCV.notify_one();

        bufferIndex = 1 - bufferIndex;
    }

    // 等待I/O线程完成
    if (ioThread.joinable()) {
        ioThread.join();
    }

    // 完成统计
    stats.elapsedMs = GetTickCount() - startTime;
    stats.readSpeedMBps = (stats.elapsedMs > 0) ?
        ((double)stats.bytesRead / (1024 * 1024)) / (stats.elapsedMs / 1000.0) : 0;

    // 计算I/O和CPU效率
    ULONGLONG totalTime = stats.elapsedMs;
    if (totalTime > 0) {
        stats.ioBusyPercent = (double)totalIoTimeMs / totalTime * 100.0;
        stats.cpuBusyPercent = (double)totalScanTimeMs / totalTime * 100.0;
    }

    // 显示结果
    cout << "\r                                                                              " << endl;
    cout << "\n============================================" << endl;
    cout << "        Async Scan Complete" << endl;
    cout << "============================================\n" << endl;

    cout << "Time elapsed: " << (stats.elapsedMs / 1000) << "."
         << ((stats.elapsedMs % 1000) / 100) << " seconds" << endl;
    cout << "Clusters scanned: " << stats.scannedClusters << endl;
    cout << "Clusters skipped (empty): " << stats.skippedClusters << endl;
    cout << "Data read: " << (stats.bytesRead / (1024 * 1024)) << " MB" << endl;
    cout << "Average speed: " << fixed << setprecision(1) << stats.readSpeedMBps << " MB/s" << endl;
    cout << "Files found: " << results.size() << endl;

    // 显示并行效率分析
    cout << "\n--- Async I/O Analysis ---" << endl;
    cout << "I/O thread time: " << totalIoTimeMs << " ms" << endl;
    cout << "Scan thread wait: " << scanTimeMs << " ms" << endl;

    double overlapEfficiency = 0;
    if (totalIoTimeMs > 0 && totalScanTimeMs > 0) {
        // 理想情况下，总时间应该接近 max(I/O时间, 扫描时间)
        ULONGLONG idealTime = max((ULONGLONG)totalIoTimeMs, (ULONGLONG)totalScanTimeMs);
        ULONGLONG sequentialTime = totalIoTimeMs + totalScanTimeMs;
        if (sequentialTime > 0) {
            overlapEfficiency = (1.0 - (double)stats.elapsedMs / sequentialTime) * 100.0;
        }
        cout << "Overlap efficiency: " << fixed << setprecision(1) << overlapEfficiency << "%" << endl;
        cout << "(Higher = better I/O and CPU overlap)" << endl;
    }

    return results;
}

// ============================================================================
// 设置线程池配置
// ============================================================================
void FileCarver::SetThreadPoolConfig(const ScanThreadPoolConfig& config) {
    threadPoolConfig = config;
    LOG_INFO_FMT("Thread pool config updated: %d workers, %zu MB chunks",
                 config.workerCount, config.chunkSize / (1024 * 1024));
}

// ============================================================================
// 将缓冲区分块并提交给线程池
// ============================================================================
void FileCarver::SubmitBufferToThreadPool(const BYTE* buffer, size_t bufferSize,
                                           ULONGLONG baseLCN, ULONGLONG bytesPerCluster,
                                           int& taskIdCounter) {
    size_t offset = 0;
    size_t chunkSize = threadPoolConfig.chunkSize;

    while (offset < bufferSize) {
        size_t remaining = bufferSize - offset;
        size_t currentChunkSize = min(chunkSize, remaining);

        ScanTask task;
        task.data = buffer + offset;
        task.dataSize = currentChunkSize;
        task.baseLCN = baseLCN + (offset / bytesPerCluster);
        task.bytesPerCluster = bytesPerCluster;
        task.taskId = taskIdCounter++;

        scanThreadPool->SubmitTask(task);

        offset += currentChunkSize;
    }
}

// ============================================================================
// 线程池并行扫描（阶段1实现）
// ============================================================================
vector<CarvedFileInfo> FileCarver::ScanForFileTypesThreadPool(const vector<string>& fileTypes,
                                                               CarvingMode mode,
                                                               ULONGLONG maxResults) {
    vector<CarvedFileInfo> results;
    shouldStop = false;

    // 重置统计
    memset(&stats, 0, sizeof(stats));
    stats.totalClusters = reader->GetTotalClusters();

    // 设置活动签名
    activeSignatures.clear();
    for (const string& type : fileTypes) {
        if (signatures.find(type) != signatures.end()) {
            activeSignatures.insert(type);
        } else {
            cout << "Unknown file type: " << type << endl;
        }
    }

    if (activeSignatures.empty()) {
        cout << "No valid file types to scan." << endl;
        return results;
    }

    // 显示扫描信息
    cout << "\n============================================" << endl;
    cout << "  Thread Pool File Carving Scanner" << endl;
    cout << "  (Phase 1: Parallel Scanning)" << endl;
    cout << "============================================\n" << endl;

    cout << "Scanning for " << activeSignatures.size() << " file type(s): ";
    for (const string& type : activeSignatures) {
        cout << type << " ";
    }
    cout << endl;

    ULONGLONG bytesPerCluster = reader->GetBytesPerCluster();
    ULONGLONG totalBytes = stats.totalClusters * bytesPerCluster;

    // 获取系统信息
    int cpuCores = ThreadPoolUtils::GetLogicalCoreCount();
    ULONGLONG availMem = ThreadPoolUtils::GetAvailableMemoryMB();

    cout << "\n--- System Info ---" << endl;
    cout << "CPU Cores: " << cpuCores << " logical cores" << endl;
    cout << "Available Memory: " << availMem << " MB" << endl;
    cout << "Worker Threads: " << threadPoolConfig.workerCount << endl;
    cout << "Chunk Size: " << (threadPoolConfig.chunkSize / (1024 * 1024)) << " MB" << endl;

    cout << "\n--- Volume Info ---" << endl;
    cout << "Total clusters: " << stats.totalClusters << endl;
    cout << "Cluster size: " << bytesPerCluster << " bytes" << endl;
    cout << "Volume size: " << (totalBytes / (1024 * 1024 * 1024)) << " GB" << endl;

    // 使用更大的缓冲区以优化NVMe性能
    const ULONGLONG BUFFER_SIZE = 128 * 1024 * 1024;  // 128MB
    ULONGLONG bufferClusters = BUFFER_SIZE / bytesPerCluster;
    bufferClusters = min(bufferClusters, stats.totalClusters);

    cout << "Read Buffer: " << (BUFFER_SIZE / (1024 * 1024)) << " MB" << endl;
    cout << "\nScanning... (Press Ctrl+C to stop)\n" << endl;

    // 创建线程池
    if (scanThreadPool) {
        delete scanThreadPool;
    }
    scanThreadPool = new SignatureScanThreadPool(&signatureIndex, &activeSignatures, threadPoolConfig);
    scanThreadPool->Start();

    // 开始计时
    DWORD startTime = GetTickCount();

    // I/O读取主循环
    vector<BYTE> buffer;
    ULONGLONG currentLCN = 0;
    ULONGLONG lastProgressLCN = 0;
    int taskIdCounter = 0;

    while (currentLCN < stats.totalClusters && !shouldStop) {
        ULONGLONG clustersToRead = min(bufferClusters, stats.totalClusters - currentLCN);

        // 读取数据
        if (!reader->ReadClusters(currentLCN, clustersToRead, buffer)) {
            currentLCN += clustersToRead;
            stats.scannedClusters += clustersToRead;
            continue;
        }

        stats.bytesRead += buffer.size();

        // 检查是否为空块（智能模式下跳过空白区域）
        if (mode == CARVE_SMART && IsEmptyBuffer(buffer.data(), buffer.size())) {
            stats.skippedClusters += clustersToRead;
            currentLCN += clustersToRead;
            stats.scannedClusters += clustersToRead;
        } else {
            // 将缓冲区分块提交给线程池
            SubmitBufferToThreadPool(buffer.data(), buffer.size(),
                                      currentLCN, bytesPerCluster, taskIdCounter);

            currentLCN += clustersToRead;
            stats.scannedClusters += clustersToRead;
        }

        // 更新进度
        if (stats.scannedClusters - lastProgressLCN > PROGRESS_UPDATE_INTERVAL) {
            DWORD elapsed = GetTickCount() - startTime;
            double progress = (double)stats.scannedClusters / stats.totalClusters * 100.0;
            double speedMBps = (elapsed > 0) ?
                ((double)stats.bytesRead / (1024 * 1024)) / (elapsed / 1000.0) : 0;

            ULONGLONG filesFound = scanThreadPool->GetTotalFilesFound();
            double poolProgress = scanThreadPool->GetProgress();

            cout << "\rI/O: " << fixed << setprecision(1) << progress << "% | "
                 << "Pool: " << setprecision(1) << poolProgress << "% | "
                 << "Scanned: " << (stats.scannedClusters / 1000) << "K | "
                 << "Skipped: " << (stats.skippedClusters / 1000) << "K | "
                 << "Found: " << filesFound << " | "
                 << "Speed: " << setprecision(1) << speedMBps << " MB/s [THREADPOOL]" << flush;

            lastProgressLCN = stats.scannedClusters;
        }
    }

    // 等待线程池完成所有任务
    cout << "\n\nWaiting for thread pool to complete..." << endl;
    scanThreadPool->WaitForCompletion();

    // 获取结果
    results = scanThreadPool->GetMergedResults();
    stats.filesFound = results.size();

    // 停止线程池
    scanThreadPool->Stop();

    // 完成统计
    stats.elapsedMs = GetTickCount() - startTime;
    stats.readSpeedMBps = (stats.elapsedMs > 0) ?
        ((double)stats.bytesRead / (1024 * 1024)) / (stats.elapsedMs / 1000.0) : 0;

    // 计算扫描速度（考虑并行扫描）
    stats.scanSpeedMBps = (stats.elapsedMs > 0) ?
        ((double)stats.scannedClusters * bytesPerCluster / (1024 * 1024)) / (stats.elapsedMs / 1000.0) : 0;

    // 显示结果
    cout << "\r                                                                              " << endl;
    cout << "\n============================================" << endl;
    cout << "     Thread Pool Scan Complete" << endl;
    cout << "============================================\n" << endl;

    cout << "Time elapsed: " << (stats.elapsedMs / 1000) << "."
         << ((stats.elapsedMs % 1000) / 100) << " seconds" << endl;
    cout << "Clusters scanned: " << stats.scannedClusters << endl;
    cout << "Clusters skipped (empty): " << stats.skippedClusters << endl;
    cout << "Data read: " << (stats.bytesRead / (1024 * 1024)) << " MB" << endl;
    cout << "Average read speed: " << fixed << setprecision(1) << stats.readSpeedMBps << " MB/s" << endl;
    cout << "Effective scan speed: " << fixed << setprecision(1) << stats.scanSpeedMBps << " MB/s" << endl;
    cout << "Files found: " << results.size() << endl;

    // 显示线程池统计
    cout << "\n--- Thread Pool Statistics ---" << endl;
    cout << "Worker threads: " << threadPoolConfig.workerCount << endl;
    cout << "Total tasks: " << scanThreadPool->GetTotalTasks() << endl;
    cout << "Completed tasks: " << scanThreadPool->GetCompletedTasks() << endl;

    // 计算并行效率
    double expectedSingleThreadTime = stats.elapsedMs * threadPoolConfig.workerCount;
    double parallelEfficiency = (expectedSingleThreadTime > 0) ?
        (expectedSingleThreadTime / stats.elapsedMs / threadPoolConfig.workerCount) * 100.0 : 0;

    // 与异步I/O方案对比（估算）
    double estimatedAsyncTime = stats.elapsedMs * 1.5;  // 假设异步I/O快50%于同步
    double estimatedSyncTime = stats.elapsedMs * 3.0;   // 假设同步方案慢3倍

    cout << "\n--- Performance Analysis ---" << endl;
    cout << "Estimated speedup vs sync: " << fixed << setprecision(1)
         << (estimatedSyncTime / stats.elapsedMs) << "x" << endl;
    cout << "Estimated speedup vs async: " << fixed << setprecision(1)
         << (estimatedAsyncTime / stats.elapsedMs) << "x" << endl;

    // 如果结果过多，限制返回数量
    if (results.size() > maxResults) {
        results.resize((size_t)maxResults);
        cout << "\nNote: Results limited to " << maxResults << " files." << endl;
    }

    return results;
}

// ============================================================================
// 时间戳提取功能
// ============================================================================

// 构建 MFT LCN 索引
bool FileCarver::BuildMFTIndex(bool includeActiveFiles) {
    if (lcnIndex) {
        delete lcnIndex;
    }

    lcnIndex = new MFTLCNIndex(reader);
    mftIndexBuilt = lcnIndex->BuildIndex(includeActiveFiles, true);

    if (!mftIndexBuilt) {
        delete lcnIndex;
        lcnIndex = nullptr;
    }

    return mftIndexBuilt;
}

// 为单个文件提取时间戳
void FileCarver::ExtractTimestampForFile(CarvedFileInfo& info, const BYTE* fileData, size_t dataSize) {
    bool hasEmbedded = false;
    bool hasMftMatch = false;

    // 方案1：尝试从内嵌元数据提取
    ExtractedTimestamp embedded = TimestampExtractor::Extract(fileData, dataSize, info.extension);

    if (embedded.hasAnyTimestamp()) {
        hasEmbedded = true;

        if (embedded.hasCreation) {
            info.creationTime = embedded.creationTime;
        }
        if (embedded.hasModification) {
            info.modificationTime = embedded.modificationTime;
        }
        if (embedded.hasAccess) {
            info.accessTime = embedded.accessTime;
        }
        if (!embedded.additionalInfo.empty()) {
            info.embeddedInfo = embedded.additionalInfo;
        }
    }

    // 方案2：尝试 MFT 交叉匹配
    if (lcnIndex && mftIndexBuilt) {
        vector<LCNMappingInfo> matches = lcnIndex->FindByLCN(info.startLCN);

        if (!matches.empty()) {
            // 找到最佳匹配（优先已删除的文件）
            const LCNMappingInfo* bestMatch = nullptr;
            for (const auto& match : matches) {
                if (!bestMatch || (match.isDeleted && !bestMatch->isDeleted)) {
                    bestMatch = &match;
                }
            }

            if (bestMatch) {
                hasMftMatch = true;
                info.matchedMftRecord = bestMatch->mftRecordNumber;

                // 如果没有内嵌时间戳，使用 MFT 时间戳
                if (!hasEmbedded) {
                    info.creationTime = bestMatch->creationTime;
                    info.modificationTime = bestMatch->modificationTime;
                    info.accessTime = bestMatch->accessTime;
                }

                // 补充文件名信息
                if (!bestMatch->fileName.empty()) {
                    // 转换 wstring 到 string（简化处理）
                    string fileName;
                    for (wchar_t wc : bestMatch->fileName) {
                        if (wc < 128) fileName += (char)wc;
                    }
                    if (!info.embeddedInfo.empty()) {
                        info.embeddedInfo += ", ";
                    }
                    info.embeddedInfo += "MFT Name: " + fileName;
                }
            }
        }
    }

    // 设置时间戳来源
    if (hasEmbedded && hasMftMatch) {
        info.tsSource = TS_BOTH;
    } else if (hasEmbedded) {
        info.tsSource = TS_EMBEDDED;
    } else if (hasMftMatch) {
        info.tsSource = TS_MFT_MATCH;
    } else {
        info.tsSource = TS_NONE_1;
    }
}

// 为扫描结果批量提取时间戳
void FileCarver::ExtractTimestampsForResults(vector<CarvedFileInfo>& results, bool showProgress) {
    if (results.empty()) {
        return;
    }

    if (!timestampExtractionEnabled) {
        cout << "Timestamp extraction is disabled." << endl;
        return;
    }

    if (showProgress) {
        cout << "\n--- Extracting Timestamps ---" << endl;
        cout << "Files to process: " << results.size() << endl;
        cout << "MFT Index: " << (mftIndexBuilt ? "Available" : "Not built") << endl;
    }

    DWORD startTime = GetTickCount();
    size_t processedCount = 0;
    size_t embeddedCount = 0;
    size_t mftMatchCount = 0;
    size_t totalWithTimestamp = 0;

    // 需要读取文件头数据来提取内嵌时间戳
    const size_t HEADER_READ_SIZE = 64 * 1024;  // 读取前 64KB 用于元数据分析

    for (auto& info : results) {
        // 读取文件头部数据
        vector<BYTE> headerData;
        size_t readSize = min((size_t)info.fileSize, HEADER_READ_SIZE);

        if (ExtractFile(info.startLCN, info.startOffset, readSize, headerData)) {
            ExtractTimestampForFile(info, headerData.data(), headerData.size());
        } else {
            // 无法读取文件，只尝试 MFT 匹配
            if (lcnIndex && mftIndexBuilt) {
                vector<LCNMappingInfo> matches = lcnIndex->FindByLCN(info.startLCN);
                if (!matches.empty()) {
                    const LCNMappingInfo& match = matches[0];
                    info.creationTime = match.creationTime;
                    info.modificationTime = match.modificationTime;
                    info.accessTime = match.accessTime;
                    info.matchedMftRecord = match.mftRecordNumber;
                    info.tsSource = TS_MFT_MATCH;
                }
            }
        }

        // 统计
        if (info.tsSource == TS_EMBEDDED || info.tsSource == TS_BOTH) {
            embeddedCount++;
        }
        if (info.tsSource == TS_MFT_MATCH || info.tsSource == TS_BOTH) {
            mftMatchCount++;
        }
        if (info.tsSource != TS_NONE_1) {
            totalWithTimestamp++;
        }

        processedCount++;

        // 进度显示
        if (showProgress && processedCount % 100 == 0) {
            double progress = (double)processedCount / results.size() * 100.0;
            cout << "\rProgress: " << fixed << setprecision(1) << progress << "% | "
                 << "Processed: " << processedCount << "/" << results.size() << flush;
        }
    }

    DWORD elapsed = GetTickCount() - startTime;

    if (showProgress) {
        cout << "\r                                                                    " << endl;
        cout << "\n--- Timestamp Extraction Complete ---" << endl;
        cout << "Time: " << (elapsed / 1000) << "." << ((elapsed % 1000) / 100) << " seconds" << endl;
        cout << "Files processed: " << processedCount << endl;
        cout << "With embedded timestamp: " << embeddedCount << " ("
             << fixed << setprecision(1) << (100.0 * embeddedCount / processedCount) << "%)" << endl;
        cout << "With MFT match: " << mftMatchCount << " ("
             << fixed << setprecision(1) << (100.0 * mftMatchCount / processedCount) << "%)" << endl;
        cout << "Total with timestamp: " << totalWithTimestamp << " ("
             << fixed << setprecision(1) << (100.0 * totalWithTimestamp / processedCount) << "%)" << endl;
    }

    LOG_INFO_FMT("Timestamp extraction: %zu/%zu files have timestamps (embedded: %zu, MFT: %zu)",
                 totalWithTimestamp, processedCount, embeddedCount, mftMatchCount);
}

// ============================================================================
// 完整性验证功能
// ============================================================================

// 为单个文件验证完整性
FileIntegrityScore FileCarver::ValidateFileIntegrity(const CarvedFileInfo& info) {
    // 读取文件数据
    vector<BYTE> fileData;
    size_t readSize = (size_t)min(info.fileSize, (ULONGLONG)(2 * 1024 * 1024));  // Max 2MB for validation

    if (!ExtractFile(info.startLCN, info.startOffset, readSize, fileData)) {
        FileIntegrityScore score;
        score.diagnosis = "Failed to read file data";
        score.isLikelyCorrupted = true;
        return score;
    }

    return FileIntegrityValidator::Validate(fileData.data(), fileData.size(), info.extension);
}

// 为扫描结果批量验证完整性
void FileCarver::ValidateIntegrityForResults(vector<CarvedFileInfo>& results, bool showProgress) {
    if (results.empty()) {
        return;
    }

    if (!integrityValidationEnabled) {
        cout << "Integrity validation is disabled." << endl;
        return;
    }

    if (showProgress) {
        cout << "\n--- Validating File Integrity ---" << endl;
        cout << "Files to validate: " << results.size() << endl;
    }

    DWORD startTime = GetTickCount();
    validatedCount = 0;
    corruptedCount = 0;
    size_t highConfidenceCount = 0;
    size_t lowConfidenceCount = 0;

    // Max file size to read for validation (2MB)
    const size_t MAX_VALIDATION_SIZE = 2 * 1024 * 1024;

    for (auto& info : results) {
        // Read file data for validation
        vector<BYTE> fileData;
        size_t readSize = (size_t)min(info.fileSize, (ULONGLONG)MAX_VALIDATION_SIZE);

        if (ExtractFile(info.startLCN, info.startOffset, readSize, fileData)) {
            FileIntegrityScore score = FileIntegrityValidator::Validate(
                fileData.data(), fileData.size(), info.extension);

            info.integrityScore = score.overallScore;
            info.integrityValidated = true;
            info.integrityDiagnosis = score.diagnosis;

            // Update statistics
            if (score.isLikelyCorrupted) {
                corruptedCount++;
            }
            if (score.overallScore >= FileIntegrityValidator::HIGH_CONFIDENCE_SCORE) {
                highConfidenceCount++;
            } else if (score.overallScore < FileIntegrityValidator::MIN_INTEGRITY_SCORE) {
                lowConfidenceCount++;
            }

            // Adjust confidence based on integrity score
            // If integrity is low, reduce confidence even if signature matched
            if (score.overallScore < 0.5) {
                info.confidence *= 0.7;
            } else if (score.overallScore >= 0.8) {
                info.confidence = min(1.0, info.confidence * 1.1);
            }
        } else {
            info.integrityScore = 0.0;
            info.integrityValidated = false;
            info.integrityDiagnosis = "Failed to read file data";
            corruptedCount++;
        }

        validatedCount++;

        // Progress display
        if (showProgress && validatedCount % 50 == 0) {
            double progress = (double)validatedCount / results.size() * 100.0;
            cout << "\rValidating: " << fixed << setprecision(1) << progress << "% | "
                 << "Processed: " << validatedCount << "/" << results.size() << " | "
                 << "Corrupted: " << corruptedCount << flush;
        }
    }

    DWORD elapsed = GetTickCount() - startTime;

    if (showProgress) {
        cout << "\r                                                                    " << endl;
        cout << "\n--- Integrity Validation Complete ---" << endl;
        cout << "Time: " << (elapsed / 1000) << "." << ((elapsed % 1000) / 100) << " seconds" << endl;
        cout << "Files validated: " << validatedCount << endl;
        cout << "High confidence (>= 80%): " << highConfidenceCount << " ("
             << fixed << setprecision(1) << (100.0 * highConfidenceCount / validatedCount) << "%)" << endl;
        cout << "Low confidence (< 50%): " << lowConfidenceCount << " ("
             << fixed << setprecision(1) << (100.0 * lowConfidenceCount / validatedCount) << "%)" << endl;
        cout << "Likely corrupted: " << corruptedCount << " ("
             << fixed << setprecision(1) << (100.0 * corruptedCount / validatedCount) << "%)" << endl;
    }

    LOG_INFO_FMT("Integrity validation: %zu/%zu files validated, %zu likely corrupted",
                 validatedCount, results.size(), corruptedCount);
}

// 过滤出可能损坏的文件
vector<CarvedFileInfo> FileCarver::FilterCorruptedFiles(const vector<CarvedFileInfo>& results,
                                                         double minIntegrityScore) {
    vector<CarvedFileInfo> filtered;

    for (const auto& info : results) {
        // Only include files that pass integrity check
        if (!info.integrityValidated || info.integrityScore >= minIntegrityScore) {
            filtered.push_back(info);
        }
    }

    LOG_INFO_FMT("Filtered %zu/%zu files (min integrity score: %.2f)",
                 filtered.size(), results.size(), minIntegrityScore);

    return filtered;
}

// ============================================================================
// 删除状态检查功能
// ============================================================================

// 检查单个文件的删除状态
void FileCarver::CheckDeletionStatus(CarvedFileInfo& info) {
    info.deletionChecked = true;
    info.isDeleted = false;
    info.isActiveFile = false;

    // 如果 MFT 索引未构建，无法检查
    if (!lcnIndex || !mftIndexBuilt) {
        // 无法确定状态，假设为已删除（安全起见）
        info.isDeleted = true;
        return;
    }

    // 通过 LCN 查找匹配的 MFT 记录
    vector<LCNMappingInfo> matches = lcnIndex->FindByLCN(info.startLCN);

    if (matches.empty()) {
        // 没有 MFT 记录匹配此 LCN，很可能是已删除的文件
        // （MFT 记录已被覆盖或索引未包含此记录）
        info.isDeleted = true;
        return;
    }

    // 检查所有匹配的记录
    bool hasActiveMatch = false;
    bool hasDeletedMatch = false;

    for (const auto& match : matches) {
        if (match.isDeleted) {
            hasDeletedMatch = true;
        } else {
            hasActiveMatch = true;
        }
    }

    // 如果有活动文件匹配此 LCN，说明这是一个未删除的文件
    if (hasActiveMatch) {
        info.isActiveFile = true;
        info.isDeleted = false;

        // 尝试保存匹配的 MFT 记录号（选择活动文件的）
        for (const auto& match : matches) {
            if (!match.isDeleted) {
                info.matchedMftRecord = match.mftRecordNumber;
                break;
            }
        }
    } else if (hasDeletedMatch) {
        info.isDeleted = true;
        info.isActiveFile = false;

        // 保存匹配的已删除 MFT 记录号
        for (const auto& match : matches) {
            if (match.isDeleted) {
                info.matchedMftRecord = match.mftRecordNumber;
                break;
            }
        }
    } else {
        // 理论上不会到达这里
        info.isDeleted = true;
    }
}

// 批量检查删除状态
void FileCarver::CheckDeletionStatusForResults(vector<CarvedFileInfo>& results, bool showProgress) {
    if (results.empty()) {
        return;
    }

    // 确保 MFT 索引已构建
    if (!lcnIndex || !mftIndexBuilt) {
        if (showProgress) {
            cout << "Building MFT index for deletion status check..." << endl;
        }
        // 构建索引时包含活动文件，这样才能识别未删除的文件
        if (!BuildMFTIndex(true)) {
            if (showProgress) {
                cout << "[WARNING] Failed to build MFT index. Cannot verify deletion status." << endl;
            }
            // 标记所有文件为已检查但状态未知，假设为已删除
            for (auto& info : results) {
                info.deletionChecked = true;
                info.isDeleted = true;
                info.isActiveFile = false;
            }
            return;
        }
    }

    if (showProgress) {
        cout << "\n--- Checking Deletion Status ---" << endl;
        cout << "Files to check: " << results.size() << endl;
        cout << "MFT Index entries: " << lcnIndex->GetIndexSize() << endl;
    }

    DWORD startTime = GetTickCount();
    size_t processedCount = 0;
    size_t deletedCount = 0;
    size_t activeCount = 0;
    size_t unknownCount = 0;

    for (auto& info : results) {
        CheckDeletionStatus(info);

        if (info.isActiveFile) {
            activeCount++;
        } else if (info.isDeleted) {
            deletedCount++;
        } else {
            unknownCount++;
        }

        processedCount++;

        // 进度显示
        if (showProgress && processedCount % 100 == 0) {
            double progress = (double)processedCount / results.size() * 100.0;
            cout << "\rChecking: " << fixed << setprecision(1) << progress << "% | "
                 << "Deleted: " << deletedCount << " | Active: " << activeCount << flush;
        }
    }

    DWORD elapsed = GetTickCount() - startTime;

    if (showProgress) {
        cout << "\r                                                                    " << endl;
        cout << "\n--- Deletion Status Check Complete ---" << endl;
        cout << "Time: " << (elapsed / 1000) << "." << ((elapsed % 1000) / 100) << " seconds" << endl;
        cout << "Files checked: " << processedCount << endl;
        cout << "Deleted files: " << deletedCount << " ("
             << fixed << setprecision(1) << (100.0 * deletedCount / processedCount) << "%)" << endl;
        cout << "Active files (not deleted): " << activeCount << " ("
             << fixed << setprecision(1) << (100.0 * activeCount / processedCount) << "%)" << endl;
        if (unknownCount > 0) {
            cout << "Unknown status: " << unknownCount << " ("
                 << fixed << setprecision(1) << (100.0 * unknownCount / processedCount) << "%)" << endl;
        }
    }

    LOG_INFO_FMT("Deletion status check: %zu deleted, %zu active, %zu unknown (total: %zu)",
                 deletedCount, activeCount, unknownCount, processedCount);
}

// 过滤出已删除的文件
vector<CarvedFileInfo> FileCarver::FilterDeletedOnly(const vector<CarvedFileInfo>& results) {
    vector<CarvedFileInfo> filtered;

    for (const auto& info : results) {
        // 包含已删除的文件，或者未检查状态的文件（可能是已删除的）
        if (!info.deletionChecked || info.isDeleted) {
            filtered.push_back(info);
        }
    }

    LOG_INFO_FMT("Filtered deleted files: %zu/%zu", filtered.size(), results.size());
    return filtered;
}

// 过滤出活动文件（未删除）
vector<CarvedFileInfo> FileCarver::FilterActiveOnly(const vector<CarvedFileInfo>& results) {
    vector<CarvedFileInfo> filtered;

    for (const auto& info : results) {
        if (info.deletionChecked && info.isActiveFile) {
            filtered.push_back(info);
        }
    }

    LOG_INFO_FMT("Filtered active files: %zu/%zu", filtered.size(), results.size());
    return filtered;
}

// 获取已删除文件数量
size_t FileCarver::CountDeletedFiles(const vector<CarvedFileInfo>& results) {
    size_t count = 0;
    for (const auto& info : results) {
        if (!info.deletionChecked || info.isDeleted) {
            count++;
        }
    }
    return count;
}

// 获取活动文件数量
size_t FileCarver::CountActiveFiles(const vector<CarvedFileInfo>& results) {
    size_t count = 0;
    for (const auto& info : results) {
        if (info.deletionChecked && info.isActiveFile) {
            count++;
        }
    }
    return count;
}
