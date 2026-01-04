#include "FileCarver.h"
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
      ioWaitTimeMs(0), scanTimeMs(0), totalIoTimeMs(0), totalScanTimeMs(0) {
    memset(&stats, 0, sizeof(stats));

    // 初始化双缓冲区
    for (int i = 0; i < 2; i++) {
        buffers[i].ready = false;
        buffers[i].isEmpty = false;
        buffers[i].isLast = false;
        buffers[i].startLCN = 0;
        buffers[i].clusterCount = 0;
    }

    InitializeSignatures();
    BuildSignatureIndex();
    LOG_INFO("FileCarver initialized with async I/O support");
}

FileCarver::~FileCarver() {
    shouldStop = true;  // 确保停止任何运行中的线程
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
