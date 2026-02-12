#include "FileCarverRecovery.h"
#include "MFTReader.h"
#include "Logger.h"
#include <iostream>
#include <iomanip>
#include <filesystem>

using namespace std;

// ============================================================================
// 构造函数
// ============================================================================

FileCarverRecovery::FileCarverRecovery(MFTReader* reader, const map<string, FileSignature>& signatures)
    : reader_(reader), signatures_(signatures), shouldStop_(false) {
}

// ============================================================================
// 提取文件数据
// ============================================================================

bool FileCarverRecovery::ExtractFile(ULONGLONG startLCN, ULONGLONG startOffset,
                                     ULONGLONG fileSize, vector<BYTE>& fileData) {
    ULONGLONG bytesPerCluster = reader_->GetBytesPerCluster();

    ULONGLONG totalBytes = startOffset + fileSize;
    ULONGLONG clustersNeeded = (totalBytes + bytesPerCluster - 1) / bytesPerCluster;

    const ULONGLONG MAX_READ_CLUSTERS = 100000;
    if (clustersNeeded > MAX_READ_CLUSTERS) {
        LOG_WARNING_FMT("File too large, limiting to %llu clusters", MAX_READ_CLUSTERS);
        clustersNeeded = MAX_READ_CLUSTERS;
        fileSize = clustersNeeded * bytesPerCluster - startOffset;
    }

    vector<BYTE> clusterData;
    if (!reader_->ReadClusters(startLCN, clustersNeeded, clusterData)) {
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
// 验证文件（优化版：避免重复计算）
// ============================================================================

double FileCarverRecovery::ValidateFileOptimized(const BYTE* data, size_t dataSize,
                                                  const FileSignature& sig,
                                                  bool signatureAlreadyMatched,
                                                  ULONGLONG footerPos) {
    double confidence = 0.5;

    if (signatureAlreadyMatched) {
        confidence += 0.3;
    }

    if (footerPos > 0) {
        confidence += 0.2;
    }

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
// 恢复 carved 文件
// ============================================================================

bool FileCarverRecovery::RecoverCarvedFile(const CarvedFileInfo& info, const string& outputPath) {
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
    namespace fs = std::filesystem;
    try {
        fs::path outPath(outputPath);
        fs::path parentDir = outPath.parent_path();

        if (!parentDir.empty() && !fs::exists(parentDir)) {
            std::error_code ec;
            fs::create_directories(parentDir, ec);
            if (ec) {
                cout << "Failed to create directory: " << parentDir.string() << endl;
                cout << "Error: " << ec.message() << endl;
                return false;
            }
        }
    } catch (const std::exception& e) {
        cout << "Path error: " << e.what() << endl;
        return false;
    }

    HANDLE hFile = CreateFileA(outputPath.c_str(), GENERIC_WRITE,
        0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        cout << "Failed to create output file: " << outputPath << endl;
        cout << "Error code: " << err;
        switch (err) {
            case 5:  cout << " (ACCESS_DENIED - check permissions or run as admin)"; break;
            case 3:  cout << " (PATH_NOT_FOUND - invalid path)"; break;
            case 32: cout << " (SHARING_VIOLATION - file in use)"; break;
            case 123: cout << " (INVALID_NAME - invalid characters in path)"; break;
        }
        cout << endl;
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
// 恢复前精细化
// ============================================================================

bool FileCarverRecovery::RefineCarvedFileInfo(CarvedFileInfo& info, bool verbose) {
    // 查找对应的签名定义
    string lookupExt = info.extension;
    // OOXML 类型映射回 zip
    if (lookupExt == "docx" || lookupExt == "xlsx" || lookupExt == "pptx" || lookupExt == "ooxml") {
        lookupExt = "zip";
    }

    auto sigIt = signatures_.find(lookupExt);
    if (sigIt == signatures_.end()) {
        if (verbose) {
            cout << "  [精细化] 未知文件类型: " << info.extension << "，跳过" << endl;
        }
        return true;
    }
    const FileSignature& sig = sigIt->second;

    bool needSizeRefinement = info.sizeIsEstimated;

    if (!needSizeRefinement && !info.integrityValidated) {
        if (verbose) {
            cout << "  [精细化] 文件大小已精确 (" << info.fileSize << " bytes)，执行完整性验证..." << endl;
        }

        // 读取数据进行验证
        vector<BYTE> fileData;
        size_t readSize = (size_t)min(info.fileSize, (ULONGLONG)(2 * 1024 * 1024));
        if (!ExtractFile(info.startLCN, info.startOffset, readSize, fileData)) {
            if (verbose) {
                cout << "  [精细化] 无法读取文件数据" << endl;
            }
            return true;
        }

        FileIntegrityScore integrity = FileIntegrityValidator::Validate(
            fileData.data(), fileData.size(), info.extension);
        info.integrityScore = integrity.overallScore;
        info.integrityValidated = true;
        info.integrityDiagnosis = integrity.diagnosis;

        if (integrity.overallScore >= 0.8) {
            info.confidence = min(1.0, info.confidence * 1.1);
        } else if (integrity.overallScore < 0.5) {
            info.confidence *= 0.7;
        }

        if (verbose) {
            cout << "  [精细化] 完整性: " << fixed << setprecision(1)
                 << (integrity.overallScore * 100) << "% - " << integrity.diagnosis << endl;
        }
        return !integrity.isLikelyCorrupted;
    }

    if (!needSizeRefinement) {
        return true;
    }

    // ===== 需要精确计算文件大小 =====
    if (verbose) {
        cout << "  [精细化] 大小为估计值 (" << info.fileSize << " bytes)，正在精确计算..." << endl;
    }

    ULONGLONG readSize = info.fileSize;
    const ULONGLONG MAX_REFINE_READ = 64ULL * 1024 * 1024;
    if (readSize > MAX_REFINE_READ) {
        readSize = MAX_REFINE_READ;
    }

    vector<BYTE> fileData;
    if (!ExtractFile(info.startLCN, info.startOffset, readSize, fileData)) {
        if (verbose) {
            cout << "  [精细化] 无法读取文件数据，保持原始估计" << endl;
        }
        return true;
    }

    // 调用 FileFormatUtils 进行精确大小计算
    ULONGLONG footerPos = 0;
    bool isComplete = false;
    ULONGLONG refinedSize = FileFormatUtils::EstimateFileSizeStatic(
        fileData.data(), fileData.size(), sig, &footerPos, &isComplete);

    ULONGLONG originalSize = info.fileSize;
    bool sizeChanged = false;

    if (refinedSize > 0 && refinedSize != originalSize) {
        info.fileSize = refinedSize;
        info.sizeIsEstimated = !isComplete;
        sizeChanged = true;

        if (verbose) {
            double ratio = (double)refinedSize / originalSize * 100.0;
            cout << "  [精细化] 大小: " << originalSize << " -> " << refinedSize
                 << " bytes (" << fixed << setprecision(1) << ratio << "%)" << endl;
        }
    } else if (refinedSize == 0) {
        if (verbose) {
            cout << "  [精细化] 警告: 无法确定有效文件边界" << endl;
        }
        info.confidence *= 0.5;
    }

    if (footerPos > 0) {
        info.hasValidFooter = true;
        if (isComplete) {
            info.sizeIsEstimated = false;
        }
    }

    // OOXML 检测
    if (lookupExt == "zip" && info.extension == "zip") {
        string ooxmlType = FileFormatUtils::DetectOOXMLTypeStatic(fileData.data(), fileData.size());
        if (!ooxmlType.empty()) {
            info.extension = ooxmlType;
            info.description = ooxmlType + " (Office)";
            if (verbose) {
                cout << "  [精细化] 检测到 Office 文档类型: " << ooxmlType << endl;
            }
        }
    }

    // ===== 完整性验证 =====
    size_t validateSize = (size_t)min((ULONGLONG)fileData.size(), (ULONGLONG)(2 * 1024 * 1024));
    FileIntegrityScore integrity = FileIntegrityValidator::Validate(
        fileData.data(), validateSize, info.extension);

    info.integrityScore = integrity.overallScore;
    info.integrityValidated = true;
    info.integrityDiagnosis = integrity.diagnosis;

    // ===== 置信度重估 =====
    double originalConfidence = info.confidence;

    if (info.hasValidFooter && isComplete) {
        info.confidence = min(1.0, info.confidence * 1.15);
    } else if (!info.hasValidFooter && (info.extension == "jpg" || info.extension == "png" ||
               info.extension == "pdf" || info.extension == "gif")) {
        info.confidence *= 0.75;
    }

    if (integrity.overallScore >= 0.8) {
        info.confidence = min(1.0, info.confidence * 1.1);
    } else if (integrity.overallScore >= 0.5) {
        // 中等，不调整
    } else {
        info.confidence *= 0.6;
    }

    if (verbose) {
        cout << "  [精细化] 完整性: " << fixed << setprecision(1)
             << (integrity.overallScore * 100) << "% - " << integrity.diagnosis << endl;
        cout << "  [精细化] 置信度: " << fixed << setprecision(1)
             << (originalConfidence * 100) << "% -> " << (info.confidence * 100) << "%" << endl;

        if (info.hasValidFooter) {
            cout << "  [精细化] 文件尾: 有效" << (isComplete ? " (结构完整)" : "") << endl;
        } else {
            cout << "  [精细化] 文件尾: 未找到" << endl;
        }
    }

    return !integrity.isLikelyCorrupted;
}

// ============================================================================
// ZIP EOCD 扫描恢复（策略分发）
// ============================================================================

FileCarverRecovery::ZipRecoveryResult FileCarverRecovery::RecoverZipWithEOCDScan(
    ULONGLONG startLCN,
    const string& outputPath,
    const ZipRecoveryConfig& config
) {
    ULONGLONG estimatedSize = config.expectedSize;
    bool useStreamingMode = false;

    if (estimatedSize == 0) {
        useStreamingMode = true;
        LOG_INFO("Unknown size: using streaming mode with " +
                 to_string(MAX_STREAMING_LIMIT / (1024 * 1024)) + "MB limit");
    } else if (estimatedSize <= MEMORY_BUFFER_THRESHOLD) {
        useStreamingMode = false;
        LOG_INFO("Small file (" + to_string(estimatedSize / (1024 * 1024)) +
                 "MB): using memory buffer");
    } else {
        useStreamingMode = true;
        LOG_INFO("Large file (" + to_string(estimatedSize / (1024 * 1024)) +
                 "MB): using streaming mode");
    }

    if (useStreamingMode) {
        return RecoverZipWithEOCDScan_Streaming(startLCN, outputPath, config, estimatedSize);
    } else {
        return RecoverZipWithEOCDScan_MemoryBuffer(startLCN, outputPath, config, estimatedSize);
    }
}

// ============================================================================
// 内存缓冲模式（小文件/已知大小 ≤256MB）
// ============================================================================

FileCarverRecovery::ZipRecoveryResult FileCarverRecovery::RecoverZipWithEOCDScan_MemoryBuffer(
    ULONGLONG startLCN,
    const string& outputPath,
    const ZipRecoveryConfig& config,
    ULONGLONG estimatedSize
) {
    ZipRecoveryResult result;
    result.diagnosis = "";

    LOG_INFO("Starting ZIP recovery (memory buffer mode) from LCN " + to_string(startLCN));

    ULONGLONG bytesPerCluster = reader_->GetBytesPerCluster();
    if (bytesPerCluster == 0) {
        result.diagnosis = "Invalid bytes per cluster";
        LOG_ERROR(result.diagnosis);
        return result;
    }

    const size_t CHUNK_SIZE = 4 * 1024 * 1024;
    const size_t CLUSTERS_PER_CHUNK = CHUNK_SIZE / static_cast<size_t>(bytesPerCluster);

    ULONGLONG maxSearchSize = config.maxSize;
    if (config.expectedSize > 0) {
        ULONGLONG tolerance = config.expectedSizeTolerance;
        if (tolerance == 0) {
            tolerance = config.expectedSize / 10;
        }
        maxSearchSize = min(maxSearchSize, config.expectedSize + tolerance);
        LOG_INFO("Using expected size " + to_string(config.expectedSize) +
                 " with tolerance " + to_string(tolerance));
    }

    vector<BYTE> fileBuffer;
    fileBuffer.reserve(min(maxSearchSize, (ULONGLONG)256 * 1024 * 1024));

    vector<BYTE> chunk(CHUNK_SIZE);
    ULONGLONG currentLCN = startLCN;
    ULONGLONG bytesRead = 0;
    bool foundEOCD = false;
    ULONGLONG eocdPosition = 0;

    cout << "Scanning for ZIP EOCD (max " << (maxSearchSize / (1024 * 1024)) << " MB)..." << endl;

    while (bytesRead < maxSearchSize && !shouldStop_) {
        if (!reader_->ReadClusters(currentLCN, CLUSTERS_PER_CHUNK, chunk)) {
            result.diagnosis = "Failed to read cluster at LCN " + to_string(currentLCN);
            LOG_WARNING(result.diagnosis);

            if (config.allowFragmented) {
                result.isFragmented = true;
                currentLCN += CLUSTERS_PER_CHUNK;
                bytesRead += CHUNK_SIZE;
                continue;
            } else {
                break;
            }
        }

        if (chunk.empty()) {
            break;
        }

        size_t chunkSize = chunk.size();

        ULONGLONG eocdInChunk = FileFormatUtils::FindZipEndOfCentralDirectoryStatic(chunk.data(), chunkSize);

        if (eocdInChunk > 0) {
            foundEOCD = true;
            eocdPosition = bytesRead + eocdInChunk;

            fileBuffer.insert(fileBuffer.end(), chunk.begin(), chunk.begin() + eocdInChunk);
            bytesRead += eocdInChunk;

            LOG_INFO("Found EOCD at position " + to_string(eocdPosition));
            result.diagnosis = "EOCD found at offset " + to_string(eocdPosition);

            if (config.stopOnFirstEOCD) {
                break;
            }
        } else {
            ULONGLONG maxBufferSize = estimatedSize > 0 ?
                min(estimatedSize * 2, MAX_STREAMING_LIMIT) : MAX_STREAMING_LIMIT;

            if (fileBuffer.size() + chunkSize > maxBufferSize) {
                result.success = false;
                result.diagnosis = "Memory buffer limit (" +
                                  to_string(maxBufferSize / (1024 * 1024)) +
                                  "MB) exceeded without finding EOCD. File likely corrupted.";
                LOG_WARNING(result.diagnosis);
                break;
            }

            fileBuffer.insert(fileBuffer.end(), chunk.begin(), chunk.end());
            bytesRead += chunkSize;
        }

        currentLCN += CLUSTERS_PER_CHUNK;

        if (bytesRead % (16 * 1024 * 1024) == 0) {
            cout << "\rScanning: " << (bytesRead / (1024 * 1024)) << " MB" << flush;
        }

        if (config.expectedSize > 0 && bytesRead > config.expectedSize * 2) {
            result.diagnosis = "Exceeded 2x expected size without finding EOCD";
            LOG_WARNING(result.diagnosis);
            break;
        }
    }

    cout << "\r                                    " << endl;

    if (!foundEOCD) {
        result.success = false;
        if (result.diagnosis.empty()) {
            result.diagnosis = "EOCD not found within " + to_string(bytesRead) + " bytes";
        }
        LOG_WARNING(result.diagnosis);

        if (!fileBuffer.empty() && !outputPath.empty()) {
            string truncatedPath = outputPath + ".incomplete";
            HANDLE hFile = CreateFileA(truncatedPath.c_str(), GENERIC_WRITE, 0, NULL,
                                       CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile != INVALID_HANDLE_VALUE) {
                DWORD written = 0;
                WriteFile(hFile, fileBuffer.data(),
                         static_cast<DWORD>(fileBuffer.size()), &written, NULL);
                CloseHandle(hFile);
                result.bytesWritten = written;
                result.diagnosis += "; Incomplete data saved to " + truncatedPath;
            }
        }

        return result;
    }

    result.actualSize = fileBuffer.size();

    HANDLE hFile = CreateFileA(outputPath.c_str(), GENERIC_WRITE, 0, NULL,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        result.diagnosis = "Failed to create output file: " + outputPath;
        LOG_ERROR(result.diagnosis);
        return result;
    }

    DWORD written = 0;
    BOOL writeSuccess = WriteFile(hFile, fileBuffer.data(),
                                  static_cast<DWORD>(fileBuffer.size()), &written, NULL);
    CloseHandle(hFile);

    if (!writeSuccess || written != fileBuffer.size()) {
        result.diagnosis = "Failed to write file data";
        LOG_ERROR(result.diagnosis);
        return result;
    }

    result.bytesWritten = written;
    result.success = true;

    if (config.verifyCRC) {
        cout << "Verifying ZIP integrity..." << endl;
        ZipRecoveryResult validation = ValidateZipData(fileBuffer.data(), fileBuffer.size());
        result.crcValid = validation.crcValid;
        result.totalFiles = validation.totalFiles;
        result.corruptedFiles = validation.corruptedFiles;

        if (!validation.crcValid) {
            result.diagnosis += "; CRC validation failed: " + validation.diagnosis;
        } else {
            result.diagnosis += "; CRC validation passed (" +
                               to_string(result.totalFiles) + " files)";
        }
    }

    LOG_INFO("ZIP recovery completed: " + to_string(result.actualSize) +
             " bytes written to " + outputPath);
    cout << "Recovered " << (result.actualSize / 1024) << " KB to " << outputPath << endl;

    return result;
}

// ============================================================================
// 流式写入模式（大文件/未知大小）
// ============================================================================

FileCarverRecovery::ZipRecoveryResult FileCarverRecovery::RecoverZipWithEOCDScan_Streaming(
    ULONGLONG startLCN,
    const string& outputPath,
    const ZipRecoveryConfig& config,
    ULONGLONG estimatedSize
) {
    ZipRecoveryResult result;
    result.diagnosis = "";

    LOG_INFO("Starting ZIP recovery (streaming mode) from LCN " + to_string(startLCN));

    ULONGLONG bytesPerCluster = reader_->GetBytesPerCluster();
    if (bytesPerCluster == 0) {
        result.diagnosis = "Invalid bytes per cluster";
        LOG_ERROR(result.diagnosis);
        return result;
    }

    const size_t STREAM_BUFFER_SIZE = 32 * 1024 * 1024;
    const size_t CHUNK_SIZE = 4 * 1024 * 1024;
    const size_t CLUSTERS_PER_CHUNK = CHUNK_SIZE / static_cast<size_t>(bytesPerCluster);

    ULONGLONG maxSearchSize = config.maxSize;
    if (estimatedSize > 0) {
        maxSearchSize = min(maxSearchSize, static_cast<ULONGLONG>(estimatedSize * 1.2));
    } else {
        maxSearchSize = min(maxSearchSize, MAX_STREAMING_LIMIT);
    }

    LOG_INFO("Streaming mode: maxSearch=" + to_string(maxSearchSize / (1024 * 1024)) + "MB");

    HANDLE hFile = CreateFileA(outputPath.c_str(), GENERIC_WRITE, 0, NULL,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        result.diagnosis = "Failed to create output file: " + outputPath;
        LOG_ERROR(result.diagnosis);
        return result;
    }

    vector<BYTE> streamBuffer;
    streamBuffer.reserve(STREAM_BUFFER_SIZE);

    vector<BYTE> chunk(CHUNK_SIZE);
    ULONGLONG currentLCN = startLCN;
    ULONGLONG bytesRead = 0;
    ULONGLONG bytesWrittenTotal = 0;
    bool foundEOCD = false;
    ULONGLONG eocdPosition = 0;

    cout << "Streaming ZIP recovery (max " << (maxSearchSize / (1024 * 1024))
         << " MB)..." << endl;

    while (bytesRead < maxSearchSize && !shouldStop_) {
        if (!reader_->ReadClusters(currentLCN, CLUSTERS_PER_CHUNK, chunk)) {
            result.diagnosis = "Failed to read cluster at LCN " + to_string(currentLCN);
            LOG_WARNING(result.diagnosis);

            if (config.allowFragmented) {
                result.isFragmented = true;
                currentLCN += CLUSTERS_PER_CHUNK;
                bytesRead += CHUNK_SIZE;
                continue;
            } else {
                break;
            }
        }

        if (chunk.empty()) {
            break;
        }

        size_t chunkSize = chunk.size();

        ULONGLONG eocdInChunk = FileFormatUtils::FindZipEndOfCentralDirectoryStatic(chunk.data(), chunkSize);

        if (eocdInChunk > 0) {
            foundEOCD = true;
            eocdPosition = bytesRead + eocdInChunk;

            if (!streamBuffer.empty()) {
                DWORD written = 0;
                WriteFile(hFile, streamBuffer.data(),
                         static_cast<DWORD>(streamBuffer.size()), &written, NULL);
                bytesWrittenTotal += written;
                streamBuffer.clear();
            }

            DWORD written = 0;
            WriteFile(hFile, chunk.data(), static_cast<DWORD>(eocdInChunk), &written, NULL);
            bytesWrittenTotal += written;

            LOG_INFO("EOCD found at position " + to_string(eocdPosition));
            result.diagnosis = "EOCD found at offset " + to_string(eocdPosition);

            if (config.stopOnFirstEOCD) {
                break;
            }
        } else {
            streamBuffer.insert(streamBuffer.end(), chunk.begin(), chunk.end());
            bytesRead += chunkSize;

            if (streamBuffer.size() >= STREAM_BUFFER_SIZE) {
                DWORD written = 0;
                BOOL success = WriteFile(hFile, streamBuffer.data(),
                                        static_cast<DWORD>(streamBuffer.size()),
                                        &written, NULL);
                if (!success) {
                    CloseHandle(hFile);
                    result.diagnosis = "Failed to write to file";
                    LOG_ERROR(result.diagnosis);
                    return result;
                }

                bytesWrittenTotal += written;
                streamBuffer.clear();

                LOG_DEBUG("Flushed " + to_string(written / (1024 * 1024)) + "MB to disk");
            }
        }

        currentLCN += CLUSTERS_PER_CHUNK;

        if (bytesRead % (16 * 1024 * 1024) == 0) {
            cout << "\rStreaming: " << (bytesRead / (1024 * 1024)) << " MB"
                 << " (written: " << (bytesWrittenTotal / (1024 * 1024)) << " MB)" << flush;
        }
    }

    if (!streamBuffer.empty()) {
        DWORD written = 0;
        WriteFile(hFile, streamBuffer.data(),
                 static_cast<DWORD>(streamBuffer.size()), &written, NULL);
        bytesWrittenTotal += written;
    }

    CloseHandle(hFile);
    cout << "\r                                                " << endl;

    result.actualSize = bytesWrittenTotal;
    result.bytesWritten = bytesWrittenTotal;

    if (!foundEOCD) {
        result.success = false;
        result.diagnosis = "EOCD not found within " + to_string(bytesRead) + " bytes";
        LOG_WARNING(result.diagnosis);

        string incompletePath = outputPath + ".incomplete";
        if (MoveFileA(outputPath.c_str(), incompletePath.c_str())) {
            result.diagnosis += "; Incomplete data saved to " + incompletePath;
        }

        return result;
    }

    result.success = true;
    LOG_INFO("ZIP streaming recovery successful: " + to_string(bytesWrittenTotal) + " bytes");
    cout << "Recovered " << (result.actualSize / 1024) << " KB to " << outputPath << endl;

    return result;
}

// ============================================================================
// 验证 ZIP 文件完整性（从文件路径）
// ============================================================================

FileCarverRecovery::ZipRecoveryResult FileCarverRecovery::ValidateZipFile(const string& filePath) {
    ZipRecoveryResult result;

    HANDLE hFile = CreateFileA(filePath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        result.diagnosis = "Failed to open file: " + filePath;
        return result;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        result.diagnosis = "Failed to get file size";
        return result;
    }

    const ULONGLONG MAX_VALIDATE_SIZE = 512ULL * 1024 * 1024;
    if ((ULONGLONG)fileSize.QuadPart > MAX_VALIDATE_SIZE) {
        CloseHandle(hFile);
        result.diagnosis = "File too large for full validation (>" +
                          to_string(MAX_VALIDATE_SIZE / (1024 * 1024)) + "MB)";
        return result;
    }

    vector<BYTE> data((size_t)fileSize.QuadPart);
    DWORD bytesRead = 0;
    if (!ReadFile(hFile, data.data(), (DWORD)fileSize.QuadPart, &bytesRead, NULL)) {
        CloseHandle(hFile);
        result.diagnosis = "Failed to read file";
        return result;
    }
    CloseHandle(hFile);

    return ValidateZipData(data.data(), data.size());
}

// ============================================================================
// 验证 ZIP 数据完整性（从内存）
// ============================================================================

FileCarverRecovery::ZipRecoveryResult FileCarverRecovery::ValidateZipData(const BYTE* data, size_t dataSize) {
    // 委托给 FileFormatUtils 进行实际验证
    auto fmtResult = FileFormatUtils::ValidateZipData(data, dataSize);

    // 转换结果类型
    ZipRecoveryResult result;
    result.success = fmtResult.success;
    result.actualSize = fmtResult.actualSize;
    result.crcValid = fmtResult.crcValid;
    result.totalFiles = fmtResult.totalFiles;
    result.corruptedFiles = fmtResult.corruptedFiles;
    result.diagnosis = fmtResult.diagnosis;

    return result;
}
