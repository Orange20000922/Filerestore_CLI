#include "FileFormatUtils.h"
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace std;

// ============================================================================
// 快速熵计算
// ============================================================================

float FileFormatUtils::QuickEntropy(const BYTE* data, size_t size) {
    if (data == nullptr || size == 0) return 0.0f;

    // 快速字节频率统计
    size_t counts[256] = {0};
    for (size_t i = 0; i < size; i++) {
        counts[data[i]]++;
    }

    // 计算 Shannon 熵
    double entropy = 0.0;
    double sizeD = static_cast<double>(size);

    for (int i = 0; i < 256; i++) {
        if (counts[i] > 0) {
            double p = static_cast<double>(counts[i]) / sizeD;
            entropy -= p * log2(p);
        }
    }

    return static_cast<float>(entropy);
}

// ============================================================================
// 查找文件尾（正向搜索）
// ============================================================================

ULONGLONG FileFormatUtils::FindFooterStatic(const BYTE* data, size_t dataSize,
                                            const vector<BYTE>& footer, ULONGLONG maxSearch) {
    if (footer.empty() || dataSize < footer.size()) {
        return 0;
    }

    size_t searchLimit = min((size_t)maxSearch, dataSize - footer.size());

    // 从前向后搜索（适用于 JPEG EOI 等）
    for (size_t i = 0; i <= searchLimit; i++) {
        if (memcmp(data + i, footer.data(), footer.size()) == 0) {
            return i + footer.size();
        }
    }
    return 0;
}

// ============================================================================
// 查找文件尾（反向搜索）
// ============================================================================

ULONGLONG FileFormatUtils::FindFooterReverseStatic(const BYTE* data, size_t dataSize,
                                                   const vector<BYTE>& footer, ULONGLONG maxSearch) {
    if (footer.empty() || dataSize < footer.size()) {
        return 0;
    }

    // 从后向前搜索，适用于文件尾在末尾的格式（ZIP, PDF）
    size_t searchStart = dataSize - footer.size();
    size_t searchLimit = (maxSearch < dataSize) ? (dataSize - maxSearch) : 0;

    for (size_t i = searchStart; i >= searchLimit && i < dataSize; i--) {
        if (memcmp(data + i, footer.data(), footer.size()) == 0) {
            return i + footer.size();
        }
        if (i == 0) break;  // 防止无符号下溢
    }
    return 0;
}

// ============================================================================
// PNG chunk 遍历查找 IEND
// ============================================================================

ULONGLONG FileFormatUtils::FindPngEndByChunksStatic(const BYTE* data, size_t dataSize) {
    const size_t PNG_SIGNATURE_SIZE = 8;
    const size_t CHUNK_HEADER_SIZE = 8;  // length + type
    const size_t CHUNK_CRC_SIZE = 4;

    // 最小 PNG: 签名(8) + IHDR chunk(25) + IEND chunk(12) = 45 字节
    if (dataSize < 45) {
        return 0;
    }

    // 验证 PNG 签名
    const BYTE pngSig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (memcmp(data, pngSig, PNG_SIGNATURE_SIZE) != 0) {
        return 0;
    }

    size_t offset = PNG_SIGNATURE_SIZE;

    // 遍历 chunk
    while (offset + CHUNK_HEADER_SIZE + CHUNK_CRC_SIZE <= dataSize) {
        // 读取 chunk 长度 (大端序)
        DWORD chunkLength = (data[offset] << 24) | (data[offset + 1] << 16) |
                           (data[offset + 2] << 8) | data[offset + 3];

        // 安全检查：chunk 长度不能超过 2GB (PNG 规范限制)
        if (chunkLength > 0x7FFFFFFF) {
            return 0;
        }

        // 读取 chunk 类型
        char chunkType[5] = {0};
        memcpy(chunkType, data + offset + 4, 4);

        // 计算完整 chunk 大小
        size_t fullChunkSize = CHUNK_HEADER_SIZE + chunkLength + CHUNK_CRC_SIZE;

        // 检查是否超出数据范围
        if (offset + fullChunkSize > dataSize) {
            return 0;
        }

        // 检查是否是 IEND chunk
        if (chunkType[0] == 'I' && chunkType[1] == 'E' &&
            chunkType[2] == 'N' && chunkType[3] == 'D') {
            if (chunkLength == 0) {
                return offset + fullChunkSize;
            }
        }

        // 验证 chunk 类型的有效性（每个字符应该是 ASCII 字母）
        bool validType = true;
        for (int i = 0; i < 4; i++) {
            char c = chunkType[i];
            if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) {
                validType = false;
                break;
            }
        }

        if (!validType) {
            return 0;
        }

        offset += fullChunkSize;
    }

    return 0;
}

// ============================================================================
// 检测 OOXML Office 文档类型
// ============================================================================

string FileFormatUtils::DetectOOXMLTypeStatic(const BYTE* data, size_t dataSize) {
    if (dataSize < 30) return "";

    // 验证是 ZIP 文件
    if (data[0] != 0x50 || data[1] != 0x4B ||
        data[2] != 0x03 || data[3] != 0x04) {
        return "";
    }

    WORD filenameLen = *(WORD*)(data + 26);
    WORD extraLen = *(WORD*)(data + 28);

    // 安全检查：文件名长度不应超过 1KB
    const WORD MAX_FILENAME_LEN = 1024;
    if (filenameLen == 0 || filenameLen > MAX_FILENAME_LEN ||
        30 + filenameLen > dataSize) {
        return "";
    }

    string filename((char*)(data + 30), filenameLen);

    bool hasContentTypes = (filename == "[Content_Types].xml");

    if (filename.substr(0, 5) == "word/" || filename == "word") {
        return "docx";
    }
    if (filename.substr(0, 3) == "xl/" || filename == "xl") {
        return "xlsx";
    }
    if (filename.substr(0, 4) == "ppt/" || filename == "ppt") {
        return "pptx";
    }

    if (hasContentTypes || filename.substr(0, 6) == "_rels/") {
        size_t offset = 30 + filenameLen + extraLen;
        DWORD compressedSize = *(DWORD*)(data + 18);
        offset += compressedSize;

        for (int i = 0; i < 10 && offset + 30 < dataSize; i++) {
            if (data[offset] != 0x50 || data[offset + 1] != 0x4B) {
                break;
            }

            if (data[offset + 2] == 0x03 && data[offset + 3] == 0x04) {
                WORD fnLen = *(WORD*)(data + offset + 26);
                WORD exLen = *(WORD*)(data + offset + 28);
                DWORD cmpSize = *(DWORD*)(data + offset + 18);

                if (fnLen > 0 && fnLen <= MAX_FILENAME_LEN &&
                    offset + 30 + fnLen <= dataSize) {
                    string fn((char*)(data + offset + 30), fnLen);

                    if (fn.substr(0, 5) == "word/" || fn == "word") {
                        return "docx";
                    }
                    if (fn.substr(0, 3) == "xl/" || fn == "xl") {
                        return "xlsx";
                    }
                    if (fn.substr(0, 4) == "ppt/" || fn == "ppt") {
                        return "pptx";
                    }
                }

                offset += 30 + fnLen + exLen + cmpSize;
            } else {
                break;
            }
        }

        if (hasContentTypes) {
            return "ooxml";
        }
    }

    return "";
}

// ============================================================================
// 查找 ZIP EOCD
// ============================================================================

ULONGLONG FileFormatUtils::FindZipEndOfCentralDirectoryStatic(const BYTE* data, size_t dataSize) {
    if (dataSize < 22) {
        return 0;
    }

    // 从后向前搜索 EOCD 签名（最多搜索 65KB + 22 字节）
    size_t searchStart = dataSize - 22;
    size_t searchLimit = (dataSize > 65557) ? (dataSize - 65557) : 0;

    for (size_t i = searchStart; i >= searchLimit && i < dataSize; i--) {
        if (i + 22 <= dataSize &&
            data[i] == 0x50 && data[i + 1] == 0x4B &&
            data[i + 2] == 0x05 && data[i + 3] == 0x06) {

            WORD commentLen = *(WORD*)(data + i + 20);
            size_t eocdEnd = i + 22 + commentLen;

            if (eocdEnd <= dataSize) {
                if (commentLen == 0 || eocdEnd == dataSize) {
                    return eocdEnd;
                }
                if (eocdEnd < dataSize) {
                    bool allZero = true;
                    for (size_t j = eocdEnd; j < min(eocdEnd + 512, dataSize); j++) {
                        if (data[j] != 0) {
                            allZero = false;
                            break;
                        }
                    }
                    if (allZero || eocdEnd + 512 >= dataSize) {
                        return eocdEnd;
                    }
                }
            }
        }
        if (i == 0) break;
    }

    return 0;
}

// ============================================================================
// 通过 Local File Headers 估算 ZIP 大小
// ============================================================================

ULONGLONG FileFormatUtils::EstimateZipSizeByHeaders(const BYTE* data, size_t dataSize,
                                                     bool* outIsComplete) {
    const size_t LOCAL_HEADER_MIN_SIZE = 30;

    if (dataSize < LOCAL_HEADER_MIN_SIZE) {
        if (outIsComplete) *outIsComplete = false;
        return 0;
    }

    // 验证起始是 ZIP 签名
    if (data[0] != 0x50 || data[1] != 0x4B || data[2] != 0x03 || data[3] != 0x04) {
        if (outIsComplete) *outIsComplete = false;
        return 0;
    }

    size_t offset = 0;
    int fileCount = 0;
    const int MAX_FILES = 100000;

    while (offset + LOCAL_HEADER_MIN_SIZE <= dataSize && fileCount < MAX_FILES) {
        BYTE sig0 = data[offset];
        BYTE sig1 = data[offset + 1];
        BYTE sig2 = data[offset + 2];
        BYTE sig3 = data[offset + 3];

        // Local File Header: PK\x03\x04
        if (sig0 == 0x50 && sig1 == 0x4B && sig2 == 0x03 && sig3 == 0x04) {
            if (offset + LOCAL_HEADER_MIN_SIZE > dataSize) {
                if (outIsComplete) *outIsComplete = false;
                return dataSize;
            }

            DWORD compressedSize = *(DWORD*)(data + offset + 18);
            WORD filenameLen = *(WORD*)(data + offset + 26);
            WORD extraLen = *(WORD*)(data + offset + 28);

            WORD flags = *(WORD*)(data + offset + 6);
            bool hasDataDescriptor = (flags & 0x0008) != 0;

            if (hasDataDescriptor && compressedSize == 0) {
                if (outIsComplete) *outIsComplete = false;
                return max((ULONGLONG)offset, (ULONGLONG)dataSize);
            }

            if (filenameLen > 1024 || extraLen > 65535) {
                if (outIsComplete) *outIsComplete = false;
                return offset > 0 ? offset : dataSize;
            }

            size_t entrySize = LOCAL_HEADER_MIN_SIZE + filenameLen + extraLen + compressedSize;

            if (hasDataDescriptor) {
                entrySize += 16;
            }

            if (offset + entrySize > dataSize) {
                if (outIsComplete) *outIsComplete = false;
                return offset + entrySize;
            }

            offset += entrySize;
            fileCount++;
            continue;
        }

        // Central Directory Header: PK\x01\x02
        if (sig0 == 0x50 && sig1 == 0x4B && sig2 == 0x01 && sig3 == 0x02) {
            const size_t CD_HEADER_MIN_SIZE = 46;

            while (offset + 4 <= dataSize) {
                BYTE cd0 = data[offset];
                BYTE cd1 = data[offset + 1];
                BYTE cd2 = data[offset + 2];
                BYTE cd3 = data[offset + 3];

                if (cd0 == 0x50 && cd1 == 0x4B && cd2 == 0x01 && cd3 == 0x02) {
                    if (offset + CD_HEADER_MIN_SIZE > dataSize) {
                        if (outIsComplete) *outIsComplete = false;
                        return dataSize;
                    }

                    WORD fnLen = *(WORD*)(data + offset + 28);
                    WORD exLen = *(WORD*)(data + offset + 30);
                    WORD cmtLen = *(WORD*)(data + offset + 32);

                    offset += CD_HEADER_MIN_SIZE + fnLen + exLen + cmtLen;
                }
                else if (cd0 == 0x50 && cd1 == 0x4B && cd2 == 0x05 && cd3 == 0x06) {
                    if (offset + 22 > dataSize) {
                        if (outIsComplete) *outIsComplete = false;
                        return dataSize;
                    }

                    WORD commentLen = *(WORD*)(data + offset + 20);
                    size_t zipEnd = offset + 22 + commentLen;

                    if (outIsComplete) *outIsComplete = true;
                    return zipEnd;
                }
                else if (cd0 == 0x50 && cd1 == 0x4B && cd2 == 0x06 && cd3 == 0x06) {
                    offset += 56;  // ZIP64 EOCD
                }
                else if (cd0 == 0x50 && cd1 == 0x4B && cd2 == 0x06 && cd3 == 0x07) {
                    offset += 20;  // ZIP64 EOCD Locator
                }
                else {
                    break;
                }
            }

            if (outIsComplete) *outIsComplete = false;
            return offset;
        }

        // EOCD: PK\x05\x06
        if (sig0 == 0x50 && sig1 == 0x4B && sig2 == 0x05 && sig3 == 0x06) {
            if (offset + 22 <= dataSize) {
                WORD commentLen = *(WORD*)(data + offset + 20);
                if (outIsComplete) *outIsComplete = true;
                return offset + 22 + commentLen;
            }
        }

        if (outIsComplete) *outIsComplete = false;
        return offset > 0 ? offset : dataSize;
    }

    if (outIsComplete) *outIsComplete = false;
    return offset;
}

// ============================================================================
// 综合估算文件大小
// ============================================================================

ULONGLONG FileFormatUtils::EstimateFileSizeStatic(const BYTE* data, size_t dataSize,
                                                  const FileSignature& sig,
                                                  ULONGLONG* outFooterPos,
                                                  bool* outIsComplete) {
    ULONGLONG footerPos = 0;
    if (outIsComplete) *outIsComplete = false;

    // PNG 文件特殊处理：通过遍历 chunk 结构查找 IEND
    if (sig.extension == "png") {
        footerPos = FindPngEndByChunksStatic(data, dataSize);
        if (footerPos > 0) {
            if (outFooterPos) *outFooterPos = footerPos;
            if (outIsComplete) *outIsComplete = true;
            return footerPos;
        }
        // 如果遍历失败，返回保守估计
        if (outFooterPos) *outFooterPos = 0;
        return min((ULONGLONG)dataSize, sig.maxSize);
    }

    // ZIP 文件特殊处理：使用EOCD定位文件结束，并验证结构合理性
    if (sig.extension == "zip") {
        footerPos = FindZipEndOfCentralDirectoryStatic(data, dataSize);
        if (footerPos > 0) {
            if (outFooterPos) *outFooterPos = footerPos;

            // 验证 ZIP 结构合理性（排除自解压等情况）
            if (footerPos >= 22 && dataSize >= 4) {
                size_t eocdStart = footerPos - 22;
                DWORD cdOffset = *(DWORD*)(data + eocdStart + 16);

                bool isStandardZip = (data[0] == 0x50 && data[1] == 0x4B &&
                                     data[2] == 0x03 && data[3] == 0x04);

                if (!isStandardZip) {
                    for (size_t i = 0; i < min((size_t)cdOffset, dataSize - 4); i++) {
                        if (data[i] == 0x50 && data[i + 1] == 0x4B &&
                            data[i + 2] == 0x03 && data[i + 3] == 0x04) {
                            ULONGLONG actualSize = footerPos - i;
                            if (outIsComplete) *outIsComplete = true;
                            return actualSize;
                        }
                    }
                }

                if (cdOffset > footerPos - 22) {
                    if (outIsComplete) *outIsComplete = false;
                } else {
                    if (outIsComplete) *outIsComplete = true;
                }
            } else {
                if (outIsComplete) *outIsComplete = true;
            }
            return footerPos;
        }

        // 如果找不到 EOCD，返回0表示无效
        if (outFooterPos) *outFooterPos = 0;
        if (outIsComplete) *outIsComplete = false;
        return 0;
    }

    // PDF 和其他文件尾在末尾的格式：使用反向搜索
    if (sig.hasFooter && !sig.footer.empty()) {
        if (sig.extension == "pdf") {
            footerPos = FindFooterReverseStatic(data, dataSize, sig.footer, min((ULONGLONG)dataSize, sig.maxSize));
        } else {
            footerPos = FindFooterStatic(data, dataSize, sig.footer, min((ULONGLONG)dataSize, sig.maxSize));
        }

        if (outFooterPos) *outFooterPos = footerPos;
        if (footerPos > 0) {
            if (outIsComplete) *outIsComplete = true;
            return footerPos;
        }
    }

    // 特殊格式处理：从头部读取大小
    if (sig.extension == "bmp" && dataSize >= 6) {
        DWORD size = *(DWORD*)(data + 2);
        if (size > sig.minSize && size <= sig.maxSize) {
            if (outIsComplete) *outIsComplete = (size <= dataSize);
            return size;
        }
    }

    if (sig.extension == "mp4" && dataSize >= 12) {
        if (data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p') {
            DWORD atomSize = _byteswap_ulong(*(DWORD*)data);
            if (atomSize >= 8) {
                return min((ULONGLONG)dataSize, sig.maxSize);
            }
        }
    }

    if (sig.extension == "avi" && dataSize >= 12) {
        if (data[8] == 'A' && data[9] == 'V' && data[10] == 'I' && data[11] == ' ') {
            DWORD riffSize = *(DWORD*)(data + 4);
            if (riffSize > 0 && riffSize <= sig.maxSize) {
                ULONGLONG totalSize = (ULONGLONG)riffSize + 8;
                if (outIsComplete) *outIsComplete = (totalSize <= dataSize);
                return totalSize;
            }
        }
    }

    if (sig.extension == "wav" && dataSize >= 12) {
        if (data[8] == 'W' && data[9] == 'A' && data[10] == 'V' && data[11] == 'E') {
            DWORD riffSize = *(DWORD*)(data + 4);
            if (riffSize > 0 && riffSize <= sig.maxSize) {
                ULONGLONG totalSize = (ULONGLONG)riffSize + 8;
                if (outIsComplete) *outIsComplete = (totalSize <= dataSize);
                return totalSize;
            }
        }
    }

    // 默认：使用最大大小或可用数据大小
    return min((ULONGLONG)dataSize, sig.maxSize);
}

// ============================================================================
// 估算文本文件大小（ML 模式）
// ============================================================================

ULONGLONG FileFormatUtils::EstimateFileSizeML(const BYTE* data, size_t maxSize, const string& type) {
    if (data == nullptr || maxSize == 0) return 0;

    if (type == "txt" || type == "html" || type == "xml") {
        // 文本文件：扫描到第一个 NULL 字节序列
        size_t nullCount = 0;
        for (size_t i = 0; i < maxSize; i++) {
            if (data[i] == 0) {
                nullCount++;
                if (nullCount >= 8) {  // 连续8个NULL视为结束
                    return i - nullCount + 1;
                }
            } else {
                nullCount = 0;
            }
        }

        // 未找到结束，使用默认大小
        if (type == "txt") return min(maxSize, (size_t)64 * 1024);      // 64KB
        if (type == "html") return min(maxSize, (size_t)256 * 1024);    // 256KB
        if (type == "xml") return min(maxSize, (size_t)128 * 1024);     // 128KB
    }

    return min(maxSize, (size_t)64 * 1024);  // 默认 64KB
}

// ============================================================================
// 验证 ZIP 数据完整性
// ============================================================================

FileFormatUtils::ZipValidationResult FileFormatUtils::ValidateZipData(const BYTE* data, size_t dataSize) {
    ZipValidationResult result;
    result.actualSize = dataSize;

    if (dataSize < 22) {
        result.diagnosis = "Data too small for ZIP file";
        return result;
    }

    // 1. 查找并验证 EOCD
    ULONGLONG eocdPos = FindZipEndOfCentralDirectoryStatic(data, dataSize);
    if (eocdPos == 0) {
        result.diagnosis = "EOCD not found";
        return result;
    }

    // 向后搜索找到 EOCD 起始位置
    size_t eocdStart = 0;
    for (size_t i = dataSize - 22; i > 0; i--) {
        if (data[i] == 0x50 && data[i + 1] == 0x4B &&
            data[i + 2] == 0x05 && data[i + 3] == 0x06) {
            WORD commentLen = *(WORD*)(data + i + 20);
            if (i + 22 + commentLen <= dataSize) {
                eocdStart = i;
                break;
            }
        }
    }

    if (eocdStart == 0) {
        result.diagnosis = "Could not locate EOCD start";
        return result;
    }

    WORD totalEntries = *(WORD*)(data + eocdStart + 10);
    DWORD cdSize = *(DWORD*)(data + eocdStart + 12);
    DWORD cdOffset = *(DWORD*)(data + eocdStart + 16);

    result.totalFiles = totalEntries;

    // 2. 验证 Central Directory 偏移
    if (cdOffset >= dataSize || cdOffset + cdSize > dataSize) {
        result.diagnosis = "Invalid Central Directory offset/size";
        return result;
    }

    // 3. 遍历 Central Directory 验证每个条目
    size_t cdPos = cdOffset;
    int validEntries = 0;
    int crcErrors = 0;

    while (cdPos + 46 <= eocdStart && validEntries < totalEntries) {
        if (data[cdPos] != 0x50 || data[cdPos + 1] != 0x4B ||
            data[cdPos + 2] != 0x01 || data[cdPos + 3] != 0x02) {
            result.diagnosis = "Invalid CD entry signature at offset " + to_string(cdPos);
            break;
        }

        WORD compression = *(WORD*)(data + cdPos + 10);
        DWORD expectedCRC = *(DWORD*)(data + cdPos + 16);
        DWORD compressedSize = *(DWORD*)(data + cdPos + 20);
        DWORD uncompressedSize = *(DWORD*)(data + cdPos + 24);
        WORD filenameLen = *(WORD*)(data + cdPos + 28);
        WORD extraLen = *(WORD*)(data + cdPos + 30);
        WORD commentLen = *(WORD*)(data + cdPos + 32);
        DWORD localHeaderOffset = *(DWORD*)(data + cdPos + 42);

        // 验证 Local File Header 存在
        if (localHeaderOffset + 30 > dataSize) {
            result.diagnosis = "Invalid local header offset for entry " + to_string(validEntries);
            break;
        }

        // 检查 Local File Header 签名
        if (data[localHeaderOffset] != 0x50 || data[localHeaderOffset + 1] != 0x4B ||
            data[localHeaderOffset + 2] != 0x03 || data[localHeaderOffset + 3] != 0x04) {
            result.diagnosis = "Invalid local header signature for entry " + to_string(validEntries);
            break;
        }

        WORD localFilenameLen = *(WORD*)(data + localHeaderOffset + 26);
        WORD localExtraLen = *(WORD*)(data + localHeaderOffset + 28);

        size_t dataStart = localHeaderOffset + 30 + localFilenameLen + localExtraLen;

        // CRC 验证（仅对未压缩文件）
        if (compression == 0 && compressedSize > 0 && dataStart + compressedSize <= dataSize) {
            DWORD actualCRC = 0xFFFFFFFF;
            for (size_t i = 0; i < compressedSize; i++) {
                BYTE b = data[dataStart + i];
                actualCRC ^= b;
                for (int j = 0; j < 8; j++) {
                    if (actualCRC & 1) {
                        actualCRC = (actualCRC >> 1) ^ 0xEDB88320;
                    } else {
                        actualCRC >>= 1;
                    }
                }
            }
            actualCRC ^= 0xFFFFFFFF;

            if (actualCRC != expectedCRC) {
                crcErrors++;
            }
        }

        validEntries++;
        cdPos += 46 + filenameLen + extraLen + commentLen;
    }

    result.corruptedFiles = crcErrors;
    result.crcValid = (crcErrors == 0 && validEntries == totalEntries);

    if (result.crcValid) {
        result.success = true;
        result.diagnosis = "ZIP structure valid, " + to_string(validEntries) +
                          " files verified";
    } else if (validEntries < totalEntries) {
        result.diagnosis = "Only " + to_string(validEntries) + " of " +
                          to_string(totalEntries) + " entries parsed";
    } else if (crcErrors > 0) {
        result.diagnosis = to_string(crcErrors) + " of " + to_string(totalEntries) +
                          " files have CRC errors";
    }

    return result;
}
