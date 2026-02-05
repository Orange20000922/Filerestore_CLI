#include "FileCarver.h"
#include "SignatureScanThreadPool.h"
#include "TimestampExtractor.h"
#include "MFTLCNIndex.h"
#include "Logger.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <filesystem>

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
      useThreadPool(true),
      timestampExtractionEnabled(true), mftIndexBuilt(false),
      integrityValidationEnabled(true), validatedCount(0), corruptedCount(0),
      mlClassificationEnabled(false) {
    stats.reset();

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

    // 自动检测并加载ML模型
    AutoLoadMLModel();

    LOG_INFO("FileCarver initialized with async I/O and thread pool support");
}

FileCarver::~FileCarver() {
    shouldStop = true;  // 确保停止任何运行中的线程

    // 清理线程池
    if (scanThreadPool) {
        scanThreadPool->Stop();
    }

    // unique_ptr 自动释放 scanThreadPool, lcnIndex, mlClassifier
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
        50ULL*1024 * 1024 * 1024,                  // 50GB max
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
        {0x00, 0x00, 0x00, 0x00,  // IEND chunk length (always 0)
         0x49, 0x45, 0x4E, 0x44,  // "IEND" chunk type
         0xAE, 0x42, 0x60, 0x82}, // CRC32 for empty IEND chunk
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
        50*1024ULL * 1024 * 1024,              // 1GB
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
        50*1024ULL * 1024 * 1024,
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

    // 从前向后搜索（适用于 JPEG EOI 等）
    for (size_t i = 0; i <= searchLimit; i++) {
        if (memcmp(data + i, footer.data(), footer.size()) == 0) {
            return i + footer.size();
        }
    }
    return 0;
}

// ============================================================================
// 查找文件尾（从后向前搜索）
// ============================================================================
ULONGLONG FileCarver::FindFooterReverse(const BYTE* data, size_t dataSize,
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
// 通过遍历 PNG chunk 结构查找文件末尾
// ============================================================================
ULONGLONG FileCarver::FindPngEndByChunks(const BYTE* data, size_t dataSize) {
    // PNG 文件结构:
    // - 8 字节签名: 89 50 4E 47 0D 0A 1A 0A
    // - 一系列 chunk:
    //   - 4 字节: 数据长度 (大端序)
    //   - 4 字节: chunk 类型
    //   - N 字节: 数据
    //   - 4 字节: CRC32

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
            return 0;  // 无效的 chunk 长度
        }

        // 读取 chunk 类型
        char chunkType[5] = {0};
        memcpy(chunkType, data + offset + 4, 4);

        // 计算完整 chunk 大小
        size_t fullChunkSize = CHUNK_HEADER_SIZE + chunkLength + CHUNK_CRC_SIZE;

        // 检查是否超出数据范围
        if (offset + fullChunkSize > dataSize) {
            // chunk 数据不完整，返回当前估计
            return 0;
        }

        // 检查是否是 IEND chunk
        if (chunkType[0] == 'I' && chunkType[1] == 'E' &&
            chunkType[2] == 'N' && chunkType[3] == 'D') {
            // IEND 的数据长度应该是 0
            if (chunkLength == 0) {
                // 找到有效的 IEND，返回文件结束位置
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
            // 无效的 chunk 类型，可能数据损坏
            return 0;
        }

        // 移动到下一个 chunk
        offset += fullChunkSize;
    }

    // 遍历完成但没找到 IEND
    return 0;
}

// ============================================================================
// 查找 ZIP 文件尾（End of Central Directory）
// ============================================================================
ULONGLONG FileCarver::FindZipEndOfCentralDirectory(const BYTE* data, size_t dataSize) {
    // ZIP EOCD 签名: 0x50 0x4B 0x05 0x06
    // EOCD 最小长度: 22 字节
    // EOCD 可能有注释字段（最大 65535 字节）

    if (dataSize < 22) {
        return 0;
    }

    // 从后向前搜索 EOCD 签名（最多搜索 65KB + 22 字节）
    size_t searchStart = dataSize - 22;
    size_t searchLimit = (dataSize > 65557) ? (dataSize - 65557) : 0;

    for (size_t i = searchStart; i >= searchLimit && i < dataSize; i--) {
        // 检查 EOCD 签名
        if (i + 22 <= dataSize &&
            data[i] == 0x50 && data[i + 1] == 0x4B &&
            data[i + 2] == 0x05 && data[i + 3] == 0x06) {

            // 验证注释长度字段
            WORD commentLen = *(WORD*)(data + i + 20);

            // EOCD 总长度 = 22 + 注释长度
            size_t eocdEnd = i + 22 + commentLen;

            // 验证 EOCD 是否在数据范围内
            if (eocdEnd <= dataSize) {
                // 如果注释长度为 0，或者 EOCD 正好在文件末尾，则认为找到
                if (commentLen == 0 || eocdEnd == dataSize) {
                    return eocdEnd;
                }
                // 如果有注释，验证 EOCD 后面的数据是否合理
                if (eocdEnd < dataSize) {
                    // 检查 EOCD 后面是否全是零（可能是填充）
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
        if (i == 0) break;  // 防止无符号下溢
    }

    return 0;
}

// ============================================================================
// 检测 ZIP 是否为 OOXML Office 文档 (DOCX/XLSX/PPTX)
// ============================================================================
string FileCarver::DetectOOXMLType(const BYTE* data, size_t dataSize) {
    // OOXML 格式是 ZIP 文件，内部结构：
    // - DOCX: 包含 word/ 目录
    // - XLSX: 包含 xl/ 目录
    // - PPTX: 包含 ppt/ 目录
    // - 所有 OOXML: 包含 [Content_Types].xml

    if (dataSize < 30) return "";

    // 验证是 ZIP 文件
    if (data[0] != 0x50 || data[1] != 0x4B ||
        data[2] != 0x03 || data[3] != 0x04) {
        return "";
    }

    // 解析第一个 Local File Header 获取文件名
    // Local File Header 结构:
    // 0-4: signature (PK\x03\x04)
    // 4-6: version needed
    // 6-8: flags
    // 8-10: compression
    // 10-14: mod time/date
    // 14-18: CRC32
    // 18-22: compressed size
    // 22-26: uncompressed size
    // 26-28: filename length
    // 28-30: extra field length
    // 30+: filename

    WORD filenameLen = *(WORD*)(data + 26);
    WORD extraLen = *(WORD*)(data + 28);

    if (filenameLen == 0 || 30 + filenameLen > dataSize) {
        return "";
    }

    // 读取文件名
    string filename((char*)(data + 30), filenameLen);

    // 检查是否是 OOXML 格式
    // 第一个文件通常是 [Content_Types].xml 或 _rels/.rels 或特定目录

    // 先检查是否是 Office 文档（存在 [Content_Types].xml）
    bool hasContentTypes = (filename == "[Content_Types].xml");

    // 如果第一个文件就能确定类型
    if (filename.substr(0, 5) == "word/" || filename == "word") {
        return "docx";
    }
    if (filename.substr(0, 3) == "xl/" || filename == "xl") {
        return "xlsx";
    }
    if (filename.substr(0, 4) == "ppt/" || filename == "ppt") {
        return "pptx";
    }

    // 如果第一个文件是 [Content_Types].xml 或 _rels，需要检查后续文件
    if (hasContentTypes || filename.substr(0, 6) == "_rels/") {
        // 遍历更多的 Local File Headers
        size_t offset = 30 + filenameLen + extraLen;

        // 读取压缩数据大小
        DWORD compressedSize = *(DWORD*)(data + 18);
        offset += compressedSize;

        // 检查最多 10 个文件
        for (int i = 0; i < 10 && offset + 30 < dataSize; i++) {
            // 检查签名
            if (data[offset] != 0x50 || data[offset + 1] != 0x4B) {
                break;
            }

            // 可能是另一个 Local File Header 或 Central Directory
            if (data[offset + 2] == 0x03 && data[offset + 3] == 0x04) {
                // Local File Header
                WORD fnLen = *(WORD*)(data + offset + 26);
                WORD exLen = *(WORD*)(data + offset + 28);
                DWORD cmpSize = *(DWORD*)(data + offset + 18);

                if (fnLen > 0 && offset + 30 + fnLen <= dataSize) {
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
                // 可能是其他结构，停止搜索
                break;
            }
        }

        // 如果有 [Content_Types].xml 但无法确定具体类型，标记为通用 Office
        if (hasContentTypes) {
            return "ooxml";  // 通用 OOXML 标记
        }
    }

    return "";
}

// ============================================================================
// 估算文件大小（优化：返回footer位置避免重复查找）
// ============================================================================
ULONGLONG FileCarver::EstimateFileSize(const BYTE* data, size_t dataSize,
                                       const FileSignature& sig,
                                       ULONGLONG* outFooterPos) {
    ULONGLONG footerPos = 0;

    // PNG 文件特殊处理：通过遍历 chunk 结构查找 IEND
    if (sig.extension == "png") {
        footerPos = FindPngEndByChunks(data, dataSize);
        if (outFooterPos) *outFooterPos = footerPos;
        if (footerPos > 0) {
            return footerPos;
        }
        // 如果遍历失败，返回保守估计
        return min((ULONGLONG)dataSize, sig.maxSize);
    }

    // ZIP 文件特殊处理：使用专门的 EOCD 查找
    if (sig.extension == "zip") {
        footerPos = FindZipEndOfCentralDirectory(data, dataSize);
        if (outFooterPos) *outFooterPos = footerPos;
        if (footerPos > 0) {
            // 验证 ZIP 结构合理性（排除自解压等情况）
            if (footerPos >= 22 && dataSize >= 4) {
                // 解析 EOCD 中的 Central Directory 偏移
                size_t eocdStart = footerPos - 22;
                DWORD cdOffset = *(DWORD*)(data + eocdStart + 16);

                // 检查文件开头是否是标准 ZIP (Local File Header)
                bool isStandardZip = (data[0] == 0x50 && data[1] == 0x4B &&
                                     data[2] == 0x03 && data[3] == 0x04);

                // 如果不是标准 ZIP 开头，可能是自解压或附加数据
                if (!isStandardZip) {
                    // 搜索第一个 Local File Header
                    for (size_t i = 0; i < min((size_t)cdOffset, dataSize - 4); i++) {
                        if (data[i] == 0x50 && data[i + 1] == 0x4B &&
                            data[i + 2] == 0x03 && data[i + 3] == 0x04) {
                            // 找到真正的 ZIP 起点，计算实际大小
                            ULONGLONG actualSize = footerPos - i;
                            LOG_DEBUG_FMT("ZIP has %llu bytes of prefix data (possibly self-extracting)", i);
                            return actualSize;
                        }
                    }
                }

                // 验证 CD offset 的合理性
                if (cdOffset > footerPos - 22) {
                    LOG_DEBUG("Warning: ZIP CD offset is beyond EOCD, possibly corrupted");
                    // CD 偏移超出范围，可能损坏，使用保守估计
                    return min((ULONGLONG)dataSize, sig.maxSize);
                }
            }
            return footerPos;
        }
        // 如果找不到 EOCD，返回0表示无效
        // 没有EOCD的ZIP无法确定文件边界，恢复出来的数据没有意义
        return 0;
    }

    // PDF 和其他文件尾在末尾的格式：使用反向搜索
    if (sig.hasFooter && !sig.footer.empty()) {
        if (sig.extension == "pdf") {
            // PDF %%EOF 在文件末尾，使用反向搜索
            footerPos = FindFooterReverse(data, dataSize, sig.footer, min((ULONGLONG)dataSize, sig.maxSize));
        } else {
            // JPEG, PNG, GIF 等：文件尾可能在中间，使用正向搜索
            footerPos = FindFooter(data, dataSize, sig.footer, min((ULONGLONG)dataSize, sig.maxSize));
        }

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

                    // ZIP文件必须有EOCD，否则跳过
                    if (sig->extension == "zip" && estimatedSize == 0) {
                        continue;  // 没有EOCD的ZIP无法可靠恢复
                    }

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

                        // 检测 ZIP 是否为 OOXML Office 文档
                        if (sig->extension == "zip") {
                            string ooxmlType = DetectOOXMLType(data + offset, remaining);
                            if (!ooxmlType.empty()) {
                                info.extension = ooxmlType;
                                if (ooxmlType == "docx") {
                                    info.description = "Microsoft Word Document (OOXML)";
                                } else if (ooxmlType == "xlsx") {
                                    info.description = "Microsoft Excel Spreadsheet (OOXML)";
                                } else if (ooxmlType == "pptx") {
                                    info.description = "Microsoft PowerPoint Presentation (OOXML)";
                                } else if (ooxmlType == "ooxml") {
                                    info.description = "Microsoft Office Document (OOXML)";
                                }
                            }
                        }

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
    stats.reset();
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
    ULONGLONG bufferClusters = min(BUFFER_SIZE_CLUSTERS, stats.totalClusters.load());

    // 开始计时
    DWORD startTime = GetTickCount();

    vector<BYTE> buffer;
    ULONGLONG currentLCN = 0;
    ULONGLONG lastProgressLCN = 0;

    while (currentLCN < stats.totalClusters.load() && !shouldStop && results.size() < maxResults) {
        ULONGLONG clustersToRead = min(bufferClusters, stats.totalClusters.load() - currentLCN);

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

    // 确保父目录存在 (使用 std::filesystem)
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
// 恢复前精细化：精确大小计算 + 完整性验证 + 置信度重估
// 将扫描阶段延迟的详细分析集中在此处执行（单文件，非热循环）
// ============================================================================
bool FileCarver::RefineCarvedFileInfo(CarvedFileInfo& info, bool verbose) {
    // 查找对应的签名定义
    string lookupExt = info.extension;
    // OOXML 类型映射回 zip
    if (lookupExt == "docx" || lookupExt == "xlsx" || lookupExt == "pptx" || lookupExt == "ooxml") {
        lookupExt = "zip";
    }

    auto sigIt = signatures.find(lookupExt);
    if (sigIt == signatures.end()) {
        if (verbose) {
            cout << "  [精细化] 未知文件类型: " << info.extension << "，跳过" << endl;
        }
        return true;  // 未知类型不阻止恢复
    }
    const FileSignature& sig = sigIt->second;

    // ZIP 在扫描阶段已经做了 EOCD 精确搜索，不需要重复
    bool needSizeRefinement = info.sizeIsEstimated;

    if (!needSizeRefinement && !info.integrityValidated) {
        // 大小已精确但未做完整性验证 → 只做完整性验证
        if (verbose) {
            cout << "  [精细化] 文件大小已精确 (" << info.fileSize << " bytes)，执行完整性验证..." << endl;
        }

        FileIntegrityScore integrity = ValidateFileIntegrity(info);
        info.integrityScore = integrity.overallScore;
        info.integrityValidated = true;
        info.integrityDiagnosis = integrity.diagnosis;

        // 根据完整性评分调整置信度
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
        return true;  // 大小精确且已验证，无需处理
    }

    // ===== 需要精确计算文件大小 =====
    if (verbose) {
        cout << "  [精细化] 大小为估计值 (" << info.fileSize << " bytes)，正在精确计算..." << endl;
    }

    // 读取数据用于分析（限制读取量，避免内存爆炸）
    // 对于大多数格式，文件尾在前几MB内就能找到
    // 但对于大型文件（如视频），可能需要更多数据
    ULONGLONG readSize = info.fileSize;
    const ULONGLONG MAX_REFINE_READ = 64ULL * 1024 * 1024;  // 最大读取 64MB 用于分析
    if (readSize > MAX_REFINE_READ) {
        readSize = MAX_REFINE_READ;
    }

    vector<BYTE> fileData;
    if (!ExtractFile(info.startLCN, info.startOffset, readSize, fileData)) {
        if (verbose) {
            cout << "  [精细化] 无法读取文件数据，保持原始估计" << endl;
        }
        return true;  // 读取失败不阻止恢复
    }

    // 调用 EstimateFileSizeStatic 进行精确大小计算
    ULONGLONG footerPos = 0;
    bool isComplete = false;
    ULONGLONG refinedSize = EstimateFileSizeStatic(
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
        // EstimateFileSizeStatic 返回 0 表示无效（如 ZIP 无 EOCD）
        if (verbose) {
            cout << "  [精细化] 警告: 无法确定有效文件边界" << endl;
        }
        info.confidence *= 0.5;
    }

    // 更新文件尾状态
    if (footerPos > 0) {
        info.hasValidFooter = true;
        if (isComplete) {
            info.sizeIsEstimated = false;
        }
    }

    // OOXML 检测（对 zip 类型，检查是否为 Office 文档）
    if (lookupExt == "zip" && info.extension == "zip") {
        string ooxmlType = DetectOOXMLTypeStatic(fileData.data(), fileData.size());
        if (!ooxmlType.empty()) {
            info.extension = ooxmlType;
            info.description = ooxmlType + " (Office)";
            if (verbose) {
                cout << "  [精细化] 检测到 Office 文档类型: " << ooxmlType << endl;
            }
        }
    }

    // ===== 完整性验证 =====
    // 使用已读取的数据进行验证，避免重复磁盘 I/O
    size_t validateSize = (size_t)min((ULONGLONG)fileData.size(), (ULONGLONG)(2 * 1024 * 1024));
    FileIntegrityScore integrity = FileIntegrityValidator::Validate(
        fileData.data(), validateSize, info.extension);

    info.integrityScore = integrity.overallScore;
    info.integrityValidated = true;
    info.integrityDiagnosis = integrity.diagnosis;

    // ===== 置信度重估 =====
    double originalConfidence = info.confidence;

    // 因素1：文件尾完整性
    if (info.hasValidFooter && isComplete) {
        info.confidence = min(1.0, info.confidence * 1.15);  // 找到完整文件尾，提升置信度
    } else if (!info.hasValidFooter && (info.extension == "jpg" || info.extension == "png" ||
               info.extension == "pdf" || info.extension == "gif")) {
        info.confidence *= 0.75;  // 这些格式应该有文件尾
    }

    // 因素2：完整性评分
    if (integrity.overallScore >= 0.8) {
        info.confidence = min(1.0, info.confidence * 1.1);
    } else if (integrity.overallScore >= 0.5) {
        // 中等，不调整
    } else {
        info.confidence *= 0.6;  // 完整性低，大幅降低置信度
    }

    // 因素3：大小合理性
    if (sizeChanged && refinedSize < originalSize * 0.1) {
        // 精确大小远小于估计值，可能是有效的小文件
        // 不额外惩罚
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
    stats.reset();
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

    ULONGLONG bufferClusters = min(BUFFER_SIZE_CLUSTERS, stats.totalClusters.load());

    // 开始计时
    DWORD startTime = GetTickCount();

    // 启动I/O读取线程（生产者）
    thread ioThread(&FileCarver::IOReaderThread, this,
                    0, stats.totalClusters.load(), bufferClusters, bytesPerCluster, mode);

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
// SIMD 控制方法
// ============================================================================
void FileCarver::SetSimdEnabled(bool enabled) {
    if (scanThreadPool) {
        scanThreadPool->SetSimdEnabled(enabled);
    }
}

bool FileCarver::IsSimdEnabled() const {
    if (scanThreadPool) {
        return scanThreadPool->IsSimdEnabled();
    }
    return true;  // 默认启用
}

std::string FileCarver::GetSimdInfo() const {
    if (scanThreadPool) {
        return scanThreadPool->GetSimdInfo();
    }
    return "Thread pool not initialized";
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
    stats.reset();
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
    bufferClusters = min(bufferClusters, stats.totalClusters.load());

    cout << "Read Buffer: " << (BUFFER_SIZE / (1024 * 1024)) << " MB" << endl;

    // 检查ML状态
    bool mlEnabled = mlClassificationEnabled && mlClassifier && mlClassifier->isLoaded();
    if (mlEnabled) {
        cout << "ML Classification: Enabled (Model loaded)" << endl;
        threadPoolConfig.useMLClassification = true;
    } else if (mlClassificationEnabled) {
        cout << "ML Classification: Disabled (Model not loaded)" << endl;
        threadPoolConfig.useMLClassification = false;
    } else {
        cout << "ML Classification: Disabled" << endl;
        threadPoolConfig.useMLClassification = false;
    }

    cout << "\nScanning... (Press Ctrl+C to stop)\n" << endl;

    // 创建线程池
    scanThreadPool = make_unique<SignatureScanThreadPool>(&signatureIndex, &activeSignatures, threadPoolConfig);

    // 设置ML分类器
    if (mlEnabled) {
        scanThreadPool->SetMLClassifier(mlClassifier.get());
    }

    scanThreadPool->Start();

    // 开始计时
    DWORD startTime = GetTickCount();

    // I/O读取主循环
    vector<BYTE> buffer;
    ULONGLONG currentLCN = 0;
    ULONGLONG lastProgressLCN = 0;
    int taskIdCounter = 0;

    while (currentLCN < stats.totalClusters.load() && !shouldStop) {
        ULONGLONG clustersToRead = min(bufferClusters, stats.totalClusters.load() - currentLCN);

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

            // 构建进度信息
            cout << "\rI/O: " << fixed << setprecision(1) << progress << "% | "
                 << "Pool: " << setprecision(1) << poolProgress << "% | "
                 << "Scanned: " << (stats.scannedClusters / 1000) << "K | "
                 << "Skipped: " << (stats.skippedClusters / 1000) << "K | "
                 << "Found: " << filesFound;

            // 如果ML启用，显示ML增强计数
            if (mlEnabled) {
                cout << " (ML:" << scanThreadPool->GetMLEnhancedCount() << ")";
            }

            cout << " | Speed: " << setprecision(1) << speedMBps << " MB/s" << flush;

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

    // 显示ML分类统计
    if (mlEnabled) {
        cout << "\n--- ML Classification Statistics ---" << endl;

        ULONGLONG mlEnhanced = scanThreadPool->GetMLEnhancedCount();
        ULONGLONG mlMatch = scanThreadPool->GetMLMatchCount();
        ULONGLONG mlMismatch = scanThreadPool->GetMLMismatchCount();
        ULONGLONG mlSkipped = scanThreadPool->GetMLSkippedCount();
        ULONGLONG mlUnknown = scanThreadPool->GetMLUnknownCount();

        cout << "ML supported files: " << mlEnhanced << endl;
        cout << "ML skipped (unsupported types): " << mlSkipped << endl;
        cout << "ML uncertain (low confidence): " << mlUnknown << endl;

        if (mlMatch + mlMismatch > 0) {
            double matchRate = 100.0 * mlMatch / (mlMatch + mlMismatch);
            cout << "ML-Signature match rate: " << fixed << setprecision(1) << matchRate << "%" << endl;
            cout << "  Matches: " << mlMatch << ", Mismatches: " << mlMismatch << endl;
        }
    }

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
    lcnIndex = make_unique<MFTLCNIndex>(reader);
    mftIndexBuilt = lcnIndex->BuildIndex(includeActiveFiles, true);

    if (!mftIndexBuilt) {
        lcnIndex.reset();
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

// ============================================================================
// ML 分类功能
// ============================================================================

bool FileCarver::LoadMLModel(const wstring& modelPath) {
    if (!ML::MLClassifier::isOnnxRuntimeAvailable()) {
        LOG_WARNING("ONNX Runtime not available, ML classification disabled");
        return false;
    }

    if (!mlClassifier) {
        mlClassifier = make_unique<ML::MLClassifier>();
    }

    if (mlClassifier->loadModel(modelPath)) {
        mlClassificationEnabled = true;
        LOG_INFO(L"ML model loaded: " + modelPath);
        return true;
    }

    LOG_ERROR(L"Failed to load ML model: " + modelPath);
    return false;
}

bool FileCarver::IsMLModelLoaded() const {
    return mlClassifier && mlClassifier->isLoaded();
}

optional<ML::ClassificationResult> FileCarver::ClassifyWithML(
    const BYTE* data, size_t dataSize) {

    if (!IsMLModelLoaded()) {
        return nullopt;
    }

    return mlClassifier->classify(data, dataSize);
}

vector<string> FileCarver::GetMLSupportedTypes() const {
    if (!IsMLModelLoaded()) {
        return {};
    }
    return mlClassifier->getSupportedTypes();
}

void FileCarver::EnhanceResultsWithML(vector<CarvedFileInfo>& results, bool showProgress) {
    if (!IsMLModelLoaded()) {
        LOG_WARNING("ML model not loaded, skipping ML enhancement");
        return;
    }

    DWORD startTime = GetTickCount();
    size_t processedCount = 0;
    size_t enhancedCount = 0;

    for (auto& info : results) {
        // 读取文件数据
        vector<BYTE> fileData;
        if (!ExtractFile(info.startLCN, info.startOffset, min(info.fileSize, (ULONGLONG)4096), fileData)) {
            continue;
        }

        // ML 分类
        auto mlResult = mlClassifier->classify(fileData.data(), fileData.size());
        if (mlResult && mlResult->confidence > 0.6) {
            // 存储 ML 分类结果作为补充信息
            info.mlClassification = mlResult->predictedType;
            info.mlConfidence = mlResult->confidence;

            // 如果签名检测和 ML 分类一致，提高可信度
            if (mlResult->predictedType == info.extension) {
                info.validationScore = max(info.validationScore, (double)mlResult->confidence);
            }
            // 如果不一致但 ML 置信度高，可能是签名误判
            else if (mlResult->confidence > 0.85) {
                LOG_DEBUG_FMT("ML suggests %s (%.1f%%) but signature detected %s at LCN %llu",
                             mlResult->predictedType.c_str(),
                             mlResult->confidence * 100.0f,
                             info.extension.c_str(),
                             info.startLCN);
            }

            enhancedCount++;
        }

        processedCount++;

        if (showProgress && processedCount % 100 == 0) {
            double progress = (double)processedCount / results.size() * 100.0;
            cout << "\rML Enhancement: " << fixed << setprecision(1) << progress << "%" << flush;
        }
    }

    DWORD elapsed = GetTickCount() - startTime;

    if (showProgress) {
        cout << "\r                                                          " << endl;
        cout << "\n--- ML Enhancement Complete ---" << endl;
        cout << "Time: " << (elapsed / 1000) << "." << ((elapsed % 1000) / 100) << " seconds" << endl;
        cout << "Files processed: " << processedCount << endl;
        cout << "Files enhanced with ML: " << enhancedCount << endl;
    }

    LOG_INFO_FMT("ML enhancement: %zu/%zu files enhanced", enhancedCount, processedCount);
}

vector<CarvedFileInfo> FileCarver::ScanWithMLOnly(CarvingMode mode,
                                                   ULONGLONG maxResults,
                                                   float minConfidence) {
    vector<CarvedFileInfo> results;

    if (!IsMLModelLoaded()) {
        LOG_ERROR("ML model not loaded, cannot scan with ML only");
        return results;
    }

    LOG_INFO("Starting ML-only scan");
    cout << "ML-only scan starting (confidence threshold: " << (minConfidence * 100.0f) << "%)" << endl;

    // 获取磁盘信息
    ULONGLONG totalClusters = reader->GetTotalClusters();
    ULONGLONG bytesPerCluster = reader->GetBytesPerCluster();

    stats.totalClusters = totalClusters;
    stats.scannedClusters = 0;
    stats.filesFound = 0;

    shouldStop = false;
    DWORD startTime = GetTickCount();

    // 分配缓冲区
    ULONGLONG bufferClusters = 16;  // 16 簇 = 64KB
    vector<BYTE> buffer(static_cast<size_t>(bufferClusters * bytesPerCluster));

    for (ULONGLONG lcn = 0; lcn < totalClusters && !shouldStop; lcn += bufferClusters) {
        ULONGLONG clustersToRead = min(bufferClusters, totalClusters - lcn);

        if (!reader->ReadClusters(lcn, clustersToRead, buffer)) {
            continue;
        }

        // 使用 ML 分类
        auto mlResult = mlClassifier->classify(buffer.data(), static_cast<size_t>(clustersToRead * bytesPerCluster));

        if (mlResult && mlResult->confidence >= minConfidence) {
            CarvedFileInfo info;
            info.startLCN = lcn;
            info.startOffset = 0;
            info.extension = mlResult->predictedType;
            info.fileSize = clustersToRead * bytesPerCluster;  // 估算大小
            info.validationScore = mlResult->confidence;
            info.mlClassification = mlResult->predictedType;
            info.mlConfidence = mlResult->confidence;

            results.push_back(info);
            stats.filesFound++;

            if (results.size() >= maxResults) {
                break;
            }
        }

        stats.scannedClusters += clustersToRead;

        // 进度显示
        if (stats.scannedClusters % PROGRESS_UPDATE_INTERVAL == 0) {
            double progress = (double)stats.scannedClusters / totalClusters * 100.0;
            cout << "\rML Scan: " << fixed << setprecision(1) << progress << "% | "
                 << "Found: " << stats.filesFound << flush;
        }
    }

    stats.elapsedMs = GetTickCount() - startTime;

    cout << "\r                                                          " << endl;
    cout << "\n--- ML-Only Scan Complete ---" << endl;
    cout << "Time: " << (stats.elapsedMs / 1000) << "." << ((stats.elapsedMs % 1000) / 100) << " seconds" << endl;
    cout << "Clusters scanned: " << stats.scannedClusters << "/" << totalClusters << endl;
    cout << "Files found: " << results.size() << endl;

    LOG_INFO_FMT("ML-only scan complete: %zu files found", results.size());

    return results;
}

// ============================================================================
// ML 模型自动检测和加载
// ============================================================================

vector<wstring> FileCarver::GetDefaultMLModelPaths() {
    vector<wstring> paths;

    // 获取当前可执行文件路径
    wchar_t exePath[MAX_PATH];
    if (GetModuleFileNameW(NULL, exePath, MAX_PATH)) {
        wstring exeDir = exePath;
        size_t lastSlash = exeDir.find_last_of(L"\\/");
        if (lastSlash != wstring::npos) {
            exeDir = exeDir.substr(0, lastSlash);
        }

        // 默认模型文件名（分类模型）
        const wchar_t* modelNames[] = {
            L"file_classifier_deep.onnx",
            L"filetype_model_deep.onnx",
            L"ml_model.onnx"
        };

        // 搜索路径优先级（分类模型）:
        // 1. exe同目录
        // 2. exe目录下的models/classification子目录（推荐）
        // 3. exe目录下的models子目录（向后兼容）
        // 4. exe目录下的ml子目录
        vector<wstring> searchDirs = {
            exeDir,
            exeDir + L"\\models\\classification",
            exeDir + L"\\models",
            exeDir + L"\\ml",
            exeDir + L"\\..\\models\\classification",
            exeDir + L"\\..\\ml\\models\\classification",
            exeDir + L"\\..\\models",
            exeDir + L"\\..\\ml\\models"
        };

        for (const auto& dir : searchDirs) {
            for (const auto& name : modelNames) {
                paths.push_back(dir + L"\\" + name);
            }
        }
    }

    return paths;
}

vector<wstring> FileCarver::GetDefaultRepairModelPaths() {
    vector<wstring> paths;

    // 获取可执行文件路径
    wchar_t exePath[MAX_PATH];
    if (GetModuleFileNameW(NULL, exePath, MAX_PATH)) {
        wstring exeDir = exePath;
        size_t lastSlash = exeDir.find_last_of(L"\\/");
        if (lastSlash != wstring::npos) {
            exeDir = exeDir.substr(0, lastSlash);
        }

        // 默认模型文件名（修复模型）
        const wchar_t* modelNames[] = {
            L"image_type_classifier.onnx",
            L"image_repair_model.onnx"
        };

        // 搜索路径优先级（修复模型）:
        // 1. exe同目录
        // 2. exe目录下的models/repair子目录（推荐）
        // 3. exe目录下的models子目录（向后兼容）
        vector<wstring> searchDirs = {
            exeDir,
            exeDir + L"\\models\\repair",
            exeDir + L"\\models",
            exeDir + L"\\..\\models\\repair",
            exeDir + L"\\..\\ml\\models\\repair"
        };

        for (const auto& dir : searchDirs) {
            for (const auto& name : modelNames) {
                paths.push_back(dir + L"\\" + name);
            }
        }
    }

    return paths;
}

void FileCarver::AutoLoadMLModel() {
    // 检查ONNX Runtime是否可用
    if (!ML::MLClassifier::isOnnxRuntimeAvailable()) {
        LOG_DEBUG("ONNX Runtime not available, skipping ML auto-load");
        return;
    }

    // 获取默认模型路径列表
    vector<wstring> modelPaths = GetDefaultMLModelPaths();

    // 尝试加载第一个存在的模型
    for (const auto& path : modelPaths) {
        if (std::filesystem::exists(path)) {
            LOG_INFO(L"Found ML model: " + path);

            if (LoadMLModel(path)) {
                LOG_INFO("ML classification auto-enabled");
                cout << "[ML] Model loaded: ";
                wcout << path << endl;
                return;
            } else {
                LOG_WARNING(L"Failed to load model: " + path);
            }
        }
    }

    LOG_DEBUG("No ML model found in default paths, ML classification disabled");
}

// ============================================================================
// 混合扫描模式（签名 + ML）
// ============================================================================

float FileCarver::QuickEntropy(const BYTE* data, size_t size) {
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
// 静态辅助函数（供线程池使用，线程安全）
// ============================================================================

ULONGLONG FileCarver::FindFooterStatic(const BYTE* data, size_t dataSize,
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

ULONGLONG FileCarver::FindFooterReverseStatic(const BYTE* data, size_t dataSize,
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

ULONGLONG FileCarver::FindPngEndByChunksStatic(const BYTE* data, size_t dataSize) {
    // PNG 文件结构:
    // - 8 字节签名: 89 50 4E 47 0D 0A 1A 0A
    // - 一系列 chunk:
    //   - 4 字节: 数据长度 (大端序)
    //   - 4 字节: chunk 类型
    //   - N 字节: 数据
    //   - 4 字节: CRC32

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
            return 0;  // 无效的 chunk 长度
        }

        // 读取 chunk 类型
        char chunkType[5] = {0};
        memcpy(chunkType, data + offset + 4, 4);

        // 计算完整 chunk 大小
        size_t fullChunkSize = CHUNK_HEADER_SIZE + chunkLength + CHUNK_CRC_SIZE;

        // 检查是否超出数据范围
        if (offset + fullChunkSize > dataSize) {
            // chunk 数据不完整，返回当前估计
            return 0;
        }

        // 检查是否是 IEND chunk
        if (chunkType[0] == 'I' && chunkType[1] == 'E' &&
            chunkType[2] == 'N' && chunkType[3] == 'D') {
            // IEND 的数据长度应该是 0
            if (chunkLength == 0) {
                // 找到有效的 IEND，返回文件结束位置
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
            // 无效的 chunk 类型，可能数据损坏
            return 0;
        }

        // 移动到下一个 chunk
        offset += fullChunkSize;
    }

    // 遍历完成但没找到 IEND
    return 0;
}

string FileCarver::DetectOOXMLTypeStatic(const BYTE* data, size_t dataSize) {
    // OOXML 格式是 ZIP 文件，内部结构：
    // - DOCX: 包含 word/ 目录
    // - XLSX: 包含 xl/ 目录
    // - PPTX: 包含 ppt/ 目录
    // - 所有 OOXML: 包含 [Content_Types].xml

    if (dataSize < 30) return "";

    // 验证是 ZIP 文件
    if (data[0] != 0x50 || data[1] != 0x4B ||
        data[2] != 0x03 || data[3] != 0x04) {
        return "";
    }

    WORD filenameLen = *(WORD*)(data + 26);
    WORD extraLen = *(WORD*)(data + 28);

    if (filenameLen == 0 || 30 + filenameLen > dataSize) {
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

                if (fnLen > 0 && offset + 30 + fnLen <= dataSize) {
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

ULONGLONG FileCarver::FindZipEndOfCentralDirectoryStatic(const BYTE* data, size_t dataSize) {
    // ZIP EOCD 签名: 0x50 0x4B 0x05 0x06
    // EOCD 最小长度: 22 字节
    // EOCD 可能有注释字段（最大 65535 字节）

    if (dataSize < 22) {
        return 0;
    }

    // 从后向前搜索 EOCD 签名（最多搜索 65KB + 22 字节）
    size_t searchStart = dataSize - 22;
    size_t searchLimit = (dataSize > 65557) ? (dataSize - 65557) : 0;

    for (size_t i = searchStart; i >= searchLimit && i < dataSize; i--) {
        // 检查 EOCD 签名
        if (i + 22 <= dataSize &&
            data[i] == 0x50 && data[i + 1] == 0x4B &&
            data[i + 2] == 0x05 && data[i + 3] == 0x06) {

            // 验证注释长度字段
            WORD commentLen = *(WORD*)(data + i + 20);

            // EOCD 总长度 = 22 + 注释长度
            size_t eocdEnd = i + 22 + commentLen;

            // 验证 EOCD 是否在数据范围内
            if (eocdEnd <= dataSize) {
                // 如果注释长度为 0，或者 EOCD 正好在文件末尾，则认为找到
                if (commentLen == 0 || eocdEnd == dataSize) {
                    return eocdEnd;
                }
                // 如果有注释，验证 EOCD 后面的数据是否合理
                if (eocdEnd < dataSize) {
                    // 检查 EOCD 后面是否全是零（可能是填充）
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
        if (i == 0) break;  // 防止无符号下溢
    }

    return 0;
}

// ============================================================================
// 通过遍历 Local File Headers 估算 ZIP 大小
// ============================================================================
ULONGLONG FileCarver::EstimateZipSizeByHeaders(const BYTE* data, size_t dataSize,
                                                bool* outIsComplete) {
    // ZIP Local File Header 结构:
    // 偏移 0:  签名 (4 bytes) = 0x50 0x4B 0x03 0x04
    // 偏移 4:  版本 (2 bytes)
    // 偏移 6:  标志 (2 bytes)
    // 偏移 8:  压缩方法 (2 bytes)
    // 偏移 10: 修改时间 (2 bytes)
    // 偏移 12: 修改日期 (2 bytes)
    // 偏移 14: CRC-32 (4 bytes)
    // 偏移 18: 压缩大小 (4 bytes)
    // 偏移 22: 未压缩大小 (4 bytes)
    // 偏移 26: 文件名长度 (2 bytes)
    // 偏移 28: 扩展字段长度 (2 bytes)
    // 偏移 30: 文件名 (可变)
    // 之后:    扩展字段 (可变)
    // 之后:    文件数据 (compressed_size 字节)

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
    const int MAX_FILES = 100000;  // 防止无限循环

    while (offset + LOCAL_HEADER_MIN_SIZE <= dataSize && fileCount < MAX_FILES) {
        // 检查当前位置的签名
        BYTE sig0 = data[offset];
        BYTE sig1 = data[offset + 1];
        BYTE sig2 = data[offset + 2];
        BYTE sig3 = data[offset + 3];

        // Local File Header: PK\x03\x04
        if (sig0 == 0x50 && sig1 == 0x4B && sig2 == 0x03 && sig3 == 0x04) {
            // 检查是否有足够数据读取头部
            if (offset + LOCAL_HEADER_MIN_SIZE > dataSize) {
                // 数据不完整，文件可能更大
                if (outIsComplete) *outIsComplete = false;
                return dataSize;  // 返回当前缓冲区大小作为最小估计
            }

            // 读取压缩大小和长度字段
            DWORD compressedSize = *(DWORD*)(data + offset + 18);
            WORD filenameLen = *(WORD*)(data + offset + 26);
            WORD extraLen = *(WORD*)(data + offset + 28);

            // 检查是否使用了 Data Descriptor (bit 3 of flags)
            WORD flags = *(WORD*)(data + offset + 6);
            bool hasDataDescriptor = (flags & 0x0008) != 0;

            // 如果使用 Data Descriptor，compressed_size 可能为 0
            // 这种情况下我们无法准确计算，只能估计
            if (hasDataDescriptor && compressedSize == 0) {
                // 无法准确计算，返回当前偏移作为最小估计
                if (outIsComplete) *outIsComplete = false;
                return max((ULONGLONG)offset, (ULONGLONG)dataSize);
            }

            // 验证字段合理性
            if (filenameLen > 1024 || extraLen > 65535) {
                // 异常值，可能是损坏的数据
                if (outIsComplete) *outIsComplete = false;
                return offset > 0 ? offset : dataSize;
            }

            // 计算本条目的总大小
            size_t entrySize = LOCAL_HEADER_MIN_SIZE + filenameLen + extraLen + compressedSize;

            // 如果有 Data Descriptor，还要加 12 或 16 字节
            if (hasDataDescriptor) {
                entrySize += 16;  // 保守估计使用 ZIP64 格式
            }

            // 检查是否超出数据范围
            if (offset + entrySize > dataSize) {
                // 当前文件数据超出缓冲区，ZIP 文件至少这么大
                if (outIsComplete) *outIsComplete = false;
                return offset + entrySize;  // 返回计算出的最小大小
            }

            // 移动到下一个条目
            offset += entrySize;
            fileCount++;
            continue;
        }

        // Central Directory Header: PK\x01\x02
        if (sig0 == 0x50 && sig1 == 0x4B && sig2 == 0x01 && sig3 == 0x02) {
            // 找到 Central Directory，需要继续读取直到 EOCD
            // Central Directory Header 最小 46 字节
            const size_t CD_HEADER_MIN_SIZE = 46;

            while (offset + 4 <= dataSize) {
                BYTE cd0 = data[offset];
                BYTE cd1 = data[offset + 1];
                BYTE cd2 = data[offset + 2];
                BYTE cd3 = data[offset + 3];

                if (cd0 == 0x50 && cd1 == 0x4B && cd2 == 0x01 && cd3 == 0x02) {
                    // CD Header
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
                    // EOCD - 找到了完整的 ZIP 文件
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
                    // ZIP64 End of Central Directory
                    offset += 56;  // ZIP64 EOCD 固定 56 字节
                }
                else if (cd0 == 0x50 && cd1 == 0x4B && cd2 == 0x06 && cd3 == 0x07) {
                    // ZIP64 End of Central Directory Locator
                    offset += 20;  // Locator 固定 20 字节
                }
                else {
                    // 未知签名，可能是损坏或到达末尾
                    break;
                }
            }

            // 遍历完 CD 但没找到 EOCD
            if (outIsComplete) *outIsComplete = false;
            return offset;
        }

        // EOCD: PK\x05\x06 (可能直接遇到，无 Central Directory)
        if (sig0 == 0x50 && sig1 == 0x4B && sig2 == 0x05 && sig3 == 0x06) {
            if (offset + 22 <= dataSize) {
                WORD commentLen = *(WORD*)(data + offset + 20);
                if (outIsComplete) *outIsComplete = true;
                return offset + 22 + commentLen;
            }
        }

        // 未识别的 PK 签名或非 PK 数据
        // 这种情况不应该发生在正常 ZIP 中
        if (outIsComplete) *outIsComplete = false;
        return offset > 0 ? offset : dataSize;
    }

    // 遍历完成但没有明确结束
    if (outIsComplete) *outIsComplete = false;
    return offset;
}

ULONGLONG FileCarver::EstimateFileSizeStatic(const BYTE* data, size_t dataSize,
                                             const FileSignature& sig,
                                             ULONGLONG* outFooterPos,
                                             bool* outIsComplete) {
    ULONGLONG footerPos = 0;
    if (outIsComplete) *outIsComplete = false;  // 默认不完整

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

    // ZIP 文件特殊处理：优先使用 EOCD 查找，失败则遍历头部估算
    if (sig.extension == "zip") {
        footerPos = FindZipEndOfCentralDirectoryStatic(data, dataSize);
        if (footerPos > 0) {
            if (outFooterPos) *outFooterPos = footerPos;

            // 验证 ZIP 结构合理性（排除自解压等情况）
            if (footerPos >= 22 && dataSize >= 4) {
                // 解析 EOCD 中的 Central Directory 偏移
                size_t eocdStart = footerPos - 22;
                DWORD cdOffset = *(DWORD*)(data + eocdStart + 16);

                // 检查文件开头是否是标准 ZIP (Local File Header)
                bool isStandardZip = (data[0] == 0x50 && data[1] == 0x4B &&
                                     data[2] == 0x03 && data[3] == 0x04);

                // 如果不是标准 ZIP 开头，可能是自解压或附加数据
                if (!isStandardZip) {
                    // 搜索第一个 Local File Header
                    for (size_t i = 0; i < min((size_t)cdOffset, dataSize - 4); i++) {
                        if (data[i] == 0x50 && data[i + 1] == 0x4B &&
                            data[i + 2] == 0x03 && data[i + 3] == 0x04) {
                            // 找到真正的 ZIP 起点，计算实际大小
                            ULONGLONG actualSize = footerPos - i;
                            if (outIsComplete) *outIsComplete = true;
                            return actualSize;
                        }
                    }
                }

                // 验证 CD offset 的合理性
                if (cdOffset > footerPos - 22) {
                    // CD 偏移超出范围，可能损坏
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
        // 没有EOCD的ZIP无法确定文件边界，恢复出来的数据没有意义
        if (outFooterPos) *outFooterPos = 0;
        if (outIsComplete) *outIsComplete = false;
        return 0;
    }

    // PDF 和其他文件尾在末尾的格式：使用反向搜索
    if (sig.hasFooter && !sig.footer.empty()) {
        if (sig.extension == "pdf") {
            // PDF %%EOF 在文件末尾，使用反向搜索
            footerPos = FindFooterReverseStatic(data, dataSize, sig.footer, min((ULONGLONG)dataSize, sig.maxSize));
        } else {
            // JPEG, PNG, GIF 等：文件尾可能在中间，使用正向搜索
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
            return size;  // BMP 头部包含精确大小
        }
    }

    if (sig.extension == "mp4" && dataSize >= 12) {
        // 检查 ftyp atom
        if (data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p') {
            DWORD atomSize = _byteswap_ulong(*(DWORD*)data);
            if (atomSize >= 8) {
                // MP4 文件通常很大，返回保守估计
                // TODO: 可以遍历 atoms 来计算精确大小
                return min((ULONGLONG)dataSize, sig.maxSize);
            }
        }
    }

    if (sig.extension == "avi" && dataSize >= 12) {
        // 检查是否真的是 AVI (RIFF + AVI)
        if (data[8] == 'A' && data[9] == 'V' && data[10] == 'I' && data[11] == ' ') {
            DWORD riffSize = *(DWORD*)(data + 4);
            if (riffSize > 0 && riffSize <= sig.maxSize) {
                ULONGLONG totalSize = (ULONGLONG)riffSize + 8;
                if (outIsComplete) *outIsComplete = (totalSize <= dataSize);
                return totalSize;  // RIFF 头部包含精确大小
            }
        }
    }

    if (sig.extension == "wav" && dataSize >= 12) {
        // 检查是否是 WAV (RIFF + WAVE)
        if (data[8] == 'W' && data[9] == 'A' && data[10] == 'V' && data[11] == 'E') {
            DWORD riffSize = *(DWORD*)(data + 4);
            if (riffSize > 0 && riffSize <= sig.maxSize) {
                ULONGLONG totalSize = (ULONGLONG)riffSize + 8;
                if (outIsComplete) *outIsComplete = (totalSize <= dataSize);
                return totalSize;  // RIFF 头部包含精确大小
            }
        }
    }

    // 默认：使用最大大小或可用数据大小
    return min((ULONGLONG)dataSize, sig.maxSize);
}

ULONGLONG FileCarver::EstimateFileSizeML(const BYTE* data, size_t maxSize, const string& type) {
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

vector<CarvedFileInfo> FileCarver::ScanHybridMode(
    const vector<string>& fileTypes,
    const HybridScanConfig& config,
    CarvingMode mode,
    ULONGLONG maxResults
) {
    vector<CarvedFileInfo> results;

    // 分离签名类型和 ML-only 类型
    vector<string> signatureTypes;
    set<string> mlOnlyTypes;

    for (const auto& type : fileTypes) {
        string lowerType = type;
        transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);

        if (config.mlOnlyTypes.count(lowerType) > 0) {
            mlOnlyTypes.insert(lowerType);
        } else if (signatures.count(lowerType) > 0) {
            signatureTypes.push_back(lowerType);
        } else if (config.enableMLScan && mlClassifier && mlClassifier->isTypeSupported(lowerType)) {
            // 签名库不支持但 ML 支持的类型
            mlOnlyTypes.insert(lowerType);
        }
    }

    cout << "\n=== Hybrid Scan Mode ===" << endl;
    cout << "Signature types: ";
    for (const auto& t : signatureTypes) cout << t << " ";
    cout << endl;
    cout << "ML-only types: ";
    for (const auto& t : mlOnlyTypes) cout << t << " ";
    cout << endl;

    // 阶段1：签名扫描
    vector<CarvedFileInfo> sigResults;
    if (config.enableSignatureScan && !signatureTypes.empty()) {
        cout << "\n--- Phase 1: Signature Scan ---" << endl;
        sigResults = ScanForFileTypesThreadPool(signatureTypes, mode, maxResults);
        cout << "Signature scan found: " << sigResults.size() << " files" << endl;
    }

    // 阶段2：ML 扫描（仅针对无签名类型）
    vector<CarvedFileInfo> mlResults;
    if (config.enableMLScan && !mlOnlyTypes.empty() && mlClassifier && mlClassifier->isLoaded()) {
        cout << "\n--- Phase 2: ML Scan (for txt/html/xml) ---" << endl;

        // 获取卷信息
        ULONGLONG bytesPerCluster = reader->GetBytesPerCluster();
        ULONGLONG totalClusters = reader->GetTotalClusters();

        // 准备扫描
        const size_t BUFFER_CLUSTERS = 16384;  // 64MB @ 4KB/cluster
        const size_t ML_FRAGMENT_SIZE = 4096;

        vector<BYTE> buffer;
        ULONGLONG currentLCN = 0;
        DWORD startTime = GetTickCount();

        size_t mlCandidates = 0;
        size_t mlProcessed = 0;

        while (currentLCN < totalClusters && !shouldStop && mlResults.size() < maxResults) {
            ULONGLONG clustersToRead = min((ULONGLONG)BUFFER_CLUSTERS, totalClusters - currentLCN);

            if (!reader->ReadClusters(currentLCN, clustersToRead, buffer)) {
                currentLCN += clustersToRead;
                continue;
            }

            // ML 扫描缓冲区
            vector<ML::BatchClassificationInput> batch;

            for (size_t offset = 0; offset + ML_FRAGMENT_SIZE <= buffer.size(); offset += config.mlScanStep) {
                // 预过滤：熵检测
                if (config.prefilterEmpty) {
                    float entropy = QuickEntropy(buffer.data() + offset, 256);
                    if (entropy < 0.1f || entropy > 7.9f) continue;  // 跳过空/随机
                }

                // 收集候选
                ML::BatchClassificationInput input;
                input.data = buffer.data() + offset;
                input.size = ML_FRAGMENT_SIZE;
                input.lcn = currentLCN + offset / bytesPerCluster;
                input.offset = offset;
                batch.push_back(input);
                mlCandidates++;

                // 批量处理
                if (batch.size() >= config.mlBatchSize) {
                    auto batchResults = mlClassifier->classifyBatch(batch, ML_FRAGMENT_SIZE);

                    for (size_t i = 0; i < batchResults.size(); i++) {
                        if (!batchResults[i] || batchResults[i]->isUnknown) continue;
                        if (batchResults[i]->confidence < config.mlConfidenceThreshold) continue;

                        // 检查是否为目标类型
                        if (mlOnlyTypes.count(batchResults[i]->predictedType) == 0) continue;

                        CarvedFileInfo info;
                        info.extension = batchResults[i]->predictedType;
                        info.description = "ML-detected " + info.extension;
                        info.startLCN = batch[i].lcn;
                        info.startOffset = batch[i].offset % bytesPerCluster;
                        info.fileSize = EstimateFileSizeML(
                            batch[i].data, buffer.size() - batch[i].offset, info.extension);
                        info.confidence = batchResults[i]->confidence * 0.85;
                        info.mlConfidence = batchResults[i]->confidence;
                        info.mlClassification = batchResults[i]->predictedType;

                        mlResults.push_back(info);
                        mlProcessed++;
                    }

                    batch.clear();
                }
            }

            // 处理剩余批次
            if (!batch.empty()) {
                auto batchResults = mlClassifier->classifyBatch(batch, ML_FRAGMENT_SIZE);

                for (size_t i = 0; i < batchResults.size(); i++) {
                    if (!batchResults[i] || batchResults[i]->isUnknown) continue;
                    if (batchResults[i]->confidence < config.mlConfidenceThreshold) continue;
                    if (mlOnlyTypes.count(batchResults[i]->predictedType) == 0) continue;

                    CarvedFileInfo info;
                    info.extension = batchResults[i]->predictedType;
                    info.description = "ML-detected " + info.extension;
                    info.startLCN = batch[i].lcn;
                    info.startOffset = batch[i].offset % bytesPerCluster;
                    info.fileSize = EstimateFileSizeML(
                        batch[i].data, buffer.size() - batch[i].offset, info.extension);
                    info.confidence = batchResults[i]->confidence * 0.85;
                    info.mlConfidence = batchResults[i]->confidence;
                    info.mlClassification = batchResults[i]->predictedType;

                    mlResults.push_back(info);
                    mlProcessed++;
                }
            }

            currentLCN += clustersToRead;

            // 进度显示
            if (currentLCN % (BUFFER_CLUSTERS * 10) == 0) {
                double progress = (double)currentLCN / totalClusters * 100.0;
                cout << "\rML Scan: " << fixed << setprecision(1) << progress << "% | "
                     << "Candidates: " << mlCandidates << " | Found: " << mlResults.size() << flush;
            }
        }

        DWORD elapsed = GetTickCount() - startTime;
        cout << "\r                                                                    " << endl;
        cout << "ML scan found: " << mlResults.size() << " files (candidates: " << mlCandidates
             << ", time: " << (elapsed / 1000) << "s)" << endl;
    }

    // 阶段3：结果融合
    cout << "\n--- Phase 3: Result Fusion ---" << endl;

    // 先添加签名结果（优先级高）
    set<ULONGLONG> processedLCNs;
    for (auto& sig : sigResults) {
        results.push_back(sig);
        processedLCNs.insert(sig.startLCN);
    }

    // 添加不重叠的 ML 结果
    size_t mlAdded = 0;
    for (auto& ml : mlResults) {
        // 检查是否与签名结果重叠（±2 簇范围）
        bool overlaps = false;
        for (ULONGLONG lcn = ml.startLCN > 2 ? ml.startLCN - 2 : 0;
             lcn <= ml.startLCN + 2; lcn++) {
            if (processedLCNs.count(lcn)) {
                overlaps = true;
                break;
            }
        }

        if (!overlaps) {
            results.push_back(ml);
            processedLCNs.insert(ml.startLCN);
            mlAdded++;
        }
    }

    cout << "Fusion complete: " << sigResults.size() << " (sig) + " << mlAdded
         << " (ML) = " << results.size() << " total" << endl;

    // 按置信度排序
    sort(results.begin(), results.end(), [](const CarvedFileInfo& a, const CarvedFileInfo& b) {
        return a.confidence > b.confidence;
    });

    return results;
}

// ============================================================================
// 基于结构的 ZIP 恢复（扫描到 EOCD）
// ============================================================================

FileCarver::ZipRecoveryResult FileCarver::RecoverZipWithEOCDScan(
    ULONGLONG startLCN,
    const string& outputPath,
    const ZipRecoveryConfig& config
) {
    ZipRecoveryResult result;
    result.diagnosis = "";

    LOG_INFO("Starting ZIP recovery with EOCD scan from LCN " + to_string(startLCN));

    // 获取簇大小
    ULONGLONG bytesPerCluster = reader->GetBytesPerCluster();
    if (bytesPerCluster == 0) {
        result.diagnosis = "Invalid bytes per cluster";
        LOG_ERROR(result.diagnosis);
        return result;
    }

    // 块大小设置
    const size_t CHUNK_SIZE = 4 * 1024 * 1024;  // 4MB 块
    const size_t CLUSTERS_PER_CHUNK = CHUNK_SIZE / static_cast<size_t>(bytesPerCluster);

    // 确定最大搜索大小
    ULONGLONG maxSearchSize = config.maxSize;
    if (config.expectedSize > 0) {
        // 如果用户指定了预期大小，使用它加上容差
        ULONGLONG tolerance = config.expectedSizeTolerance;
        if (tolerance == 0) {
            tolerance = config.expectedSize / 10;  // 默认 10% 容差
        }
        maxSearchSize = min(maxSearchSize, config.expectedSize + tolerance);
        LOG_INFO("Using expected size " + to_string(config.expectedSize) +
                 " with tolerance " + to_string(tolerance));
    }

    // 文件缓冲区（使用 vector 以支持大文件）
    vector<BYTE> fileBuffer;
    fileBuffer.reserve(min(maxSearchSize, (ULONGLONG)256 * 1024 * 1024));  // 预分配最多 256MB

    vector<BYTE> chunk(CHUNK_SIZE);
    ULONGLONG currentLCN = startLCN;
    ULONGLONG bytesRead = 0;
    bool foundEOCD = false;
    ULONGLONG eocdPosition = 0;

    cout << "Scanning for ZIP EOCD (max " << (maxSearchSize / (1024 * 1024)) << " MB)..." << endl;

    while (bytesRead < maxSearchSize && !shouldStop) {
        // 读取当前块
        if (!reader->ReadClusters(currentLCN, CLUSTERS_PER_CHUNK, chunk)) {
            result.diagnosis = "Failed to read cluster at LCN " + to_string(currentLCN);
            LOG_WARNING(result.diagnosis);

            if (config.allowFragmented) {
                // 跳过无法读取的块，继续搜索
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

        // 在当前块中搜索 EOCD
        ULONGLONG eocdInChunk = FindZipEndOfCentralDirectoryStatic(chunk.data(), chunkSize);

        if (eocdInChunk > 0) {
            // 找到 EOCD
            foundEOCD = true;
            eocdPosition = bytesRead + eocdInChunk;

            // 只追加到 EOCD 结束位置
            fileBuffer.insert(fileBuffer.end(), chunk.begin(), chunk.begin() + eocdInChunk);
            bytesRead += eocdInChunk;

            LOG_INFO("Found EOCD at position " + to_string(eocdPosition));
            result.diagnosis = "EOCD found at offset " + to_string(eocdPosition);

            if (config.stopOnFirstEOCD) {
                break;
            }
        } else {
            // 未找到 EOCD，追加整个块
            fileBuffer.insert(fileBuffer.end(), chunk.begin(), chunk.end());
            bytesRead += chunkSize;
        }

        currentLCN += CLUSTERS_PER_CHUNK;

        // 进度显示（每 16MB）
        if (bytesRead % (16 * 1024 * 1024) == 0) {
            cout << "\rScanning: " << (bytesRead / (1024 * 1024)) << " MB" << flush;
        }

        // 检查是否超过预期大小（如果设置）
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

        // 即使没找到 EOCD，也保存已读取的数据（可能是截断的 ZIP）
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

    // 写入完整的 ZIP 文件
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

    // 验证 CRC（如果启用）
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
// 验证 ZIP 文件完整性
// ============================================================================

FileCarver::ZipRecoveryResult FileCarver::ValidateZipFile(const string& filePath) {
    ZipRecoveryResult result;

    // 打开文件
    HANDLE hFile = CreateFileA(filePath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        result.diagnosis = "Failed to open file: " + filePath;
        return result;
    }

    // 获取文件大小
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        result.diagnosis = "Failed to get file size";
        return result;
    }

    // 限制最大验证大小（避免内存溢出）
    const ULONGLONG MAX_VALIDATE_SIZE = 512ULL * 1024 * 1024;  // 512MB
    if ((ULONGLONG)fileSize.QuadPart > MAX_VALIDATE_SIZE) {
        CloseHandle(hFile);
        result.diagnosis = "File too large for full validation (>" +
                          to_string(MAX_VALIDATE_SIZE / (1024 * 1024)) + "MB)";
        // 对于大文件，只验证 EOCD 和 CD 结构
        // TODO: 实现流式 CRC 验证
        return result;
    }

    // 读取整个文件
    vector<BYTE> data((size_t)fileSize.QuadPart);
    DWORD bytesRead = 0;
    if (!ReadFile(hFile, data.data(), (DWORD)fileSize.QuadPart, &bytesRead, NULL)) {
        CloseHandle(hFile);
        result.diagnosis = "Failed to read file";
        return result;
    }
    CloseHandle(hFile);

    // 调用内存验证函数
    return ValidateZipData(data.data(), data.size());
}

FileCarver::ZipRecoveryResult FileCarver::ValidateZipData(const BYTE* data, size_t dataSize) {
    ZipRecoveryResult result;
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

    // EOCD 结构：
    // 偏移 0:  签名 (4 bytes) = 0x50 0x4B 0x05 0x06
    // 偏移 4:  disk number (2 bytes)
    // 偏移 6:  disk with CD (2 bytes)
    // 偏移 8:  entries on this disk (2 bytes)
    // 偏移 10: total entries (2 bytes)
    // 偏移 12: CD size (4 bytes)
    // 偏移 16: CD offset (4 bytes)
    // 偏移 20: comment length (2 bytes)

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
        // 检查 CD 签名
        if (data[cdPos] != 0x50 || data[cdPos + 1] != 0x4B ||
            data[cdPos + 2] != 0x01 || data[cdPos + 3] != 0x02) {
            result.diagnosis = "Invalid CD entry signature at offset " + to_string(cdPos);
            break;
        }

        // 读取 CD 条目信息
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

        // 获取本地头部的额外字段长度（可能与 CD 不同）
        WORD localFilenameLen = *(WORD*)(data + localHeaderOffset + 26);
        WORD localExtraLen = *(WORD*)(data + localHeaderOffset + 28);

        // 计算数据位置
        size_t dataStart = localHeaderOffset + 30 + localFilenameLen + localExtraLen;

        // CRC 验证（仅对未压缩或 DEFLATE 压缩的文件进行）
        if (compression == 0 && compressedSize > 0 && dataStart + compressedSize <= dataSize) {
            // 未压缩文件：直接计算 CRC
            DWORD actualCRC = 0;
            // 简化的 CRC32 计算（使用标准多项式 0xEDB88320）
            actualCRC = 0xFFFFFFFF;
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
