#include "TimestampExtractor.h"
#include <cstring>
#include <algorithm>
#include <ctime>

// ============================================================================
// 字节序转换函数
// ============================================================================
WORD TimestampExtractor::ReadWord(const BYTE* data, bool bigEndian) {
    if (bigEndian) {
        return (data[0] << 8) | data[1];
    }
    return data[0] | (data[1] << 8);
}

DWORD TimestampExtractor::ReadDWord(const BYTE* data, bool bigEndian) {
    if (bigEndian) {
        return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
    }
    return data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
}

ULONGLONG TimestampExtractor::ReadQWord(const BYTE* data, bool bigEndian) {
    if (bigEndian) {
        return ((ULONGLONG)ReadDWord(data, true) << 32) | ReadDWord(data + 4, true);
    }
    return ReadDWord(data, false) | ((ULONGLONG)ReadDWord(data + 4, false) << 32);
}

// ============================================================================
// 时间格式转换函数
// ============================================================================

// EXIF 日期时间格式: "YYYY:MM:DD HH:MM:SS"
bool TimestampExtractor::ParseEXIFDateTime(const char* dateStr, FILETIME& ft) {
    if (!dateStr || strlen(dateStr) < 19) return false;

    SYSTEMTIME st = {0};
    if (sscanf_s(dateStr, "%hd:%hd:%hd %hd:%hd:%hd",
                 &st.wYear, &st.wMonth, &st.wDay,
                 &st.wHour, &st.wMinute, &st.wSecond) != 6) {
        return false;
    }

    // 验证日期合理性
    if (st.wYear < 1970 || st.wYear > 2100 ||
        st.wMonth < 1 || st.wMonth > 12 ||
        st.wDay < 1 || st.wDay > 31) {
        return false;
    }

    return SystemTimeToFileTime(&st, &ft) != FALSE;
}

// DOS 时间转 FILETIME（ZIP 格式使用）
bool TimestampExtractor::DOSTimeToFileTime(WORD dosDate, WORD dosTime, FILETIME& ft) {
    // DOS 日期: 位 0-4=日, 5-8=月, 9-15=年-1980
    // DOS 时间: 位 0-4=秒/2, 5-10=分钟, 11-15=小时

    SYSTEMTIME st = {0};
    st.wYear = ((dosDate >> 9) & 0x7F) + 1980;
    st.wMonth = (dosDate >> 5) & 0x0F;
    st.wDay = dosDate & 0x1F;
    st.wHour = (dosTime >> 11) & 0x1F;
    st.wMinute = (dosTime >> 5) & 0x3F;
    st.wSecond = (dosTime & 0x1F) * 2;

    // 验证
    if (st.wMonth < 1 || st.wMonth > 12 || st.wDay < 1 || st.wDay > 31) {
        return false;
    }

    return SystemTimeToFileTime(&st, &ft) != FALSE;
}

// Unix 时间戳转 FILETIME
bool TimestampExtractor::UnixTimeToFileTime(ULONGLONG unixTime, FILETIME& ft) {
    // Unix 时间戳: 从 1970-01-01 00:00:00 UTC 的秒数
    // FILETIME: 从 1601-01-01 00:00:00 UTC 的 100 纳秒数
    // 差值: 11644473600 秒

    ULONGLONG fileTime = (unixTime + 11644473600ULL) * 10000000ULL;
    ft.dwLowDateTime = (DWORD)fileTime;
    ft.dwHighDateTime = (DWORD)(fileTime >> 32);
    return true;
}

// MP4 时间（从 1904-01-01 的秒数）转 FILETIME
bool TimestampExtractor::MP4TimeToFileTime(ULONGLONG mp4Time, FILETIME& ft) {
    // MP4 时间基准: 1904-01-01 00:00:00 UTC
    // Unix 基准: 1970-01-01 00:00:00 UTC
    // 差值: 2082844800 秒

    if (mp4Time < 2082844800ULL) {
        // 时间值太小，可能无效
        return false;
    }

    ULONGLONG unixTime = mp4Time - 2082844800ULL;
    return UnixTimeToFileTime(unixTime, ft);
}

// ============================================================================
// 主提取函数
// ============================================================================
ExtractedTimestamp TimestampExtractor::Extract(const BYTE* data, size_t dataSize,
                                                const string& extension) {
    if (!data || dataSize < 16) {
        return ExtractedTimestamp();
    }

    string ext = extension;
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == "jpg" || ext == "jpeg") {
        return ExtractFromJPEG(data, dataSize);
    }
    else if (ext == "png") {
        return ExtractFromPNG(data, dataSize);
    }
    else if (ext == "pdf") {
        return ExtractFromPDF(data, dataSize);
    }
    else if (ext == "zip" || ext == "docx" || ext == "xlsx" || ext == "pptx") {
        return ExtractFromZIP(data, dataSize);
    }
    else if (ext == "mp4" || ext == "mov" || ext == "m4a" || ext == "m4v") {
        return ExtractFromMP4(data, dataSize);
    }

    return ExtractedTimestamp();
}

// ============================================================================
// JPEG/EXIF 提取
// ============================================================================
ExtractedTimestamp TimestampExtractor::ExtractFromJPEG(const BYTE* data, size_t dataSize) {
    ExtractedTimestamp result;

    // 验证JPEG头
    if (dataSize < 12 || data[0] != 0xFF || data[1] != 0xD8) {
        return result;
    }

    // 搜索APP1 (EXIF) 段
    size_t offset = 2;
    while (offset + 4 < dataSize) {
        if (data[offset] != 0xFF) {
            offset++;
            continue;
        }

        BYTE marker = data[offset + 1];

        // APP1 标记 (EXIF)
        if (marker == 0xE1) {
            WORD segmentLength = ReadWord(data + offset + 2, true);
            if (offset + 4 + segmentLength > dataSize) break;

            // 检查 "Exif\0\0" 标识
            if (offset + 10 < dataSize &&
                memcmp(data + offset + 4, "Exif\0\0", 6) == 0) {
                ParseEXIF(data + offset + 10, segmentLength - 8, result);
            }
            break;
        }
        // SOS (Start of Scan) - 后面是图像数据，停止搜索
        else if (marker == 0xDA) {
            break;
        }
        // 其他段，跳过
        else if (marker >= 0xE0 && marker <= 0xEF) {
            WORD segmentLength = ReadWord(data + offset + 2, true);
            offset += 2 + segmentLength;
        }
        else {
            offset++;
        }
    }

    return result;
}

// EXIF IFD 解析函数
bool TimestampExtractor::ParseEXIF(const BYTE* data, size_t dataSize, ExtractedTimestamp& result) {
    if (dataSize < 8) return false;

    // TIFF 头部
    bool bigEndian;
    if (data[0] == 'M' && data[1] == 'M') {
        bigEndian = true;
    } else if (data[0] == 'I' && data[1] == 'I') {
        bigEndian = false;
    } else {
        return false;
    }

    // 验证 TIFF 魔数 (0x002A)
    WORD magic = ReadWord(data + 2, bigEndian);
    if (magic != 0x002A) return false;

    // IFD0 偏移
    DWORD ifd0Offset = ReadDWord(data + 4, bigEndian);
    if (ifd0Offset >= dataSize) return false;

    return ParseIFD(data, dataSize, ifd0Offset, bigEndian, result);
}

bool TimestampExtractor::ParseIFD(const BYTE* tiffStart, size_t tiffSize, DWORD ifdOffset,
                                   bool bigEndian, ExtractedTimestamp& result) {
    if (ifdOffset + 2 > tiffSize) return false;

    WORD entryCount = ReadWord(tiffStart + ifdOffset, bigEndian);
    if (entryCount > 200) return false;  // 合理性检查

    const BYTE* entry = tiffStart + ifdOffset + 2;
    DWORD exifIFDOffset = 0;

    for (WORD i = 0; i < entryCount && (entry - tiffStart + 12) <= tiffSize; i++, entry += 12) {
        WORD tag = ReadWord(entry, bigEndian);
        WORD type = ReadWord(entry + 2, bigEndian);
        DWORD count = ReadDWord(entry + 4, bigEndian);
        DWORD valueOffset = ReadDWord(entry + 8, bigEndian);

        // 标签 0x8769: ExifIFD 偏移
        if (tag == 0x8769) {
            exifIFDOffset = valueOffset;
        }
        // 标签 0x010F: 相机制造商
        else if (tag == 0x010F && type == 2 && count < 64) {
            const char* str = (valueOffset + count <= tiffSize) ?
                              (const char*)(tiffStart + valueOffset) : nullptr;
            if (str && count > 4) {
                result.additionalInfo = "Make: " + string(str, strnlen(str, count - 1));
            }
        }
        // 标签 0x0110: 相机型号
        else if (tag == 0x0110 && type == 2 && count < 64) {
            const char* str = (valueOffset + count <= tiffSize) ?
                              (const char*)(tiffStart + valueOffset) : nullptr;
            if (str && count > 4) {
                if (!result.additionalInfo.empty()) result.additionalInfo += ", ";
                result.additionalInfo += "Model: " + string(str, strnlen(str, count - 1));
            }
        }
        // 标签 0x0132: 修改日期时间
        else if (tag == 0x0132 && type == 2 && count == 20) {
            const char* dateStr = (valueOffset + 20 <= tiffSize) ?
                                  (const char*)(tiffStart + valueOffset) : nullptr;
            if (dateStr && ParseEXIFDateTime(dateStr, result.modificationTime)) {
                result.hasModification = true;
            }
        }
    }

    // 解析 EXIF IFD（包含更详细的时间信息）
    if (exifIFDOffset > 0 && exifIFDOffset + 2 < tiffSize) {
        WORD exifEntryCount = ReadWord(tiffStart + exifIFDOffset, bigEndian);
        if (exifEntryCount <= 200) {
            const BYTE* exifEntry = tiffStart + exifIFDOffset + 2;

            for (WORD i = 0; i < exifEntryCount && (exifEntry - tiffStart + 12) <= tiffSize; i++, exifEntry += 12) {
                WORD tag = ReadWord(exifEntry, bigEndian);
                WORD type = ReadWord(exifEntry + 2, bigEndian);
                DWORD count = ReadDWord(exifEntry + 4, bigEndian);
                DWORD valueOffset = ReadDWord(exifEntry + 8, bigEndian);

                // 标签 0x9003: 原始拍摄日期时间 (DateTimeOriginal)
                if (tag == 0x9003 && type == 2 && count == 20) {
                    const char* dateStr = (valueOffset + 20 <= tiffSize) ?
                                          (const char*)(tiffStart + valueOffset) : nullptr;
                    if (dateStr && ParseEXIFDateTime(dateStr, result.creationTime)) {
                        result.hasCreation = true;
                    }
                }
                // 标签 0x9004: 数字化日期时间 (DateTimeDigitized)
                else if (tag == 0x9004 && type == 2 && count == 20 && !result.hasCreation) {
                    const char* dateStr = (valueOffset + 20 <= tiffSize) ?
                                          (const char*)(tiffStart + valueOffset) : nullptr;
                    if (dateStr && ParseEXIFDateTime(dateStr, result.creationTime)) {
                        result.hasCreation = true;
                    }
                }
            }
        }
    }

    return result.hasAnyTimestamp();
}

// ============================================================================
// PNG tIME 块提取
// ============================================================================
ExtractedTimestamp TimestampExtractor::ExtractFromPNG(const BYTE* data, size_t dataSize) {
    ExtractedTimestamp result;

    // 验证 PNG 签名
    static const BYTE pngSig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (dataSize < 8 || memcmp(data, pngSig, 8) != 0) {
        return result;
    }

    // 遍历数据块
    size_t offset = 8;
    while (offset + 12 < dataSize) {
        DWORD chunkLength = ReadDWord(data + offset, true);
        const char* chunkType = (const char*)(data + offset + 4);

        if (offset + 12 + chunkLength > dataSize) break;

        // tIME 块
        if (memcmp(chunkType, "tIME", 4) == 0 && chunkLength == 7) {
            const BYTE* timeData = data + offset + 8;
            SYSTEMTIME st = {0};
            st.wYear = ReadWord(timeData, true);
            st.wMonth = timeData[2];
            st.wDay = timeData[3];
            st.wHour = timeData[4];
            st.wMinute = timeData[5];
            st.wSecond = timeData[6];

            if (st.wYear >= 1970 && st.wYear <= 2100 &&
                st.wMonth >= 1 && st.wMonth <= 12 &&
                st.wDay >= 1 && st.wDay <= 31) {
                if (SystemTimeToFileTime(&st, &result.modificationTime)) {
                    result.hasModification = true;
                }
            }
        }
        // tEXt 块 - 可能包含创建日期
        else if (memcmp(chunkType, "tEXt", 4) == 0) {
            string keyword((const char*)(data + offset + 8));
            if (keyword == "Creation Time" && chunkLength > keyword.length() + 1) {
                // 尝试解析创建时间字符串
                // 格式可能是 RFC 1123 或其他格式（暂未实现）
            }
        }
        // IEND - 文件结束标记
        else if (memcmp(chunkType, "IEND", 4) == 0) {
            break;
        }

        offset += 12 + chunkLength;  // 4(长度) + 4(类型) + 数据 + 4(CRC校验)
    }

    return result;
}

// ============================================================================
// PDF 元数据提取
// ============================================================================
ExtractedTimestamp TimestampExtractor::ExtractFromPDF(const BYTE* data, size_t dataSize) {
    ExtractedTimestamp result;

    // 验证 PDF 头
    if (dataSize < 8 || memcmp(data, "%PDF-", 5) != 0) {
        return result;
    }

    // 搜索 /CreationDate 和 /ModDate
    // PDF 日期格式: D:YYYYMMDDHHmmSS+HH'mm' 或类似格式
    string content((const char*)data, min(dataSize, (size_t)65536));  // 只搜索前 64KB

    auto parsePDFDate = [](const string& dateStr) -> FILETIME {
        FILETIME ft = {0};
        if (dateStr.length() < 16 || dateStr[0] != 'D' || dateStr[1] != ':') {
            return ft;
        }

        SYSTEMTIME st = {0};
        if (sscanf_s(dateStr.c_str() + 2, "%4hd%2hd%2hd%2hd%2hd%2hd",
                     &st.wYear, &st.wMonth, &st.wDay,
                     &st.wHour, &st.wMinute, &st.wSecond) >= 3) {
            if (st.wYear >= 1970 && st.wYear <= 2100) {
                SystemTimeToFileTime(&st, &ft);
            }
        }
        return ft;
    };

    // 搜索 /CreationDate
    size_t pos = content.find("/CreationDate");
    if (pos != string::npos) {
        size_t start = content.find('(', pos);
        size_t end = content.find(')', start);
        if (start != string::npos && end != string::npos && end > start) {
            string dateStr = content.substr(start + 1, end - start - 1);
            result.creationTime = parsePDFDate(dateStr);
            if (result.creationTime.dwHighDateTime != 0) {
                result.hasCreation = true;
            }
        }
    }

    // 搜索 /ModDate
    pos = content.find("/ModDate");
    if (pos != string::npos) {
        size_t start = content.find('(', pos);
        size_t end = content.find(')', start);
        if (start != string::npos && end != string::npos && end > start) {
            string dateStr = content.substr(start + 1, end - start - 1);
            result.modificationTime = parsePDFDate(dateStr);
            if (result.modificationTime.dwHighDateTime != 0) {
                result.hasModification = true;
            }
        }
    }

    // 搜索 /Producer 或 /Creator
    pos = content.find("/Producer");
    if (pos != string::npos) {
        size_t start = content.find('(', pos);
        size_t end = content.find(')', start);
        if (start != string::npos && end != string::npos && end > start && end - start < 100) {
            result.additionalInfo = "Producer: " + content.substr(start + 1, end - start - 1);
        }
    }

    return result;
}

// ============================================================================
// ZIP 时间戳提取
// ============================================================================
ExtractedTimestamp TimestampExtractor::ExtractFromZIP(const BYTE* data, size_t dataSize) {
    ExtractedTimestamp result;

    // 验证 ZIP 签名
    if (dataSize < 30 || data[0] != 0x50 || data[1] != 0x4B ||
        data[2] != 0x03 || data[3] != 0x04) {
        return result;
    }

    // 本地文件头格式:
    // 偏移 10-11: 最后修改时间 (DOS 格式)
    // 偏移 12-13: 最后修改日期 (DOS 格式)

    WORD dosTime = ReadWord(data + 10, false);
    WORD dosDate = ReadWord(data + 12, false);

    if (DOSTimeToFileTime(dosDate, dosTime, result.modificationTime)) {
        result.hasModification = true;
    }

    // 获取文件名（用于额外信息）
    WORD fileNameLength = ReadWord(data + 26, false);
    if (fileNameLength > 0 && fileNameLength < 256 && 30 + fileNameLength <= dataSize) {
        string fileName((const char*)(data + 30), fileNameLength);
        result.additionalInfo = "First entry: " + fileName;
    }

    return result;
}

// ============================================================================
// MP4 时间戳提取
// ============================================================================
ExtractedTimestamp TimestampExtractor::ExtractFromMP4(const BYTE* data, size_t dataSize) {
    ExtractedTimestamp result;

    if (dataSize < 16) return result;

    // 搜索 moov 原子，然后找 mvhd
    size_t offset = 0;

    while (offset + 8 < dataSize) {
        DWORD atomSize = ReadDWord(data + offset, true);
        const char* atomType = (const char*)(data + offset + 4);

        if (atomSize < 8 || offset + atomSize > dataSize) break;

        // ftyp 原子 - 验证是 MP4
        if (memcmp(atomType, "ftyp", 4) == 0) {
            if (atomSize >= 12) {
                string brand((const char*)(data + offset + 8), 4);
                result.additionalInfo = "Brand: " + brand;
            }
        }
        // moov 原子 - 包含 mvhd
        else if (memcmp(atomType, "moov", 4) == 0) {
            // 在 moov 原子内搜索 mvhd
            size_t moovOffset = offset + 8;
            size_t moovEnd = offset + atomSize;

            while (moovOffset + 8 < moovEnd) {
                DWORD subAtomSize = ReadDWord(data + moovOffset, true);
                const char* subAtomType = (const char*)(data + moovOffset + 4);

                if (subAtomSize < 8 || moovOffset + subAtomSize > moovEnd) break;

                if (memcmp(subAtomType, "mvhd", 4) == 0) {
                    // mvhd 格式:
                    // 版本 0: 创建/修改时间各 4 字节
                    // 版本 1: 创建/修改时间各 8 字节

                    if (subAtomSize >= 20) {
                        BYTE version = data[moovOffset + 8];
                        ULONGLONG creationTime, modTime;

                        if (version == 0 && subAtomSize >= 20) {
                            creationTime = ReadDWord(data + moovOffset + 12, true);
                            modTime = ReadDWord(data + moovOffset + 16, true);
                        }
                        else if (version == 1 && subAtomSize >= 28) {
                            creationTime = ReadQWord(data + moovOffset + 12, true);
                            modTime = ReadQWord(data + moovOffset + 20, true);
                        }
                        else {
                            break;
                        }

                        if (creationTime > 0 && MP4TimeToFileTime(creationTime, result.creationTime)) {
                            result.hasCreation = true;
                        }
                        if (modTime > 0 && MP4TimeToFileTime(modTime, result.modificationTime)) {
                            result.hasModification = true;
                        }
                    }
                    break;
                }

                moovOffset += subAtomSize;
            }
            break;  // 找到 moov 原子后停止
        }

        offset += atomSize;
    }

    return result;
}
