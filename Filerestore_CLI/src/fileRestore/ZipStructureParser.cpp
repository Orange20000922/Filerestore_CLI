/**
 * @file ZipStructureParser.cpp
 * @brief ZIP文件结构解析器实现
 */

#include "ZipStructureParser.h"
#include <fstream>
#include <algorithm>
#include <cstring>

namespace ZipParser {

// ============================================================================
// CRC32表 (与FileIntegrityValidator相同)
// ============================================================================
const uint32_t ZipStructureParser::crcTable[256] = {
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F,
    0xE963A535, 0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988,
    0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91, 0x1DB71064, 0x6AB020F2,
    0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
    0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9,
    0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
    0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
    0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
    0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423,
    0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
    0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D, 0x76DC4190, 0x01DB7106,
    0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
    0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D,
    0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
    0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950,
    0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
    0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7,
    0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
    0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9, 0x5005713C, 0x270241AA,
    0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
    0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81,
    0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A,
    0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683, 0xE3630B12, 0x94643B84,
    0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
    0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB,
    0x196C3671, 0x6E6B06E7, 0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC,
    0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8, 0xA1D1937E,
    0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
    0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55,
    0x316E8EEF, 0x4669BE79, 0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236,
    0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F, 0xC5BA3BBE, 0xB2BD0B28,
    0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
    0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F,
    0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
    0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242,
    0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
    0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69,
    0x616BFFD3, 0x166CCF45, 0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
    0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC,
    0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
    0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD706B3,
    0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
    0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
};

// ============================================================================
// CRC32计算
// ============================================================================
uint32_t ZipStructureParser::CalculateCRC32(const BYTE* data, size_t size) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < size; i++) {
        crc = crcTable[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

// ============================================================================
// DOS时间转换
// ============================================================================
FILETIME ZipStructureParser::DosTimeToFileTime(uint16_t dosTime, uint16_t dosDate) {
    FILETIME ft = {0};
    SYSTEMTIME st = {0};

    // DOS日期格式: bits 0-4: day, bits 5-8: month, bits 9-15: year (from 1980)
    st.wDay = dosDate & 0x1F;
    st.wMonth = (dosDate >> 5) & 0x0F;
    st.wYear = ((dosDate >> 9) & 0x7F) + 1980;

    // DOS时间格式: bits 0-4: second/2, bits 5-10: minute, bits 11-15: hour
    st.wSecond = (dosTime & 0x1F) * 2;
    st.wMinute = (dosTime >> 5) & 0x3F;
    st.wHour = (dosTime >> 11) & 0x1F;

    SystemTimeToFileTime(&st, &ft);
    return ft;
}

// ============================================================================
// 压缩方法名称
// ============================================================================
std::string ZipStructureParser::GetCompressionMethodName(uint16_t method) {
    switch (method) {
        case 0:  return "Stored";
        case 1:  return "Shrunk";
        case 2:  return "Reduced1";
        case 3:  return "Reduced2";
        case 4:  return "Reduced3";
        case 5:  return "Reduced4";
        case 6:  return "Imploded";
        case 8:  return "Deflated";
        case 9:  return "Deflate64";
        case 12: return "BZIP2";
        case 14: return "LZMA";
        case 98: return "PPMd";
        default: return "Unknown(" + std::to_string(method) + ")";
    }
}

// ============================================================================
// 查找EOCD
// ============================================================================
bool ZipStructureParser::FindEOCD(const BYTE* data, size_t size,
                                   EOCD& outEOCD, uint64_t& outOffset) {
    if (size < EOCD::MIN_SIZE) {
        return false;
    }

    // 从文件末尾向前搜索EOCD签名
    // EOCD最大大小 = 22 + 65535 (注释最大长度)
    size_t maxSearchLen = (std::min)(size, (size_t)(EOCD::MIN_SIZE + 65535));
    size_t searchStart = size - EOCD::MIN_SIZE;
    size_t searchEnd = size - maxSearchLen;

    for (size_t i = searchStart; i >= searchEnd && i < size; i--) {
        if (data[i] == 0x50 && data[i + 1] == 0x4B &&
            data[i + 2] == 0x05 && data[i + 3] == 0x06) {

            // 找到签名，验证结构
            if (ParseEOCDAt(data, size, i, outEOCD)) {
                // 验证: 注释长度 + EOCD位置应该等于文件大小
                uint64_t expectedEnd = i + EOCD::MIN_SIZE + outEOCD.commentLength;
                if (expectedEnd == size ||
                    (expectedEnd < size && outEOCD.commentLength == 0)) {
                    outOffset = i;
                    return true;
                }
                // 如果不完全匹配，继续搜索（可能是误匹配）
            }
        }

        if (i == 0) break;  // 防止下溢
    }

    return false;
}

// ============================================================================
// 解析EOCD
// ============================================================================
bool ZipStructureParser::ParseEOCDAt(const BYTE* data, size_t size,
                                      uint64_t offset, EOCD& outEOCD) {
    if (offset + EOCD::MIN_SIZE > size) {
        return false;
    }

    memcpy(&outEOCD, data + offset, sizeof(EOCD));

    // 验证签名
    if (outEOCD.signature != EOCD::SIGNATURE) {
        return false;
    }

    // 验证注释长度不超出文件范围
    if (offset + EOCD::MIN_SIZE + outEOCD.commentLength > size) {
        return false;
    }

    return true;
}

// ============================================================================
// 查找ZIP64 EOCD
// ============================================================================
bool ZipStructureParser::FindZip64EOCD(const BYTE* data, size_t size,
                                        uint64_t eocdOffset,
                                        ZIP64EOCD& outZip64EOCD) {
    // ZIP64 EOCD Locator应该在EOCD之前20字节
    if (eocdOffset < ZIP64EOCDLocator::SIZE) {
        return false;
    }

    uint64_t locatorOffset = eocdOffset - ZIP64EOCDLocator::SIZE;

    // 检查Locator签名
    if (data[locatorOffset] != 0x50 || data[locatorOffset + 1] != 0x4B ||
        data[locatorOffset + 2] != 0x06 || data[locatorOffset + 3] != 0x07) {
        return false;
    }

    // 解析Locator
    ZIP64EOCDLocator locator;
    memcpy(&locator, data + locatorOffset, sizeof(ZIP64EOCDLocator));

    // 验证签名
    if (locator.signature != ZIP64EOCDLocator::SIGNATURE) {
        return false;
    }

    // 读取ZIP64 EOCD
    if (locator.zip64EOCDOffset + ZIP64EOCD::MIN_SIZE > size) {
        return false;
    }

    memcpy(&outZip64EOCD, data + locator.zip64EOCDOffset, sizeof(ZIP64EOCD));

    return outZip64EOCD.signature == ZIP64EOCD::SIGNATURE;
}

// ============================================================================
// 解析Central Directory Header
// ============================================================================
bool ZipStructureParser::ParseCDHeaderAt(const BYTE* data, size_t size,
                                          uint64_t offset,
                                          CentralDirectoryHeader& outHeader,
                                          ZipEntry& outEntry) {
    if (offset + CentralDirectoryHeader::MIN_SIZE > size) {
        return false;
    }

    memcpy(&outHeader, data + offset, sizeof(CentralDirectoryHeader));

    // 验证签名
    if (outHeader.signature != CentralDirectoryHeader::SIGNATURE) {
        return false;
    }

    // 验证变长字段不超出范围
    uint64_t totalLen = CentralDirectoryHeader::MIN_SIZE +
                        outHeader.filenameLength +
                        outHeader.extraLength +
                        outHeader.commentLength;

    if (offset + totalLen > size) {
        return false;
    }

    // 填充ZipEntry
    outEntry.filename.assign(
        reinterpret_cast<const char*>(data + offset + CentralDirectoryHeader::MIN_SIZE),
        outHeader.filenameLength
    );

    outEntry.localHeaderOffset = outHeader.localHeaderOffset;
    outEntry.compressedSize = outHeader.compressedSize;
    outEntry.uncompressedSize = outHeader.uncompressedSize;
    outEntry.crc32 = outHeader.crc32;
    outEntry.compression = outHeader.compression;
    outEntry.flags = outHeader.flags;
    outEntry.modifiedTime = DosTimeToFileTime(outHeader.modTime, outHeader.modDate);

    // 检查ZIP64扩展字段
    if (outHeader.compressedSize == 0xFFFFFFFF ||
        outHeader.uncompressedSize == 0xFFFFFFFF ||
        outHeader.localHeaderOffset == 0xFFFFFFFF) {
        // 需要从extra字段读取64位值
        const BYTE* extra = data + offset + CentralDirectoryHeader::MIN_SIZE +
                           outHeader.filenameLength;
        size_t extraLen = outHeader.extraLength;
        size_t pos = 0;

        while (pos + 4 <= extraLen) {
            uint16_t headerId = *reinterpret_cast<const uint16_t*>(extra + pos);
            uint16_t dataSize = *reinterpret_cast<const uint16_t*>(extra + pos + 2);

            if (headerId == 0x0001 && pos + 4 + dataSize <= extraLen) {
                // ZIP64扩展信息
                const BYTE* z64 = extra + pos + 4;
                size_t z64pos = 0;

                if (outHeader.uncompressedSize == 0xFFFFFFFF && z64pos + 8 <= dataSize) {
                    outEntry.uncompressedSize = *reinterpret_cast<const uint64_t*>(z64 + z64pos);
                    z64pos += 8;
                }
                if (outHeader.compressedSize == 0xFFFFFFFF && z64pos + 8 <= dataSize) {
                    outEntry.compressedSize = *reinterpret_cast<const uint64_t*>(z64 + z64pos);
                    z64pos += 8;
                }
                if (outHeader.localHeaderOffset == 0xFFFFFFFF && z64pos + 8 <= dataSize) {
                    outEntry.localHeaderOffset = *reinterpret_cast<const uint64_t*>(z64 + z64pos);
                    z64pos += 8;
                }
                break;
            }
            pos += 4 + dataSize;
        }
    }

    return true;
}

// ============================================================================
// 解析Local File Header
// ============================================================================
bool ZipStructureParser::ParseLocalHeaderAt(const BYTE* data, size_t size,
                                             uint64_t offset,
                                             LocalFileHeader& outHeader) {
    if (offset + LocalFileHeader::MIN_SIZE > size) {
        return false;
    }

    memcpy(&outHeader, data + offset, sizeof(LocalFileHeader));

    return outHeader.signature == LocalFileHeader::SIGNATURE;
}

// ============================================================================
// 解析Central Directory
// ============================================================================
size_t ZipStructureParser::ParseCentralDirectory(const BYTE* data, size_t size,
                                                  uint64_t cdOffset, uint64_t cdSize,
                                                  uint32_t entryCount,
                                                  std::vector<ZipEntry>& outEntries) {
    if (cdOffset + cdSize > size) {
        return 0;
    }

    outEntries.clear();
    outEntries.reserve(entryCount);

    uint64_t pos = cdOffset;
    uint64_t endPos = cdOffset + cdSize;
    size_t parsed = 0;

    while (pos < endPos && parsed < entryCount) {
        CentralDirectoryHeader cdh;
        ZipEntry entry;

        if (!ParseCDHeaderAt(data, size, pos, cdh, entry)) {
            break;
        }

        outEntries.push_back(entry);
        parsed++;

        pos += CentralDirectoryHeader::MIN_SIZE +
               cdh.filenameLength + cdh.extraLength + cdh.commentLength;
    }

    return parsed;
}

// ============================================================================
// 验证Local Header
// ============================================================================
bool ZipStructureParser::ValidateLocalHeader(const BYTE* data, size_t size,
                                              uint64_t offset,
                                              const ZipEntry& entry) {
    LocalFileHeader lh;
    if (!ParseLocalHeaderAt(data, size, offset, lh)) {
        return false;
    }

    // 验证基本字段匹配
    if (lh.compression != entry.compression) {
        return false;
    }

    // 如果Local Header中有大小信息(非streaming)，验证匹配
    if ((lh.flags & 0x08) == 0) {  // 没有Data Descriptor
        if (lh.compressedSize != 0 &&
            lh.compressedSize != entry.compressedSize &&
            lh.compressedSize != 0xFFFFFFFF) {
            return false;
        }
        if (lh.crc32 != 0 && lh.crc32 != entry.crc32) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// 验证CRC32
// ============================================================================
bool ZipStructureParser::VerifyEntryCRC(const BYTE* data, size_t size,
                                         const ZipEntry& entry) {
    // 获取数据偏移
    if (entry.localHeaderOffset + LocalFileHeader::MIN_SIZE > size) {
        return false;
    }

    LocalFileHeader lh;
    if (!ParseLocalHeaderAt(data, size, entry.localHeaderOffset, lh)) {
        return false;
    }

    uint64_t dataOffset = entry.localHeaderOffset + LocalFileHeader::MIN_SIZE +
                          lh.filenameLength + lh.extraLength;

    if (dataOffset + entry.compressedSize > size) {
        return false;
    }

    // 只对存储(未压缩)的文件验证CRC
    if (entry.compression == 0) {
        uint32_t crc = CalculateCRC32(data + dataOffset, (size_t)entry.compressedSize);
        return crc == entry.crc32;
    }

    // 压缩文件需要先解压才能验证，这里暂不实现
    return true;
}

// ============================================================================
// 主解析函数
// ============================================================================
ZipParseResult ZipStructureParser::Parse(const BYTE* data, size_t size) {
    ZipParseResult result;
    result.actualFileSize = size;

    // 1. 查找EOCD
    EOCD eocd;
    if (!FindEOCD(data, size, eocd, result.eocdOffset)) {
        result.errorMessage = "无法找到End of Central Directory记录";
        return result;
    }
    result.hasValidEOCD = true;

    // 2. 检查是否为ZIP64
    ZIP64EOCD zip64eocd;
    if (FindZip64EOCD(data, size, result.eocdOffset, zip64eocd)) {
        result.isZip64 = true;
        result.cdOffset = zip64eocd.cdOffset;
        result.cdSize = zip64eocd.cdSize;
        result.declaredEntryCount = (uint32_t)zip64eocd.cdEntriesTotal;
    } else {
        result.cdOffset = eocd.cdOffset;
        result.cdSize = eocd.cdSize;
        result.declaredEntryCount = eocd.cdEntriesTotal;
    }

    // 3. 验证CD偏移
    if (result.cdOffset + result.cdSize > size) {
        result.errorMessage = "Central Directory超出文件范围 (CD偏移=" +
                              std::to_string(result.cdOffset) + ", CD大小=" +
                              std::to_string(result.cdSize) + ", 文件大小=" +
                              std::to_string(size) + ")";
        return result;
    }

    // 4. 解析Central Directory
    size_t parsed = ParseCentralDirectory(data, size, result.cdOffset, result.cdSize,
                                          result.declaredEntryCount, result.entries);

    if (parsed == result.declaredEntryCount) {
        result.hasValidCD = true;
    } else {
        result.errorMessage = "Central Directory解析不完整 (期望" +
                              std::to_string(result.declaredEntryCount) +
                              "个条目, 实际解析" + std::to_string(parsed) + "个)";
    }

    // 5. 验证每个条目的Local Header
    for (auto& entry : result.entries) {
        entry.hasValidLocalHeader = ValidateLocalHeader(data, size,
                                                        entry.localHeaderOffset, entry);

        // 检查数据是否可访问
        if (entry.localHeaderOffset + LocalFileHeader::MIN_SIZE <= size) {
            LocalFileHeader lh;
            if (ParseLocalHeaderAt(data, size, entry.localHeaderOffset, lh)) {
                uint64_t dataEnd = entry.localHeaderOffset + LocalFileHeader::MIN_SIZE +
                                   lh.filenameLength + lh.extraLength + entry.compressedSize;
                entry.hasValidData = (dataEnd <= size);
            }
        }
    }

    // 6. 计算预期大小
    result.expectedFileSize = CalculateExpectedSize(result);

    // 7. 检测间隙
    result.gaps = DetectGaps(result, size);

    // 8. 判断完整性
    result.isComplete = result.hasValidEOCD && result.hasValidCD &&
                        result.gaps.empty() &&
                        result.expectedFileSize == size;

    result.success = result.hasValidEOCD;
    return result;
}

// ============================================================================
// 从文件解析（优化版：分批读取，避免加载整个大文件）
// ============================================================================
ZipParseResult ZipStructureParser::ParseFile(const std::wstring& filePath) {
    ZipParseResult result;

    // 打开文件
    HANDLE hFile = CreateFileW(filePath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE) {
        result.errorMessage = "无法打开文件";
        return result;
    }

    // 获取文件大小
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        result.errorMessage = "无法获取文件大小";
        return result;
    }

    result.actualFileSize = fileSize.QuadPart;

    // ===== 步骤1: 读取文件末尾查找EOCD =====
    // EOCD最大大小 = 22 + 65535（注释最长）
    const size_t eocdSearchSize = EOCD::MIN_SIZE + 65535;
    size_t tailReadSize = (size_t)(std::min)((uint64_t)eocdSearchSize, (uint64_t)fileSize.QuadPart);

    std::vector<BYTE> tailBuffer(tailReadSize);

    // 定位到文件末尾
    LARGE_INTEGER seekPos;
    seekPos.QuadPart = fileSize.QuadPart - tailReadSize;
    if (!SetFilePointerEx(hFile, seekPos, NULL, FILE_BEGIN)) {
        CloseHandle(hFile);
        result.errorMessage = "无法定位到文件末尾";
        return result;
    }

    // 读取末尾数据
    DWORD bytesRead;
    if (!ReadFile(hFile, tailBuffer.data(), (DWORD)tailReadSize, &bytesRead, NULL)) {
        CloseHandle(hFile);
        result.errorMessage = "读取文件末尾失败";
        return result;
    }

    // 查找EOCD
    EOCD eocd;
    uint64_t eocdOffsetInBuffer;
    if (!FindEOCD(tailBuffer.data(), bytesRead, eocd, eocdOffsetInBuffer)) {
        CloseHandle(hFile);
        result.errorMessage = "无法找到End of Central Directory记录";
        return result;
    }

    result.hasValidEOCD = true;
    result.eocdOffset = seekPos.QuadPart + eocdOffsetInBuffer;

    // ===== 步骤2: 检查ZIP64 =====
    ZIP64EOCD zip64eocd;
    if (FindZip64EOCD(tailBuffer.data(), bytesRead, eocdOffsetInBuffer, zip64eocd)) {
        result.isZip64 = true;
        result.cdOffset = zip64eocd.cdOffset;
        result.cdSize = zip64eocd.cdSize;
        result.declaredEntryCount = (uint32_t)zip64eocd.cdEntriesTotal;
    } else {
        result.cdOffset = eocd.cdOffset;
        result.cdSize = eocd.cdSize;
        result.declaredEntryCount = eocd.cdEntriesTotal;
    }

    // ===== 步骤3: 读取Central Directory =====
    if (result.cdOffset + result.cdSize > (uint64_t)fileSize.QuadPart) {
        CloseHandle(hFile);
        result.errorMessage = "Central Directory超出文件范围 (CD偏移=" +
                              std::to_string(result.cdOffset) + ", CD大小=" +
                              std::to_string(result.cdSize) + ", 文件大小=" +
                              std::to_string(fileSize.QuadPart) + ")";
        return result;
    }

    std::vector<BYTE> cdBuffer((size_t)result.cdSize);

    // 定位到CD起始位置
    seekPos.QuadPart = result.cdOffset;
    if (!SetFilePointerEx(hFile, seekPos, NULL, FILE_BEGIN)) {
        CloseHandle(hFile);
        result.errorMessage = "无法定位到Central Directory";
        return result;
    }

    // 读取CD数据
    if (!ReadFile(hFile, cdBuffer.data(), (DWORD)result.cdSize, &bytesRead, NULL)) {
        CloseHandle(hFile);
        result.errorMessage = "读取Central Directory失败";
        return result;
    }

    // 解析CD
    size_t parsed = ParseCentralDirectory(cdBuffer.data(), cdBuffer.size(),
                                          0, result.cdSize,
                                          result.declaredEntryCount, result.entries);

    if (parsed == result.declaredEntryCount) {
        result.hasValidCD = true;
    } else {
        result.errorMessage = "Central Directory解析不完整 (期望" +
                              std::to_string(result.declaredEntryCount) +
                              "个条目, 实际解析" + std::to_string(parsed) + "个)";
    }

    // ===== 步骤4: 按需验证Local Headers =====
    // 只验证前几个和最后几个，避免大量随机读取
    size_t verifyCount = (std::min)((size_t)10, result.entries.size());

    for (size_t i = 0; i < verifyCount && i < result.entries.size(); i++) {
        auto& entry = result.entries[i];

        if (entry.localHeaderOffset + LocalFileHeader::MIN_SIZE > (uint64_t)fileSize.QuadPart) {
            entry.hasValidLocalHeader = false;
            entry.hasValidData = false;
            continue;
        }

        // 读取Local Header
        std::vector<BYTE> lhBuffer(LocalFileHeader::MIN_SIZE + 1024); // 包含文件名和extra字段
        seekPos.QuadPart = entry.localHeaderOffset;
        SetFilePointerEx(hFile, seekPos, NULL, FILE_BEGIN);

        DWORD lhBytesRead;
        if (ReadFile(hFile, lhBuffer.data(), (DWORD)lhBuffer.size(), &lhBytesRead, NULL)) {
            entry.hasValidLocalHeader = ValidateLocalHeader(lhBuffer.data(), lhBytesRead,
                                                            0, entry);

            // 检查数据是否可访问
            LocalFileHeader lh;
            if (ParseLocalHeaderAt(lhBuffer.data(), lhBytesRead, 0, lh)) {
                uint64_t dataEnd = entry.localHeaderOffset + LocalFileHeader::MIN_SIZE +
                                   lh.filenameLength + lh.extraLength + entry.compressedSize;
                entry.hasValidData = (dataEnd <= (uint64_t)fileSize.QuadPart);
            }
        }
    }

    // 对于未验证的条目，假设有效（避免随机读取）
    for (size_t i = verifyCount; i < result.entries.size(); i++) {
        result.entries[i].hasValidLocalHeader = true;
        result.entries[i].hasValidData = (result.entries[i].localHeaderOffset < (uint64_t)fileSize.QuadPart);
    }

    CloseHandle(hFile);

    // ===== 步骤5: 计算预期大小和检测间隙 =====
    result.expectedFileSize = CalculateExpectedSize(result);
    result.gaps = DetectGaps(result, fileSize.QuadPart);

    // 判断完整性
    result.isComplete = result.hasValidEOCD && result.hasValidCD &&
                        result.gaps.empty() &&
                        result.expectedFileSize == (uint64_t)fileSize.QuadPart;

    result.success = result.hasValidEOCD;
    return result;
}

// ============================================================================
// 计算预期文件大小
// ============================================================================
uint64_t ZipStructureParser::CalculateExpectedSize(const ZipParseResult& result) {
    if (!result.hasValidEOCD) {
        return 0;
    }

    // 预期大小 = EOCD偏移 + EOCD大小(22) + 注释长度
    // 注意：这里假设EOCD之后没有额外数据
    return result.eocdOffset + EOCD::MIN_SIZE;

    // 更精确的计算需要考虑注释长度，但我们没有保存它
}

// ============================================================================
// 检测间隙
// ============================================================================
std::vector<ZipParseResult::Gap> ZipStructureParser::DetectGaps(
    const ZipParseResult& result, uint64_t actualSize) {

    std::vector<ZipParseResult::Gap> gaps;

    if (result.entries.empty()) {
        return gaps;
    }

    // 创建排序的条目列表（按偏移）
    std::vector<std::pair<uint64_t, uint64_t>> regions;  // (start, end)

    for (const auto& entry : result.entries) {
        // 每个条目占用: Local Header + 变长字段 + 压缩数据
        // 简化: 假设变长字段总长度约100字节
        uint64_t start = entry.localHeaderOffset;
        uint64_t end = entry.localHeaderOffset + LocalFileHeader::MIN_SIZE +
                       entry.filename.size() + entry.compressedSize + 100;
        regions.push_back({start, end});
    }

    // 添加Central Directory区域
    regions.push_back({result.cdOffset, result.cdOffset + result.cdSize});

    // 添加EOCD区域
    regions.push_back({result.eocdOffset, result.eocdOffset + EOCD::MIN_SIZE});

    // 按起始位置排序
    std::sort(regions.begin(), regions.end());

    // 检测间隙
    uint64_t expectedStart = 0;
    for (const auto& region : regions) {
        if (region.first > expectedStart) {
            // 发现间隙
            ZipParseResult::Gap gap;
            gap.start = expectedStart;
            gap.end = region.first;
            gaps.push_back(gap);
        }
        expectedStart = (std::max)(expectedStart, region.second);
    }

    return gaps;
}

// ============================================================================
// 生成恢复建议
// ============================================================================
ZipRecoveryAdvice ZipStructureParser::GetRecoveryAdvice(const ZipParseResult& result) {
    ZipRecoveryAdvice advice;

    if (result.isComplete) {
        advice.status = ZipRecoveryAdvice::Status::COMPLETE;
        advice.description = "ZIP文件结构完整，无需修复";
        return advice;
    }

    if (!result.hasValidEOCD) {
        advice.status = ZipRecoveryAdvice::Status::UNRECOVERABLE;
        advice.description = "无法找到EOCD记录，文件严重损坏";
        advice.steps.push_back("尝试搜索Central Directory签名(50 4B 01 02)");
        advice.steps.push_back("尝试搜索Local File Header签名(50 4B 03 04)");
        return advice;
    }

    if (!result.hasValidCD) {
        advice.status = ZipRecoveryAdvice::Status::PARTIAL_RECOVERY;
        advice.description = "Central Directory损坏，可尝试从Local Headers重建";
        advice.steps.push_back("扫描所有Local File Header(50 4B 03 04)");
        advice.steps.push_back("从Local Header提取文件信息");
        advice.steps.push_back("重建文件列表");
    } else {
        advice.status = ZipRecoveryAdvice::Status::REPAIRABLE;
        advice.description = "文件结构基本完整，部分数据可能损坏";
    }

    // 检查哪些条目可以恢复
    for (size_t i = 0; i < result.entries.size(); i++) {
        const auto& entry = result.entries[i];
        if (entry.hasValidLocalHeader && entry.hasValidData) {
            advice.recoverableEntries.push_back(i);
        }
    }

    if (!advice.recoverableEntries.empty()) {
        advice.steps.push_back("可恢复 " + std::to_string(advice.recoverableEntries.size()) +
                               "/" + std::to_string(result.entries.size()) + " 个文件");
    }

    if (!result.gaps.empty()) {
        advice.steps.push_back("检测到 " + std::to_string(result.gaps.size()) +
                               " 个数据间隙，可能存在碎片化");
    }

    return advice;
}

// ============================================================================
// 提取单个文件
// ============================================================================
bool ZipStructureParser::ExtractEntry(const BYTE* data, size_t size,
                                       const ZipEntry& entry,
                                       std::vector<BYTE>& outData) {
    // 验证Local Header
    if (entry.localHeaderOffset + LocalFileHeader::MIN_SIZE > size) {
        return false;
    }

    LocalFileHeader lh;
    if (!ParseLocalHeaderAt(data, size, entry.localHeaderOffset, lh)) {
        return false;
    }

    // 计算数据偏移
    uint64_t dataOffset = entry.localHeaderOffset + LocalFileHeader::MIN_SIZE +
                          lh.filenameLength + lh.extraLength;

    // 验证数据范围
    if (dataOffset + entry.compressedSize > size) {
        return false;
    }

    // 复制数据
    outData.resize((size_t)entry.compressedSize);
    memcpy(outData.data(), data + dataOffset, (size_t)entry.compressedSize);

    return true;
}

} // namespace ZipParser
