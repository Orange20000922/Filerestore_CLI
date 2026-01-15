#pragma once
/**
 * @file ZipStructureParser.h
 * @brief ZIP文件结构解析器 - 用于文件恢复和完整性验证
 *
 * 功能:
 * - 解析EOCD、Central Directory、Local File Header
 * - 提取文件列表和偏移信息
 * - 验证ZIP结构完整性
 * - 支持ZIP64扩展格式
 * - 检测碎片和损坏
 */

#include <Windows.h>
#include <string>
#include <vector>
#include <optional>
#include <cstdint>

namespace ZipParser {

// ============================================================================
// ZIP结构体定义 (按规范字节对齐)
// ============================================================================

#pragma pack(push, 1)

/**
 * End of Central Directory Record (EOCD)
 * 签名: 0x06054B50 (50 4B 05 06)
 * 最小大小: 22字节
 */
struct EOCD {
    uint32_t signature;           // 0x06054B50
    uint16_t diskNumber;          // 当前磁盘号
    uint16_t diskCDStart;         // CD开始的磁盘号
    uint16_t cdEntriesThisDisk;   // 本磁盘CD条目数
    uint16_t cdEntriesTotal;      // CD条目总数
    uint32_t cdSize;              // CD总大小(字节)
    uint32_t cdOffset;            // CD偏移(从文件开头)
    uint16_t commentLength;       // 注释长度
    // comment[commentLength] 紧随其后

    static constexpr uint32_t SIGNATURE = 0x06054B50;
    static constexpr size_t MIN_SIZE = 22;
};

/**
 * ZIP64 End of Central Directory Locator
 * 签名: 0x07064B50 (50 4B 06 07)
 * 大小: 20字节
 */
struct ZIP64EOCDLocator {
    uint32_t signature;           // 0x07064B50
    uint32_t diskZip64EOCD;       // ZIP64 EOCD所在磁盘号
    uint64_t zip64EOCDOffset;     // ZIP64 EOCD偏移(64位)
    uint32_t totalDisks;          // 磁盘总数

    static constexpr uint32_t SIGNATURE = 0x07064B50;
    static constexpr size_t SIZE = 20;
};

/**
 * ZIP64 End of Central Directory Record
 * 签名: 0x06064B50 (50 4B 06 06)
 */
struct ZIP64EOCD {
    uint32_t signature;           // 0x06064B50
    uint64_t recordSize;          // 本记录剩余大小
    uint16_t versionMadeBy;       // 创建版本
    uint16_t versionNeeded;       // 所需版本
    uint32_t diskNumber;          // 当前磁盘号
    uint32_t diskCDStart;         // CD开始磁盘号
    uint64_t cdEntriesThisDisk;   // 本磁盘CD条目数(64位)
    uint64_t cdEntriesTotal;      // CD总条目数(64位)
    uint64_t cdSize;              // CD大小(64位)
    uint64_t cdOffset;            // CD偏移(64位)
    // extensible data 可选

    static constexpr uint32_t SIGNATURE = 0x06064B50;
    static constexpr size_t MIN_SIZE = 56;
};

/**
 * Central Directory File Header
 * 签名: 0x02014B50 (50 4B 01 02)
 * 最小大小: 46字节
 */
struct CentralDirectoryHeader {
    uint32_t signature;           // 0x02014B50
    uint16_t versionMadeBy;       // 创建版本
    uint16_t versionNeeded;       // 解压所需版本
    uint16_t flags;               // 通用标志
    uint16_t compression;         // 压缩方法
    uint16_t modTime;             // 修改时间(DOS格式)
    uint16_t modDate;             // 修改日期(DOS格式)
    uint32_t crc32;               // CRC校验
    uint32_t compressedSize;      // 压缩后大小
    uint32_t uncompressedSize;    // 原始大小
    uint16_t filenameLength;      // 文件名长度
    uint16_t extraLength;         // 扩展字段长度
    uint16_t commentLength;       // 注释长度
    uint16_t diskStart;           // 起始磁盘号
    uint16_t internalAttr;        // 内部属性
    uint32_t externalAttr;        // 外部属性
    uint32_t localHeaderOffset;   // Local Header偏移
    // filename[filenameLength]
    // extra[extraLength]
    // comment[commentLength]

    static constexpr uint32_t SIGNATURE = 0x02014B50;
    static constexpr size_t MIN_SIZE = 46;
};

/**
 * Local File Header
 * 签名: 0x04034B50 (50 4B 03 04)
 * 最小大小: 30字节
 */
struct LocalFileHeader {
    uint32_t signature;           // 0x04034B50
    uint16_t versionNeeded;       // 解压所需版本
    uint16_t flags;               // 通用标志
    uint16_t compression;         // 压缩方法
    uint16_t modTime;             // 修改时间
    uint16_t modDate;             // 修改日期
    uint32_t crc32;               // CRC校验(可能为0)
    uint32_t compressedSize;      // 压缩后大小(可能为0)
    uint32_t uncompressedSize;    // 原始大小(可能为0)
    uint16_t filenameLength;      // 文件名长度
    uint16_t extraLength;         // 扩展字段长度
    // filename[filenameLength]
    // extra[extraLength]
    // file_data[compressedSize]

    static constexpr uint32_t SIGNATURE = 0x04034B50;
    static constexpr size_t MIN_SIZE = 30;
};

/**
 * Data Descriptor (可选,当flags bit 3设置时使用)
 * 签名: 0x08074B50 (50 4B 07 08) - 可选
 */
struct DataDescriptor {
    uint32_t signature;           // 0x08074B50 (可选)
    uint32_t crc32;
    uint32_t compressedSize;
    uint32_t uncompressedSize;

    static constexpr uint32_t SIGNATURE = 0x08074B50;
};

#pragma pack(pop)

// ============================================================================
// 高级数据结构
// ============================================================================

/**
 * ZIP文件中的单个条目信息
 */
struct ZipEntry {
    std::string filename;              // 文件名(UTF-8)
    uint64_t localHeaderOffset;        // Local Header在文件中的绝对偏移
    uint64_t compressedSize;           // 压缩后大小
    uint64_t uncompressedSize;         // 原始大小
    uint32_t crc32;                    // CRC32校验值
    uint16_t compression;              // 压缩方法 (0=存储, 8=Deflate)
    uint16_t flags;                    // 通用标志
    FILETIME modifiedTime;             // 修改时间(Windows格式)

    // 验证状态
    bool hasValidLocalHeader;          // Local Header是否有效
    bool hasValidData;                 // 数据是否可访问
    bool crcVerified;                  // CRC是否验证通过

    // 计算数据的实际偏移 (Local Header + 变长字段之后)
    uint64_t GetDataOffset() const {
        return localHeaderOffset + LocalFileHeader::MIN_SIZE +
               filename.size() + 0;  // extra长度需要实际读取
    }
};

/**
 * ZIP文件解析结果
 */
struct ZipParseResult {
    // 状态标志
    bool success;                      // 解析是否成功
    bool hasValidEOCD;                 // EOCD是否有效
    bool hasValidCD;                   // Central Directory是否有效
    bool isZip64;                      // 是否为ZIP64格式
    bool isComplete;                   // 文件是否完整

    // 位置信息
    uint64_t eocdOffset;               // EOCD在文件中的偏移
    uint64_t cdOffset;                 // Central Directory偏移
    uint64_t cdSize;                   // Central Directory大小
    uint64_t expectedFileSize;         // 根据结构计算的预期文件大小
    uint64_t actualFileSize;           // 实际文件大小

    // 文件列表
    uint32_t declaredEntryCount;       // EOCD中声明的文件数
    std::vector<ZipEntry> entries;     // 解析出的文件列表

    // 错误信息
    std::string errorMessage;

    // 碎片/间隙信息
    struct Gap {
        uint64_t start;
        uint64_t end;
        uint64_t size() const { return end - start; }
    };
    std::vector<Gap> gaps;             // 检测到的间隙

    ZipParseResult() : success(false), hasValidEOCD(false), hasValidCD(false),
                       isZip64(false), isComplete(false), eocdOffset(0),
                       cdOffset(0), cdSize(0), expectedFileSize(0),
                       actualFileSize(0), declaredEntryCount(0) {}
};

/**
 * ZIP恢复建议
 */
struct ZipRecoveryAdvice {
    enum class Status {
        COMPLETE,           // 完整,无需修复
        REPAIRABLE,         // 可修复
        PARTIAL_RECOVERY,   // 可部分恢复
        UNRECOVERABLE       // 无法恢复
    };

    Status status;
    std::string description;
    std::vector<std::string> steps;    // 建议的修复步骤
    std::vector<size_t> recoverableEntries;  // 可恢复的文件索引
};

// ============================================================================
// ZipStructureParser 类
// ============================================================================

class ZipStructureParser {
public:
    /**
     * 从内存数据解析ZIP结构
     * @param data 文件数据
     * @param size 数据大小
     * @return 解析结果
     */
    static ZipParseResult Parse(const BYTE* data, size_t size);

    /**
     * 从文件解析ZIP结构
     * @param filePath 文件路径
     * @return 解析结果
     */
    static ZipParseResult ParseFile(const std::wstring& filePath);

    /**
     * 仅查找并解析EOCD
     * @param data 文件数据
     * @param size 数据大小
     * @param outEOCD 输出EOCD结构
     * @param outOffset 输出EOCD偏移
     * @return 是否找到有效EOCD
     */
    static bool FindEOCD(const BYTE* data, size_t size,
                         EOCD& outEOCD, uint64_t& outOffset);

    /**
     * 查找ZIP64 EOCD
     * @param data 文件数据
     * @param size 数据大小
     * @param eocdOffset 普通EOCD的偏移
     * @param outZip64EOCD 输出ZIP64 EOCD
     * @return 是否找到有效ZIP64 EOCD
     */
    static bool FindZip64EOCD(const BYTE* data, size_t size,
                              uint64_t eocdOffset,
                              ZIP64EOCD& outZip64EOCD);

    /**
     * 解析Central Directory
     * @param data 文件数据
     * @param size 数据大小
     * @param cdOffset CD偏移
     * @param cdSize CD大小
     * @param entryCount 期望的条目数
     * @param outEntries 输出文件列表
     * @return 成功解析的条目数
     */
    static size_t ParseCentralDirectory(const BYTE* data, size_t size,
                                        uint64_t cdOffset, uint64_t cdSize,
                                        uint32_t entryCount,
                                        std::vector<ZipEntry>& outEntries);

    /**
     * 验证Local File Header
     * @param data 文件数据
     * @param size 数据大小
     * @param offset Local Header偏移
     * @param entry 对应的CD条目(用于验证)
     * @return 是否有效
     */
    static bool ValidateLocalHeader(const BYTE* data, size_t size,
                                    uint64_t offset, const ZipEntry& entry);

    /**
     * 验证文件数据CRC32
     * @param data 文件数据
     * @param size 数据大小
     * @param entry ZIP条目
     * @return CRC是否匹配
     */
    static bool VerifyEntryCRC(const BYTE* data, size_t size,
                               const ZipEntry& entry);

    /**
     * 检测数据间隙
     * @param result 解析结果
     * @param actualSize 实际文件大小
     * @return 间隙列表
     */
    static std::vector<ZipParseResult::Gap> DetectGaps(
        const ZipParseResult& result, uint64_t actualSize);

    /**
     * 生成恢复建议
     * @param result 解析结果
     * @return 恢复建议
     */
    static ZipRecoveryAdvice GetRecoveryAdvice(const ZipParseResult& result);

    /**
     * 计算预期文件大小
     * @param result 解析结果
     * @return 预期大小
     */
    static uint64_t CalculateExpectedSize(const ZipParseResult& result);

    /**
     * 尝试从损坏的ZIP中提取单个文件
     * @param data ZIP文件数据
     * @param size 数据大小
     * @param entry 要提取的文件条目
     * @param outData 输出文件数据
     * @return 是否成功
     */
    static bool ExtractEntry(const BYTE* data, size_t size,
                             const ZipEntry& entry,
                             std::vector<BYTE>& outData);

    // ========== 辅助函数 ==========

    /**
     * DOS时间转Windows FILETIME
     */
    static FILETIME DosTimeToFileTime(uint16_t dosTime, uint16_t dosDate);

    /**
     * 获取压缩方法名称
     */
    static std::string GetCompressionMethodName(uint16_t method);

    /**
     * 计算CRC32
     */
    static uint32_t CalculateCRC32(const BYTE* data, size_t size);

private:
    // 内部解析函数
    static bool ParseEOCDAt(const BYTE* data, size_t size, uint64_t offset,
                            EOCD& outEOCD);

    static bool ParseCDHeaderAt(const BYTE* data, size_t size, uint64_t offset,
                                CentralDirectoryHeader& outHeader,
                                ZipEntry& outEntry);

    static bool ParseLocalHeaderAt(const BYTE* data, size_t size, uint64_t offset,
                                   LocalFileHeader& outHeader);

    // CRC32表
    static const uint32_t crcTable[256];
};

} // namespace ZipParser
