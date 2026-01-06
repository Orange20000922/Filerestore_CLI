#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <map>

using namespace std;

// 提取的时间戳结果
struct ExtractedTimestamp {
    FILETIME creationTime;
    FILETIME modificationTime;
    FILETIME accessTime;
    bool hasCreation;
    bool hasModification;
    bool hasAccess;
    string additionalInfo;      // 额外信息（如相机型号、软件名等）

    ExtractedTimestamp() : hasCreation(false), hasModification(false), hasAccess(false) {
        memset(&creationTime, 0, sizeof(FILETIME));
        memset(&modificationTime, 0, sizeof(FILETIME));
        memset(&accessTime, 0, sizeof(FILETIME));
    }

    bool hasAnyTimestamp() const {
        return hasCreation || hasModification || hasAccess;
    }
};

// 时间戳提取器 - 从文件内嵌元数据提取时间戳
class TimestampExtractor {
public:
    // 根据文件类型自动选择提取方法
    static ExtractedTimestamp Extract(const BYTE* data, size_t dataSize, const string& extension);

    // 特定格式提取方法
    static ExtractedTimestamp ExtractFromJPEG(const BYTE* data, size_t dataSize);
    static ExtractedTimestamp ExtractFromPNG(const BYTE* data, size_t dataSize);
    static ExtractedTimestamp ExtractFromPDF(const BYTE* data, size_t dataSize);
    static ExtractedTimestamp ExtractFromZIP(const BYTE* data, size_t dataSize);
    static ExtractedTimestamp ExtractFromMP4(const BYTE* data, size_t dataSize);

private:
    // EXIF 解析辅助函数
    static bool ParseEXIF(const BYTE* data, size_t dataSize, ExtractedTimestamp& result);
    static bool ParseIFD(const BYTE* tiffStart, size_t tiffSize, DWORD ifdOffset,
                         bool bigEndian, ExtractedTimestamp& result);

    // 时间格式转换辅助函数
    static bool ParseEXIFDateTime(const char* dateStr, FILETIME& ft);
    static bool DOSTimeToFileTime(WORD dosDate, WORD dosTime, FILETIME& ft);
    static bool UnixTimeToFileTime(ULONGLONG unixTime, FILETIME& ft);
    static bool MP4TimeToFileTime(ULONGLONG mp4Time, FILETIME& ft);

    // 字节序转换
    static WORD ReadWord(const BYTE* data, bool bigEndian);
    static DWORD ReadDWord(const BYTE* data, bool bigEndian);
    static ULONGLONG ReadQWord(const BYTE* data, bool bigEndian);
};
