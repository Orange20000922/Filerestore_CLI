#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <memory>
#include <cctype>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "MFTReader.h"

using namespace std;

namespace CommandUtils {

// ============================================================================
// 驱动器字母验证和提取
// ============================================================================

/**
 * 验证驱动器字母并转换为大写
 * @param driveStr 驱动器字符串
 * @param outDrive 输出的驱动器字母（大写）
 * @return 验证成功返回 true
 */
inline bool ValidateDriveLetter(const string& driveStr, char& outDrive) {
    if (driveStr.empty()) {
        return false;
    }

    char drive = driveStr[0];
    if (!isalpha(drive)) {
        return false;
    }

    outDrive = (char)toupper(drive);
    return true;
}

// ============================================================================
// MFT 记录号解析
// ============================================================================

/**
 * 解析 MFT 记录号
 * @param recordStr 记录号字符串
 * @param outRecord 输出的记录号
 * @return 解析成功返回 true
 */
inline bool ParseRecordNumber(const string& recordStr, ULONGLONG& outRecord) {
    try {
        outRecord = stoull(recordStr);
        return true;
    } catch (const exception&) {
        return false;
    }
}

/**
 * 解析多个记录号（逗号分隔）
 * @param recordsStr 记录号列表字符串（如 "5,10,15"）
 * @param outRecords 输出的记录号向量
 * @return 解析成功返回 true
 */
inline bool ParseRecordNumbers(const string& recordsStr, vector<ULONGLONG>& outRecords) {
    string remaining = recordsStr;
    size_t pos = 0;

    while ((pos = remaining.find(',')) != string::npos) {
        string token = remaining.substr(0, pos);
        if (!token.empty()) {
            try {
                outRecords.push_back(stoull(token));
            } catch (...) {
                return false;
            }
        }
        remaining.erase(0, pos + 1);
    }

    // 处理最后一个记录号
    if (!remaining.empty()) {
        try {
            outRecords.push_back(stoull(remaining));
        } catch (...) {
            return false;
        }
    }

    return !outRecords.empty();
}

// ============================================================================
// 文件类型解析
// ============================================================================

/**
 * 解析文件类型列表
 * @param typeArg 文件类型参数（如 "all", "jpg", "jpg,png,gif"）
 * @param supportedTypes 支持的文件类型列表（格式：" type - description"）
 * @return 解析后的文件类型列表
 */
inline vector<string> ParseFileTypes(const string& typeArg,
                                      const vector<string>& supportedTypes) {
    vector<string> types;

    if (typeArg == "all") {
        // 提取所有类型名（去除描述）
        for (const auto& t : supportedTypes) {
            size_t dashPos = t.find(" - ");
            if (dashPos != string::npos) {
                types.push_back(t.substr(0, dashPos));
            }
        }
    } else {
        // 解析逗号分隔的类型列表
        string typeCopy = typeArg;
        size_t pos = 0;
        while ((pos = typeCopy.find(',')) != string::npos) {
            string type = typeCopy.substr(0, pos);
            if (!type.empty()) {
                types.push_back(type);
            }
            typeCopy.erase(0, pos + 1);
        }
        if (!typeCopy.empty()) {
            types.push_back(typeCopy);
        }
    }

    return types;
}

// ============================================================================
// MFTReader 打开封装
// ============================================================================

/**
 * 打开卷（使用裸指针）
 * @param driveLetter 驱动器字母
 * @param reader MFTReader 对象
 * @return 打开成功返回 true
 */
inline bool OpenVolume(char driveLetter, MFTReader& reader) {
    return reader.OpenVolume(driveLetter);
}

/**
 * 打开卷（使用智能指针）
 * @param driveLetter 驱动器字母
 * @param reader MFTReader 智能指针（自动创建）
 * @return 打开成功返回 true
 */
inline bool OpenVolume(char driveLetter, unique_ptr<MFTReader>& reader) {
    reader = make_unique<MFTReader>();
    return reader->OpenVolume(driveLetter);
}

// ============================================================================
// 输出目录创建
// ============================================================================

/**
 * 创建输出目录（如果不存在）
 * @param path 目录路径
 * @return 创建成功或目录已存在返回 true
 */
inline bool CreateOutputDirectory(const string& path) {
    if (!CreateDirectoryA(path.c_str(), NULL)) {
        DWORD err = GetLastError();
        if (err != ERROR_ALREADY_EXISTS) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// 大小范围解析
// ============================================================================

/**
 * 解析文件大小字符串（支持 K/M/G 后缀）
 * @param sizeStr 大小字符串（如 "1024", "10K", "5M", "2G"）
 * @param outSize 输出的大小（字节）
 * @return 解析成功返回 true
 */
inline bool ParseFileSize(const string& sizeStr, ULONGLONG& outSize) {
    if (sizeStr.empty()) {
        return false;
    }

    string numPart;
    char suffix = 0;

    // 分离数字和后缀
    for (char c : sizeStr) {
        if (isdigit(c)) {
            numPart += c;
        } else if (isalpha(c)) {
            suffix = (char)toupper(c);
            break;
        }
    }

    if (numPart.empty()) {
        return false;
    }

    try {
        ULONGLONG size = stoull(numPart);

        // 应用后缀乘数
        switch (suffix) {
            case 'K': size *= 1024ULL; break;
            case 'M': size *= 1024ULL * 1024ULL; break;
            case 'G': size *= 1024ULL * 1024ULL * 1024ULL; break;
            case 0: break;  // 无后缀
            default: return false;  // 无效后缀
        }

        outSize = size;
        return true;
    } catch (...) {
        return false;
    }
}

/**
 * 解析文件大小字符串（简化版本，直接返回大小）
 * @param sizeStr 大小字符串（如 "100MB", "2GB"）
 * @return 解析后的大小（字节），失败返回 0
 */
inline ULONGLONG ParseSize(const string& sizeStr) {
    ULONGLONG size = 0;
    ParseFileSize(sizeStr, size);
    return size;
}

// ============================================================================
// 进度显示标准格式
// ============================================================================

/**
 * 打印进度信息（标准格式）
 * @param operation 操作名称
 * @param percentage 完成百分比 (0-100)
 * @param additionalInfo 附加信息（可选）
 */
inline void PrintProgress(const string& operation, double percentage,
                          const string& additionalInfo = "") {
    cout << "\r[" << operation << "] "
         << fixed << setprecision(1) << percentage << "% "
         << additionalInfo << flush;
}

/**
 * 清除进度行
 */
inline void ClearProgressLine() {
    cout << "\r" << string(80, ' ') << "\r" << flush;
}

// ============================================================================
// 卷初始化（驱动器验证 + 打开卷）
// ============================================================================

/**
 * 验证驱动器并打开卷（一体化初始化）
 * @param driveStr 驱动器字符串
 * @param reader MFTReader 对象引用
 * @param outDrive 输出的驱动器字母（可选）
 * @return 成功返回 true，失败会输出错误信息
 */
inline bool OpenVolumeWithValidation(const string& driveStr, MFTReader& reader, char* outDrive = nullptr) {
    char driveLetter;
    if (!ValidateDriveLetter(driveStr, driveLetter)) {
        cout << "Invalid drive letter: " << driveStr << endl;
        return false;
    }

    if (!reader.OpenVolume(driveLetter)) {
        cout << "Failed to open volume " << driveLetter << ":/" << endl;
        return false;
    }

    if (outDrive) {
        *outDrive = driveLetter;
    }
    return true;
}

// ============================================================================
// 结果显示工具
// ============================================================================

/**
 * 显示结果列表（带数量限制和"还有 X 条"提示）
 * @param results 结果向量
 * @param displayFunc 显示单条结果的函数 (index, item) -> void
 * @param limit 显示数量限制（默认100）
 */
template<typename T, typename DisplayFunc>
inline void DisplayResultsWithLimit(const vector<T>& results, DisplayFunc displayFunc, size_t limit = 100) {
    size_t displayCount = min(results.size(), limit);

    for (size_t i = 0; i < displayCount; i++) {
        displayFunc(i, results[i]);
    }

    if (results.size() > limit) {
        cout << "\n... and " << (results.size() - limit) << " more items" << endl;
        cout << "Use specific commands to view full results" << endl;
    }
}

/**
 * 格式化文件大小为可读字符串
 * @param size 大小（字节）
 * @return 格式化后的字符串（如 "1.5 MB"）
 */
inline string FormatFileSize(ULONGLONG size) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double displaySize = static_cast<double>(size);

    while (displaySize >= 1024.0 && unitIndex < 4) {
        displaySize /= 1024.0;
        unitIndex++;
    }

    ostringstream oss;
    if (unitIndex == 0) {
        oss << size << " B";
    } else {
        oss << fixed << setprecision(2) << displaySize << " " << units[unitIndex];
    }
    return oss.str();
}

// ============================================================================
// 字符串转换工具
// ============================================================================

/**
 * 宽字符串转窄字符串（UTF-16 -> UTF-8/ANSI）
 * @param wide 宽字符串
 * @return 窄字符串
 */
inline string WideToNarrow(const wstring& wide) {
    if (wide.empty()) return "";

    int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) return "";

    string result(size - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, &result[0], size, nullptr, nullptr);
    return result;
}

/**
 * 窄字符串转宽字符串（UTF-8/ANSI -> UTF-16）
 * @param narrow 窄字符串
 * @return 宽字符串
 */
inline wstring NarrowToWide(const string& narrow) {
    if (narrow.empty()) return L"";

    int size = MultiByteToWideChar(CP_UTF8, 0, narrow.c_str(), -1, nullptr, 0);
    if (size <= 0) return L"";

    wstring result(size - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, narrow.c_str(), -1, &result[0], size);
    return result;
}

// ============================================================================
// 时间格式化工具
// ============================================================================

/**
 * FILETIME 转可读字符串
 * @param ft FILETIME 结构
 * @return 格式化的时间字符串
 */
inline string FileTimeToString(const FILETIME& ft) {
    if (ft.dwHighDateTime == 0 && ft.dwLowDateTime == 0) {
        return "N/A";
    }

    SYSTEMTIME st;
    FileTimeToSystemTime(&ft, &st);

    ostringstream oss;
    oss << setfill('0')
        << st.wYear << "-"
        << setw(2) << st.wMonth << "-"
        << setw(2) << st.wDay << " "
        << setw(2) << st.wHour << ":"
        << setw(2) << st.wMinute << ":"
        << setw(2) << st.wSecond;
    return oss.str();
}

} // namespace CommandUtils
