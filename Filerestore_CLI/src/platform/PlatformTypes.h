#pragma once

#include "PlatformConfig.h"
#include <chrono>
#include <string>
#include <ctime>
#include <sstream>
#include <iomanip>

#ifdef FR_PLATFORM_WINDOWS
#include <Windows.h>
#endif

namespace Platform {

// ============================================================================
// FileTime - 跨平台时间戳类
// ============================================================================
class FileTime {
private:
    // 内部存储：微秒，自 Unix 纪元 (1970-01-01 00:00:00 UTC)
    Timestamp microseconds_;

public:
    // 默认构造函数
    FileTime() : microseconds_(0) {}

    // 从微秒构造
    explicit FileTime(Timestamp microseconds) : microseconds_(microseconds) {}

    // 从 std::chrono::system_clock::time_point 构造
    explicit FileTime(const std::chrono::system_clock::time_point& tp) {
        auto duration = tp.time_since_epoch();
        microseconds_ = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }

#ifdef FR_PLATFORM_WINDOWS
    // 从 Windows FILETIME 构造
    explicit FileTime(const FILETIME& ft) {
        ULARGE_INTEGER uli;
        uli.LowPart = ft.dwLowDateTime;
        uli.HighPart = ft.dwHighDateTime;

        // FILETIME: 100纳秒间隔，自 1601-01-01
        // 转换为微秒
        Timestamp fileTimeMicroseconds = uli.QuadPart / 10;

        // 1601 到 1970 的差值（微秒）
        // 11644473600 秒 = 369年
        const Timestamp EPOCH_DIFF_MICROSECONDS = 11644473600LL * 1000000LL;

        if (fileTimeMicroseconds >= EPOCH_DIFF_MICROSECONDS) {
            microseconds_ = fileTimeMicroseconds - EPOCH_DIFF_MICROSECONDS;
        } else {
            microseconds_ = 0;
        }
    }

    // 转换为 Windows FILETIME
    FILETIME ToWindowsFileTime() const {
        const Timestamp EPOCH_DIFF_MICROSECONDS = 11644473600LL * 1000000LL;
        Timestamp fileTimeMicroseconds = microseconds_ + EPOCH_DIFF_MICROSECONDS;

        // 转换为 100 纳秒单位
        ULARGE_INTEGER uli;
        uli.QuadPart = fileTimeMicroseconds * 10;

        FILETIME ft;
        ft.dwLowDateTime = uli.LowPart;
        ft.dwHighDateTime = uli.HighPart;
        return ft;
    }
#endif

    // 获取微秒值
    Timestamp ToMicroseconds() const { return microseconds_; }

    // 获取毫秒值
    Timestamp ToMilliseconds() const { return microseconds_ / 1000; }

    // 获取秒值
    Timestamp ToSeconds() const { return microseconds_ / 1000000; }

    // 转换为 time_t
    std::time_t ToTimeT() const {
        return static_cast<std::time_t>(microseconds_ / 1000000);
    }

    // 转换为 std::chrono::system_clock::time_point
    std::chrono::system_clock::time_point ToTimePoint() const {
        return std::chrono::system_clock::time_point(
            std::chrono::microseconds(microseconds_)
        );
    }

    // 转换为本地时间字符串
    std::string ToLocalString(const char* format = "%Y-%m-%d %H:%M:%S") const {
        if (microseconds_ == 0) {
            return "N/A";
        }

        std::time_t time = ToTimeT();
        std::tm tm_local;

#ifdef FR_PLATFORM_WINDOWS
        localtime_s(&tm_local, &time);
#else
        localtime_r(&time, &tm_local);
#endif

        std::ostringstream oss;
        oss << std::put_time(&tm_local, format);
        return oss.str();
    }

    // 转换为 UTC 时间字符串
    std::string ToUtcString(const char* format = "%Y-%m-%d %H:%M:%S") const {
        if (microseconds_ == 0) {
            return "N/A";
        }

        std::time_t time = ToTimeT();
        std::tm tm_utc;

#ifdef FR_PLATFORM_WINDOWS
        gmtime_s(&tm_utc, &time);
#else
        gmtime_r(&time, &tm_utc);
#endif

        std::ostringstream oss;
        oss << std::put_time(&tm_utc, format);
        return oss.str();
    }

    // 获取当前时间
    static FileTime Now() {
        auto now = std::chrono::system_clock::now();
        return FileTime(now);
    }

    // 从 time_t 创建
    static FileTime FromTimeT(std::time_t t) {
        return FileTime(static_cast<Timestamp>(t) * 1000000);
    }

    // 比较运算符
    bool operator==(const FileTime& other) const { return microseconds_ == other.microseconds_; }
    bool operator!=(const FileTime& other) const { return microseconds_ != other.microseconds_; }
    bool operator<(const FileTime& other) const { return microseconds_ < other.microseconds_; }
    bool operator<=(const FileTime& other) const { return microseconds_ <= other.microseconds_; }
    bool operator>(const FileTime& other) const { return microseconds_ > other.microseconds_; }
    bool operator>=(const FileTime& other) const { return microseconds_ >= other.microseconds_; }

    // 时间运算
    FileTime operator+(Timestamp microseconds) const {
        return FileTime(microseconds_ + microseconds);
    }

    FileTime operator-(Timestamp microseconds) const {
        return FileTime(microseconds_ - microseconds);
    }

    Timestamp operator-(const FileTime& other) const {
        return microseconds_ - other.microseconds_;
    }

    // 检查是否有效（非零）
    bool IsValid() const { return microseconds_ != 0; }

    // 检查是否为零
    bool IsZero() const { return microseconds_ == 0; }
};

// ============================================================================
// FilePath - 跨平台路径处理
// ============================================================================

#if defined(FR_CPP17) || defined(FR_CPP20)
#include <filesystem>
namespace fs = std::filesystem;
using PathString = fs::path;
#else
// C++14 及以下使用 string
using PathString = std::string;
#endif

// 路径分隔符
#ifdef FR_PLATFORM_WINDOWS
    constexpr char PATH_SEPARATOR = '\\';
    constexpr wchar_t PATH_SEPARATOR_W = L'\\';
#else
    constexpr char PATH_SEPARATOR = '/';
    constexpr wchar_t PATH_SEPARATOR_W = L'/';
#endif

// ============================================================================
// 字节序检测和转换
// ============================================================================

// 检测系统字节序
inline bool IsLittleEndian() {
    const uint16_t value = 0x0001;
    return *reinterpret_cast<const uint8_t*>(&value) == 0x01;
}

inline bool IsBigEndian() {
    return !IsLittleEndian();
}

// 字节序转换（仅在大端系统上需要）
inline uint16_t SwapBytes16(uint16_t value) {
    return (value << 8) | (value >> 8);
}

inline uint32_t SwapBytes32(uint32_t value) {
    return ((value & 0x000000FF) << 24) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0xFF000000) >> 24);
}

inline uint64_t SwapBytes64(uint64_t value) {
    return ((value & 0x00000000000000FFULL) << 56) |
           ((value & 0x000000000000FF00ULL) << 40) |
           ((value & 0x0000000000FF0000ULL) << 24) |
           ((value & 0x00000000FF000000ULL) << 8) |
           ((value & 0x000000FF00000000ULL) >> 8) |
           ((value & 0x0000FF0000000000ULL) >> 24) |
           ((value & 0x00FF000000000000ULL) >> 40) |
           ((value & 0xFF00000000000000ULL) >> 56);
}

// 从小端字节序读取（NTFS 使用小端）
inline uint16_t ReadLE16(const void* ptr) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    return static_cast<uint16_t>(p[0]) |
           (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t ReadLE32(const void* ptr) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline uint64_t ReadLE64(const void* ptr) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    return static_cast<uint64_t>(p[0]) |
           (static_cast<uint64_t>(p[1]) << 8) |
           (static_cast<uint64_t>(p[2]) << 16) |
           (static_cast<uint64_t>(p[3]) << 24) |
           (static_cast<uint64_t>(p[4]) << 32) |
           (static_cast<uint64_t>(p[5]) << 40) |
           (static_cast<uint64_t>(p[6]) << 48) |
           (static_cast<uint64_t>(p[7]) << 56);
}

// ============================================================================
// 内存对齐
// ============================================================================

// 计算对齐后的大小
inline size_t AlignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline size_t AlignDown(size_t value, size_t alignment) {
    return value & ~(alignment - 1);
}

// 检查是否对齐
inline bool IsAligned(size_t value, size_t alignment) {
    return (value & (alignment - 1)) == 0;
}

inline bool IsAligned(const void* ptr, size_t alignment) {
    return IsAligned(reinterpret_cast<uintptr_t>(ptr), alignment);
}

} // namespace Platform
