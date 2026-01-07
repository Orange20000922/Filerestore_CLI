#pragma once

// ============================================================================
// 平台检测和配置
// ============================================================================

// 平台检测
#if defined(_WIN32) || defined(_WIN64)
    #define FR_PLATFORM_WINDOWS 1
    #define FR_PLATFORM_NAME "Windows"
#elif defined(__linux__)
    #define FR_PLATFORM_LINUX 1
    #define FR_PLATFORM_NAME "Linux"
#elif defined(__APPLE__) && defined(__MACH__)
    #define FR_PLATFORM_MACOS 1
    #define FR_PLATFORM_NAME "macOS"
#elif defined(__FreeBSD__)
    #define FR_PLATFORM_FREEBSD 1
    #define FR_PLATFORM_NAME "FreeBSD"
#else
    #define FR_PLATFORM_UNKNOWN 1
    #define FR_PLATFORM_NAME "Unknown"
#endif

// 架构检测
#if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
    #define FR_ARCH_X64 1
    #define FR_ARCH_NAME "x64"
#elif defined(_M_IX86) || defined(__i386__)
    #define FR_ARCH_X86 1
    #define FR_ARCH_NAME "x86"
#elif defined(_M_ARM64) || defined(__aarch64__)
    #define FR_ARCH_ARM64 1
    #define FR_ARCH_NAME "ARM64"
#elif defined(_M_ARM) || defined(__arm__)
    #define FR_ARCH_ARM 1
    #define FR_ARCH_NAME "ARM"
#else
    #define FR_ARCH_UNKNOWN 1
    #define FR_ARCH_NAME "Unknown"
#endif

// 文件系统支持
#ifdef FR_PLATFORM_WINDOWS
    #define FR_SUPPORT_NTFS 1
#endif

#ifdef FR_PLATFORM_LINUX
    #define FR_SUPPORT_EXT4 1
#endif

// 编译器检测
#if defined(_MSC_VER)
    #define FR_COMPILER_MSVC 1
    #define FR_COMPILER_NAME "MSVC"
    #define FR_COMPILER_VERSION _MSC_VER
#elif defined(__clang__)
    #define FR_COMPILER_CLANG 1
    #define FR_COMPILER_NAME "Clang"
    #define FR_COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__)
    #define FR_COMPILER_GCC 1
    #define FR_COMPILER_NAME "GCC"
    #define FR_COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#else
    #define FR_COMPILER_UNKNOWN 1
    #define FR_COMPILER_NAME "Unknown"
    #define FR_COMPILER_VERSION 0
#endif

// C++ 标准检测
#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
    #define FR_CPP20 1
#elif __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
    #define FR_CPP17 1
#elif __cplusplus >= 201402L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201402L)
    #define FR_CPP14 1
#elif __cplusplus >= 201103L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)
    #define FR_CPP11 1
#endif

// ============================================================================
// 平台特定包含
// ============================================================================

#ifdef FR_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

// ============================================================================
// 跨平台类型定义
// ============================================================================

#include <cstdint>
#include <cstddef>

namespace Platform {

// 固定大小的整数类型
using Int8 = std::int8_t;
using Int16 = std::int16_t;
using Int32 = std::int32_t;
using Int64 = std::int64_t;

using UInt8 = std::uint8_t;
using UInt16 = std::uint16_t;
using UInt32 = std::uint32_t;
using UInt64 = std::uint64_t;

// 指针大小的整数
using IntPtr = std::intptr_t;
using UIntPtr = std::uintptr_t;
using SizeType = std::size_t;

// 文件系统相关
using FileOffset = Int64;
using FileSize = UInt64;
using ClusterNumber = UInt64;
using RecordNumber = UInt64;

// 时间相关（微秒，自 Unix 纪元）
using Timestamp = Int64;

// ============================================================================
// 平台信息查询
// ============================================================================

// 获取平台名称
inline const char* GetPlatformName() {
    return FR_PLATFORM_NAME;
}

// 获取架构名称
inline const char* GetArchName() {
    return FR_ARCH_NAME;
}

// 获取编译器名称
inline const char* GetCompilerName() {
    return FR_COMPILER_NAME;
}

// 检查是否为 64 位平台
inline bool Is64Bit() {
#if defined(FR_ARCH_X64) || defined(FR_ARCH_ARM64)
    return true;
#else
    return false;
#endif
}

// 检查是否为 Windows
inline bool IsWindows() {
#ifdef FR_PLATFORM_WINDOWS
    return true;
#else
    return false;
#endif
}

// 检查是否支持 NTFS
inline bool SupportsNTFS() {
#ifdef FR_SUPPORT_NTFS
    return true;
#else
    return false;
#endif
}

} // namespace Platform
