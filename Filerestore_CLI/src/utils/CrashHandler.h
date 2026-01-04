#pragma once
#include <Windows.h>
#include <DbgHelp.h>
#include <string>

// 崩溃处理器 - 生成minidump文件用于调试
class CrashHandler
{
public:
    // 安装崩溃处理器
    static void Install();

    // 卸载崩溃处理器
    static void Uninstall();

private:
    // 异常过滤器
    static LONG WINAPI ExceptionFilter(EXCEPTION_POINTERS* exceptionPointers);

    // 创建minidump文件
    static void CreateMiniDump(EXCEPTION_POINTERS* exceptionPointers);

    static LPTOP_LEVEL_EXCEPTION_FILTER previousFilter;
};
