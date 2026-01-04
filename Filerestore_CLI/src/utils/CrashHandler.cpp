#include "CrashHandler.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>

#pragma comment(lib, "Dbghelp.lib")

using namespace std;

LPTOP_LEVEL_EXCEPTION_FILTER CrashHandler::previousFilter = nullptr;

void CrashHandler::Install() {
    previousFilter = SetUnhandledExceptionFilter(ExceptionFilter);
    cout << "Crash handler installed. Dump files will be created on crash." << endl;
}

void CrashHandler::Uninstall() {
    if (previousFilter != nullptr) {
        SetUnhandledExceptionFilter(previousFilter);
        previousFilter = nullptr;
    }
}

LONG WINAPI CrashHandler::ExceptionFilter(EXCEPTION_POINTERS* exceptionPointers) {
    cout << "\n\n============================================" << endl;
    cout << "FATAL ERROR: Unhandled exception detected!" << endl;
    cout << "============================================" << endl;

    if (exceptionPointers != nullptr) {
        cout << "Exception Code: 0x" << hex << exceptionPointers->ExceptionRecord->ExceptionCode << dec << endl;
        cout << "Exception Address: 0x" << hex << exceptionPointers->ExceptionRecord->ExceptionAddress << dec << endl;
    }

    cout << "\nCreating crash dump file..." << endl;
    CreateMiniDump(exceptionPointers);

    cout << "\nProgram will now terminate." << endl;
    cout << "Please send the .dmp file to developers for analysis." << endl;
    cout << "============================================\n" << endl;

    // 返回 EXCEPTION_EXECUTE_HANDLER 让程序正常终止
    return EXCEPTION_EXECUTE_HANDLER;
}

void CrashHandler::CreateMiniDump(EXCEPTION_POINTERS* exceptionPointers) {
    // 生成带时间戳的dump文件名
    time_t now = time(nullptr);
    tm timeInfo;
    localtime_s(&timeInfo, &now);

    stringstream filename;
    filename << "crash_"
             << setfill('0')
             << setw(4) << (timeInfo.tm_year + 1900)
             << setw(2) << (timeInfo.tm_mon + 1)
             << setw(2) << timeInfo.tm_mday << "_"
             << setw(2) << timeInfo.tm_hour
             << setw(2) << timeInfo.tm_min
             << setw(2) << timeInfo.tm_sec
             << ".dmp";

    string dumpFile = filename.str();

    // 创建dump文件
    HANDLE hFile = CreateFileA(
        dumpFile.c_str(),
        GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        cout << "ERROR: Failed to create dump file: " << dumpFile << endl;
        return;
    }

    // 设置dump信息
    MINIDUMP_EXCEPTION_INFORMATION dumpInfo;
    dumpInfo.ThreadId = GetCurrentThreadId();
    dumpInfo.ExceptionPointers = exceptionPointers;
    dumpInfo.ClientPointers = FALSE;

    // 写入minidump
    BOOL success = MiniDumpWriteDump(
        GetCurrentProcess(),
        GetCurrentProcessId(),
        hFile,
        MiniDumpWithFullMemory,  // 包含完整内存，便于调试
        exceptionPointers ? &dumpInfo : NULL,
        NULL,
        NULL
    );

    CloseHandle(hFile);

    if (success) {
        cout << "Crash dump created: " << dumpFile << endl;
        cout << "File size: ";

        // 显示文件大小
        WIN32_FILE_ATTRIBUTE_DATA fileInfo;
        if (GetFileAttributesExA(dumpFile.c_str(), GetFileExInfoStandard, &fileInfo)) {
            ULONGLONG fileSize = ((ULONGLONG)fileInfo.nFileSizeHigh << 32) | fileInfo.nFileSizeLow;
            cout << (fileSize / (1024 * 1024)) << " MB" << endl;
        } else {
            cout << "Unknown" << endl;
        }
    } else {
        cout << "ERROR: Failed to write dump file. Error: " << GetLastError() << endl;
    }
}
