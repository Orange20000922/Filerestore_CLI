#pragma once
#include <Windows.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// 日志级别
enum LogLevel {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARNING = 2,
    LOG_ERROR = 3,
    LOG_FATAL = 4
};

// 日志记录器类 - 单例模式
class Logger {
private:
	static ULONGLONG logRecordCount;
    ofstream logFile;
    LogLevel currentLevel;
    CRITICAL_SECTION csLock; // 线程安全
    bool consoleOutput;
    bool fileOutput;
    string logFilePath;

    Logger();
    ~Logger();

    // 禁止拷贝
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // 获取当前时间字符串
    string GetTimestamp();

    // 获取日志级别字符串
    string GetLevelString(LogLevel level);

    // 日志轮转
    void RotateLog();

    // 清理旧备份文件
    void CleanupOldBackups(const string& baseName, const string& extension);

public:
    // 获取单例实例
    static Logger& GetInstance();

    // 初始化日志系统
    void Initialize(const string& filename = "debug.log", LogLevel level = LOG_INFO);

    // 关闭日志系统
    void Close();

    // 设置是否输出到控制台
    void SetConsoleOutput(bool enable);

    // 设置是否输出到文件
    void SetFileOutput(bool enable);

    // 设置日志级别
    void SetLogLevel(LogLevel level);

    // 记录日志
    void Log(LogLevel level, const string& message);
    void Log(LogLevel level, const wstring& message);

    // 便捷方法
    void Debug(const string& message);
    void Info(const string& message);
    void Warning(const string& message);
    void Error(const string& message);
    void Fatal(const string& message);

    // 宽字符版本
    void Debug(const wstring& message);
    void Info(const wstring& message);
    void Warning(const wstring& message);
    void Error(const wstring& message);
    void Fatal(const wstring& message);

    // 格式化日志
    template<typename... Args>
    void LogFormat(LogLevel level, const char* format, Args... args) {
        char buffer[1024] ;
		ZeroMemory(buffer, sizeof(buffer));
        sprintf_s(buffer, format, args...);
        Log(level, string(buffer));
    }
};

// 全局便捷宏
#define LOG_DEBUG(msg) Logger::GetInstance().Debug(msg)
#define LOG_INFO(msg) Logger::GetInstance().Info(msg)
#define LOG_WARNING(msg) Logger::GetInstance().Warning(msg)
#define LOG_ERROR(msg) Logger::GetInstance().Error(msg)
#define LOG_FATAL(msg) Logger::GetInstance().Fatal(msg)

// 日志配置常量
#define LOG_RECORD_MAX_SIZE 50000      // 最大日志条数（减少到 5 万条）
#define LOG_ROTATION_KEEP_COUNT 3      // 保留最近 3 个日志文件

// 格式化日志宏
#define LOG_DEBUG_FMT(...) Logger::GetInstance().LogFormat(LOG_DEBUG, __VA_ARGS__)
#define LOG_INFO_FMT(...) Logger::GetInstance().LogFormat(LOG_INFO, __VA_ARGS__)
#define LOG_WARNING_FMT(...) Logger::GetInstance().LogFormat(LOG_WARNING, __VA_ARGS__)
#define LOG_ERROR_FMT(...) Logger::GetInstance().LogFormat(LOG_ERROR, __VA_ARGS__)
#define LOG_FATAL_FMT(...) Logger::GetInstance().LogFormat(LOG_FATAL, __VA_ARGS__)
