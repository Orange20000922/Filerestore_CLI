#include "Logger.h"
#include <ctime>
#include <iomanip>
#include <sstream>

ULONGLONG Logger::logRecordCount = 0;

Logger::Logger() : currentLevel(LOG_INFO), consoleOutput(true), fileOutput(true) {
    InitializeCriticalSection(&csLock);
}

Logger::~Logger() {
    Close();
    DeleteCriticalSection(&csLock);
}

Logger& Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::Initialize(const string& filename, LogLevel level) {
    EnterCriticalSection(&csLock);

    logFilePath = filename;
    currentLevel = level;
    logRecordCount = 0;

    if (fileOutput) {
        // 打开日志文件（追加模式）
        logFile.open(filename, ios::out | ios::app);
        if (logFile.is_open()) {
            logFile << "\n========================================\n";
            logFile << "Log session started at " << GetTimestamp() << "\n";
            logFile << "Log level: " << GetLevelString(level) << "\n";
            logFile << "Max records: " << LOG_RECORD_MAX_SIZE << "\n";
            logFile << "========================================\n";
            logFile.flush();
        }
    }

    LeaveCriticalSection(&csLock);
}

void Logger::Close() {
    EnterCriticalSection(&csLock);

    if (logFile.is_open()) {
        logFile << "========================================\n";
        logFile << "Log session ended at " << GetTimestamp() << "\n";
        logFile << "Total records written: " << logRecordCount << "\n";
        logFile << "========================================\n\n";
        logFile.close();
    }

    LeaveCriticalSection(&csLock);
}

void Logger::SetConsoleOutput(bool enable) {
    consoleOutput = enable;
}

void Logger::SetFileOutput(bool enable) {
    fileOutput = enable;
}

void Logger::SetLogLevel(LogLevel level) {
    currentLevel = level;
}

string Logger::GetTimestamp() {
    SYSTEMTIME st;
    GetLocalTime(&st);

    char buffer[64];
    sprintf_s(buffer, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
        st.wYear, st.wMonth, st.wDay,
        st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);

    return string(buffer);
}

string Logger::GetLevelString(LogLevel level) {
    switch (level) {
    case LOG_DEBUG:   return "[DEBUG]  ";
    case LOG_INFO:    return "[INFO]   ";
    case LOG_WARNING: return "[WARNING]";
    case LOG_ERROR:   return "[ERROR]  ";
    case LOG_FATAL:   return "[FATAL]  ";
    default:          return "[UNKNOWN]";
    }
}

void Logger::RotateLog() {
    // 关闭当前日志文件
    if (logFile.is_open()) {
        logFile << "========================================\n";
        logFile << "Log rotation at " << GetTimestamp() << "\n";
        logFile << "Total records: " << logRecordCount << "\n";
        logFile << "========================================\n\n";
        logFile.close();
    }

    // 生成带时间戳的备份文件名
    SYSTEMTIME st;
    GetLocalTime(&st);
    char timestamp[64];
    sprintf_s(timestamp, "%04d%02d%02d_%02d%02d%02d",
        st.wYear, st.wMonth, st.wDay,
        st.wHour, st.wMinute, st.wSecond);

    // 找到文件扩展名位置
    size_t dotPos = logFilePath.find_last_of('.');
    string baseName = (dotPos != string::npos) ? logFilePath.substr(0, dotPos) : logFilePath;
    string extension = (dotPos != string::npos) ? logFilePath.substr(dotPos) : ".log";

    // 创建备份文件名
    string backupPath = baseName + "_" + timestamp + extension;

    // 重命名当前日志文件为备份文件
    bool renamed = MoveFileA(logFilePath.c_str(), backupPath.c_str());

    // 清理旧的备份文件（保留最近 N 个）
    if (renamed) {
        CleanupOldBackups(baseName, extension);
    }

    // 重新打开日志文件
    logFile.open(logFilePath, ios::out | ios::app);
    if (logFile.is_open()) {
        logFile << "\n========================================\n";
        logFile << "Log rotated at " << GetTimestamp() << "\n";
        logFile << "Previous log saved to: " << backupPath << "\n";
        logFile << "========================================\n";
        logFile.flush();
    }

    // 重置计数器
    logRecordCount = 0;
}

void Logger::CleanupOldBackups(const string& baseName, const string& extension) {
    // 获取日志文件所在目录
    size_t lastSlash = baseName.find_last_of("\\/");
    string directory = (lastSlash != string::npos) ? baseName.substr(0, lastSlash + 1) : ".\\";
    string filePattern = (lastSlash != string::npos) ? baseName.substr(lastSlash + 1) : baseName;

    // 构建搜索模式：例如 "debug_*.log"
    string searchPattern = directory + filePattern + "_*" + extension;

    // 存储备份文件信息（文件名和创建时间）
    vector<pair<string, FILETIME>> backupFiles;

    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(searchPattern.c_str(), &findData);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            // 跳过目录
            if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                continue;
            }

            // 构建完整路径
            string fullPath = directory + findData.cFileName;
            backupFiles.push_back(make_pair(fullPath, findData.ftCreationTime));

        } while (FindNextFileA(hFind, &findData));

        FindClose(hFind);
    }

    // 如果备份文件数量超过限制，删除最旧的
    if (backupFiles.size() > LOG_ROTATION_KEEP_COUNT) {
        // 按创建时间排序（从旧到新）
        sort(backupFiles.begin(), backupFiles.end(),
            [](const pair<string, FILETIME>& a, const pair<string, FILETIME>& b) {
                return CompareFileTime(&a.second, &b.second) < 0;
            });

        // 删除最旧的文件，保留最近的 N 个
        size_t filesToDelete = backupFiles.size() - LOG_ROTATION_KEEP_COUNT;
        for (size_t i = 0; i < filesToDelete; i++) {
            if (DeleteFileA(backupFiles[i].first.c_str())) {
                // 可以记录删除信息，但此时日志文件还未打开
                // 所以我们在重新打开后记录
            }
        }
    }
}

void Logger::Log(LogLevel level, const string& message) {
    // 过滤低于当前级别的日志
    if (level < currentLevel) {
        return;
    }

    EnterCriticalSection(&csLock);

    string timestamp = GetTimestamp();
    string levelStr = GetLevelString(level);
    string logMessage = timestamp + " " + levelStr + " " + message;

    // 输出到文件
    if (fileOutput && logFile.is_open()) {
        // 检查是否需要轮转
        if (logRecordCount >= LOG_RECORD_MAX_SIZE) {
            RotateLog();
        }

        logFile << logMessage << endl;

        // 只对重要日志立即刷新（减少 I/O）
        if (level >= LOG_WARNING) {
            logFile.flush();
        }

        logRecordCount++;
    }

    LeaveCriticalSection(&csLock);
}

void Logger::Log(LogLevel level, const wstring& message) {
    // 将宽字符转换为多字节字符
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, message.c_str(), (int)message.length(), NULL, 0, NULL, NULL);
    string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, message.c_str(), (int)message.length(), &strTo[0], size_needed, NULL, NULL);

    Log(level, strTo);
}

void Logger::Debug(const string& message) {
    Log(LOG_DEBUG, message);
}

void Logger::Info(const string& message) {
    Log(LOG_INFO, message);
}

void Logger::Warning(const string& message) {
    Log(LOG_WARNING, message);
}

void Logger::Error(const string& message) {
    Log(LOG_ERROR, message);
}

void Logger::Fatal(const string& message) {
    Log(LOG_FATAL, message);
}

void Logger::Debug(const wstring& message) {
    Log(LOG_DEBUG, message);
}

void Logger::Info(const wstring& message) {
    Log(LOG_INFO, message);
}

void Logger::Warning(const wstring& message) {
    Log(LOG_WARNING, message);
}

void Logger::Error(const wstring& message) {
    Log(LOG_ERROR, message);
}

void Logger::Fatal(const wstring& message) {
    Log(LOG_FATAL, message);
}
