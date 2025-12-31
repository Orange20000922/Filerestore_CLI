# Main.cpp 初始化代码补充说明

## 修改概述

已在 `Main.cpp` 中补充了完整的系统初始化代码，包括：
- 控制台编码设置
- 崩溃处理器初始化
- 日志系统初始化
- 异常处理机制

## 修改详情

### 1. 添加的头文件引用

```cpp
#include "Logger.h"
#include "CrashHandler.h"
```

### 2. 系统初始化流程

#### 步骤 1: 控制台编码设置
```cpp
SetConsoleOutputCP(CP_UTF8);
```
**目的**: 支持中文等 Unicode 字符的正确显示

#### 步骤 2: 崩溃处理器初始化
```cpp
CrashHandler::Install();
cout << "Crash handler initialized." << endl;
```
**功能**:
- 捕获程序崩溃异常
- 自动生成 minidump 文件用于调试
- 帮助定位和修复程序崩溃问题

**生成的文件**: `crash_YYYY-MM-DD_HH-MM-SS.dmp`

#### 步骤 3: 日志系统初始化
```cpp
Logger& logger = Logger::GetInstance();
logger.Initialize("debug.log", LOG_INFO);
logger.SetConsoleOutput(false);  // 关闭控制台输出
logger.SetFileOutput(true);      // 启用文件输出
cout << "Logger initialized: debug.log" << endl;

LOG_INFO("==============================================");
LOG_INFO("File Recovery Tool Started");
LOG_INFO("Version: 0.1.0");
LOG_INFO("==============================================");
```

**配置说明**:
- **日志文件**: `debug.log`
- **日志级别**: `LOG_INFO`（记录 INFO、WARNING、ERROR、FATAL）
- **控制台输出**: 关闭（避免干扰 CLI 界面）
- **文件输出**: 启用（所有日志写入文件）

**日志级别**:
| 级别 | 说明 | 用途 |
|------|------|------|
| DEBUG | 调试信息 | 详细的内部状态（默认不记录）|
| INFO | 一般信息 | 程序运行状态、用户操作 |
| WARNING | 警告信息 | 可恢复的异常情况 |
| ERROR | 错误信息 | 命令执行失败、操作错误 |
| FATAL | 严重错误 | 导致程序无法继续的错误 |

### 3. 改进的 CLI 界面

```cpp
cout << "==============================================\n";
cout << "  Orange的系统文件分析工具\n";
cout << "  File Recovery & Analysis Tool\n";
cout << "  Version: 0.1.0\n";
cout << "==============================================\n";
cout << endl;
cout << "输入 'help' 获取帮助信息" << endl;
cout << "Type 'help' for command list" << endl;
```

**改进点**:
- 更清晰的界面布局
- 双语提示（中文/英文）
- 版本信息显示

### 4. 异常处理机制

```cpp
while (true) {
    string command = string();
    cout << "Command> ";
    getline(cin, command);

    // 记录命令到日志
    LOG_INFO_FMT("User command: %s", command.c_str());

    try {
        cli.Run(command);
    }
    catch (const exception& e) {
        cout << "Error executing command: " << e.what() << endl;
        LOG_ERROR_FMT("Exception in command execution: %s", e.what());
    }
    catch (...) {
        cout << "Unknown error occurred." << endl;
        LOG_ERROR("Unknown exception in command execution");
    }
}
```

**功能**:
- 捕获所有命令执行异常
- 防止单个命令错误导致程序崩溃
- 记录异常信息到日志文件
- 向用户显示友好的错误提示

### 5. 清理代码

```cpp
LOG_INFO("Application shutting down...");
logger.Close();
CrashHandler::Uninstall();
```

**注意**: 由于命令循环是无限循环，正常情况下不会执行清理代码。只有在某些特殊情况（如 exit 命令）才会到达这里。

## 日志文件管理

### 日志轮转机制

日志系统会自动管理日志文件大小：
- **轮转条件**: 日志记录数达到 50,000 条
- **轮转操作**: 将当前日志重命名为 `debug_YYYY-MM-DD_HH-MM-SS.log.bak`
- **保留策略**: 只保留最近 3 个备份文件，自动删除旧文件

### 日志文件位置

所有日志文件位于程序执行目录：
```
C:\Path\To\Program\
├── ImportTableAnalyzer.exe
├── debug.log                           (当前日志)
├── debug_2025-12-31_10-30-45.log.bak  (备份1)
├── debug_2025-12-31_09-15-22.log.bak  (备份2)
└── debug_2025-12-31_08-00-10.log.bak  (备份3)
```

## 崩溃转储文件

### 生成时机

当程序发生以下异常时自动生成：
- 访问冲突（Access Violation）
- 除零错误
- 栈溢出
- 其他未处理异常

### 文件位置和格式

```
C:\Path\To\Program\
└── crash_2025-12-31_14-30-25.dmp
```

### 使用方法

1. 在 Visual Studio 中打开 .dmp 文件
2. 点击"使用本机代码调试"
3. 查看崩溃时的调用栈和变量值
4. 定位问题所在的代码行

## 运行效果

程序启动时的输出示例：

```
Crash handler installed.
Logger initialized: debug.log

==============================================
  Orange的系统文件分析工具
  File Recovery & Analysis Tool
  Version: 0.1.0
==============================================

输入 'help' 获取帮助信息
Type 'help' for command list

Command>
```

## debug.log 内容示例

```
[2025-12-31 14:30:25] [INFO] ==============================================
[2025-12-31 14:30:25] [INFO] File Recovery Tool Started
[2025-12-31 14:30:25] [INFO] Version: 0.1.0
[2025-12-31 14:30:25] [INFO] ==============================================
[2025-12-31 14:30:32] [INFO] User command: listdeleted C
[2025-12-31 14:30:35] [INFO] User command: searchdeleted C * .xml
[2025-12-31 14:30:40] [ERROR] Exception in command execution: Invalid drive letter
```

## 日志记录策略

### 记录内容

1. **系统事件**:
   - 程序启动/关闭
   - 组件初始化

2. **用户操作**:
   - 所有输入的命令
   - 命令执行结果

3. **错误信息**:
   - 命令执行异常
   - 系统错误
   - 文件操作失败

4. **调试信息**（DEBUG 级别）:
   - 函数调用跟踪
   - 变量值
   - 详细的内部状态

### 性能优化

- **异步写入**: 日志写入不会阻塞主线程
- **批量刷新**: 累积一定数量后统一写入磁盘
- **线程安全**: 使用临界区保护，支持多线程环境

## 故障排查

### 问题 1: 中文显示乱码

**原因**: 控制台编码未设置为 UTF-8

**解决**: 已在代码中添加 `SetConsoleOutputCP(CP_UTF8);`

### 问题 2: 程序崩溃后没有 dump 文件

**可能原因**:
1. 权限不足（以管理员运行）
2. DbgHelp.dll 未找到
3. 磁盘空间不足

**解决**: 检查程序运行目录的写权限

### 问题 3: 日志文件不断增长

**说明**: 这是正常的，日志轮转会自动管理

**手动清理**: 可以安全删除 `.log.bak` 备份文件

## 开发建议

### 调试时启用 DEBUG 级别

```cpp
logger.Initialize("debug.log", LOG_DEBUG);  // 记录详细调试信息
logger.SetConsoleOutput(true);               // 同时在控制台显示
```

### 发布时使用 INFO 级别

```cpp
logger.Initialize("debug.log", LOG_INFO);   // 只记录重要信息
logger.SetConsoleOutput(false);              // 关闭控制台输出
```

### 在代码中使用日志

```cpp
// 简单日志
LOG_INFO("File scan completed");

// 格式化日志
LOG_INFO_FMT("Found %d deleted files", fileCount);
LOG_ERROR_FMT("Failed to open drive %c: Error code %d", driveLetter, GetLastError());

// 条件日志
if (result.isFullyAvailable) {
    LOG_INFO("File data is fully available");
} else {
    LOG_WARNING_FMT("File is %d%% overwritten", (int)result.overwritePercentage);
}
```

## 测试验证

### 测试项目

- [ ] 程序启动显示初始化信息
- [ ] debug.log 文件正确生成
- [ ] 中文字符正确显示
- [ ] 命令执行记录到日志
- [ ] 异常被正确捕获和记录
- [ ] 日志轮转功能正常
- [ ] 崩溃时生成 dump 文件（需要触发崩溃测试）

### 测试命令

```bash
# 1. 正常命令
listdeleted C

# 2. 错误命令（测试异常处理）
listdeleted X

# 3. 多次执行（测试日志记录）
searchdeleted C * .xml
searchdeleted C * .cat
```

### 查看日志

```bash
# Windows
type debug.log

# 或在记事本中打开
notepad debug.log
```

## 总结

Main.cpp 现在具有：
- ✅ 完整的系统初始化
- ✅ 崩溃保护机制
- ✅ 完善的日志系统
- ✅ 异常处理机制
- ✅ 双语界面提示
- ✅ UTF-8 编码支持

这些改进使程序更加健壮、可调试和用户友好。

---
**修改时间**: 2025-12-31
**修改内容**: 补充日志系统和崩溃处理器初始化代码
**影响范围**: Main.cpp
