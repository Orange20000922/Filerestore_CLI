#include <iostream>
#include <Windows.h>
#include <string>
#include <vector>
#include <sstream>
#include "ImageTable.h"
#include "cli.h"
#include "cmd.h"
#include "Logger.h"
#include "CrashHandler.h"
#include "LocalizationManager.h"
#include "TuiApp.h"
using namespace std;

// ============================================================================
// LogStreambuf - 将 cout 输出重定向到日志系统
// 用于 --test 模式，使测试脚本可以通过 debug.log 获取命令输出
// ============================================================================
class LogStreambuf : public std::streambuf {
private:
	std::string lineBuffer;

protected:
	int overflow(int c) override {
		if (c == '\n' || c == EOF) {
			if (!lineBuffer.empty()) {
				Logger::GetInstance().Log(LOG_INFO, "[OUTPUT] " + lineBuffer);
				lineBuffer.clear();
			}
		}
		else {
			lineBuffer += static_cast<char>(c);
		}
		return c;
	}

	int sync() override {
		if (!lineBuffer.empty()) {
			Logger::GetInstance().Log(LOG_INFO, "[OUTPUT] " + lineBuffer);
			lineBuffer.clear();
		}
		return 0;
	}
};

// WLogStreambuf - 将 wcout 输出重定向到日志系统
class WLogStreambuf : public std::wstreambuf {
private:
	std::wstring lineBuffer;

	// 简单的宽字符转窄字符（UTF-16 → UTF-8 近似）
	static std::string WideToNarrow(const std::wstring& wide) {
		if (wide.empty()) return "";
		int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int)wide.size(), NULL, 0, NULL, NULL);
		std::string result(size, 0);
		WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int)wide.size(), &result[0], size, NULL, NULL);
		return result;
	}

protected:
	int_type overflow(int_type c) override {
		if (c == L'\n' || c == WEOF) {
			if (!lineBuffer.empty()) {
				Logger::GetInstance().Log(LOG_INFO, "[OUTPUT] " + WideToNarrow(lineBuffer));
				lineBuffer.clear();
			}
		}
		else {
			lineBuffer += static_cast<wchar_t>(c);
		}
		return c;
	}

	int sync() override {
		if (!lineBuffer.empty()) {
			Logger::GetInstance().Log(LOG_INFO, "[OUTPUT] " + WideToNarrow(lineBuffer));
			lineBuffer.clear();
		}
		return 0;
	}
};
// 所有命令的静态成员现在通过 DEFINE_COMMAND_BASE 宏在 cmd.cpp 中定义
// 并通过 REGISTER_COMMAND 宏自动注册到 CommandRegistry
int main(int argc, char* argv[])
{
	// ========== 系统初始化 ==========

	// 1. 设置控制台输出编码为 UTF-8（支持中文显示）
	SetConsoleOutputCP(CP_UTF8);

	// 2. 初始化崩溃处理器（生成 minidump 用于调试）
	CrashHandler::Install();
	cout << "Crash handler initialized." << endl;

	// 3. 初始化日志系统
	Logger& logger = Logger::GetInstance();

	// 获取程序所在目录的绝对路径
	char exePath[MAX_PATH];
	GetModuleFileNameA(NULL, exePath, MAX_PATH);
	string exeDir = string(exePath);
	size_t lastSlash = exeDir.find_last_of("\\/");
	if (lastSlash != string::npos) {
		exeDir = exeDir.substr(0, lastSlash + 1);
	}
	string logPath = exeDir + "debug.log";

	logger.Initialize(logPath, LOG_INFO);
	logger.SetConsoleOutput(false);  // 关闭控制台输出（避免干扰CLI）
	logger.SetFileOutput(true);      // 启用文件输出
	cout << "Logger initialized: " << logPath << endl;

	// 4. 初始化多语言系统
	LocalizationManager& locMgr = LocalizationManager::Instance();
	// 默认使用中文，可通过setlang命令切换
	locMgr.SetLanguage(L"en");

	LOG_INFO("==============================================");
	LOG_INFO("File Recovery Tool Started");
	LOG_INFO("Version: 0.1.1");
	LOG_INFO("==============================================");

	// ========== CLI 初始化 ==========

	CLI cli = CLI();

	// ========== 检查启动模式 ==========

	bool useTui = false;
	std::string cmdArg;
	bool hasCmd = false;
	bool testMode = false;

	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--tui" || arg == "-t") {
			useTui = true;
		}
		else if (arg == "--test") {
			testMode = true;
		}
		else if (arg == "--cmd" || arg == "-c") {
			// 下一个参数是命令字符串
			if (i + 1 < argc) {
				cmdArg = argv[i + 1];
				hasCmd = true;
				i++; // 跳过命令参数
			}
			else {
				cout << "Error: --cmd requires a command string argument" << endl;
				cout << "Usage: Filerestore_CLI.exe --cmd \"listdeleted D:\"" << endl;
				return 1;
			}
		}
	}

	// --cmd 模式优先级最高（直接执行命令并退出）
	if (hasCmd) {
		cout << "Executing command: " << cmdArg << endl;
		LOG_INFO_FMT("Direct command mode: %s", cmdArg.c_str());

		// --test 模式：重定向 cout/wcout 到日志
		LogStreambuf* logBuf = nullptr;
		WLogStreambuf* wlogBuf = nullptr;
		std::streambuf* origCout = nullptr;
		std::wstreambuf* origWcout = nullptr;

		if (testMode) {
			LOG_INFO("Test mode enabled: redirecting console output to log");
			logBuf = new LogStreambuf();
			wlogBuf = new WLogStreambuf();
			origCout = cout.rdbuf(logBuf);
			origWcout = wcout.rdbuf(wlogBuf);
		}

		try {
			cli.Run(cmdArg);
		}
		catch (const exception& e) {
			if (testMode && origCout) {
				cout.rdbuf(origCout);
				wcout.rdbuf(origWcout);
			}
			cout << "Error executing command: " << e.what() << endl;
			LOG_ERROR_FMT("Exception in command execution: %s", e.what());
			delete logBuf;
			delete wlogBuf;
			logger.Close();
			CrashHandler::Uninstall();
			return 1;
		}
		catch (...) {
			if (testMode && origCout) {
				cout.rdbuf(origCout);
				wcout.rdbuf(origWcout);
			}
			cout << "Unknown error occurred." << endl;
			LOG_ERROR("Unknown exception in command execution");
			delete logBuf;
			delete wlogBuf;
			logger.Close();
			CrashHandler::Uninstall();
			return 1;
		}

		// 恢复原始 streambuf
		if (testMode && origCout) {
			cout.rdbuf(origCout);
			wcout.rdbuf(origWcout);
			delete logBuf;
			delete wlogBuf;
			cout << "Test mode: output written to " << logPath << endl;
		}

		// 命令执行完成，退出
		LOG_INFO("Command execution completed.");
		logger.Close();
		CrashHandler::Uninstall();
		return 0;
	}

	if (useTui) {
		// TUI 模式
		LOG_INFO("Starting in TUI mode...");
		TuiApp app;
		app.Run();
	} else {
		// 传统 CLI 模式（原有逻辑不变）
		cout << endl;
		cout << "==============================================\n";
		cout << "  NTFS File Recovery Tool (Public Version)\n";
		cout << "  File Recovery & Analysis Tool\n";
		cout << "  Version: 0.1.1\n";
		cout << "==============================================\n";
		cout << endl;
		cout << "输入 'help' 获取帮助信息" << endl;
		cout << "Type 'help' for command list" << endl;
		cout << endl;

		// ========== 命令循环 ==========

		while (!CLI::ShouldExit()) {
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
	}

	// ========== 清理 ==========

	LOG_INFO("Application shutting down...");
	logger.Close();
	CrashHandler::Uninstall();

	return 0;
}
