// ConsoleApplication5.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

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
using namespace std;
string PrintAllCommand::name = "printallcommand -list";
string HelpCommand::name = "help |name";
string QueueDLLsCommand::name = "queuedllsname |file";
string GetProcessFuncAddressCommand::name = "getfuncaddr |file |name";
string ExitCommand::name = "exit";
string PrintAllFunction::name = "printallfunc |file";
string ListDeletedFilesCommand::name = "listdeleted |name |name";
string RestoreByRecordCommand::name = "restorebyrecord |name |name |file";
string DiagnoseMFTCommand::name = "diagnosemft |name";
string DetectOverwriteCommand::name = "detectoverwrite |name |name |name";
string SearchDeletedFilesCommand::name = "searchdeleted |name |name |name |name";
string DiagnoseFileCommand::name = "diagnosefile |name |name";
string SearchUsnCommand::name = "searchusn |name |name |name";
string FilterSizeCommand::name = "filtersize |name |name |name |name";
string FindRecordCommand::name = "findrecord |name |file";
string FindUserFilesCommand::name = "finduserfiles |name |name";
string BatchRestoreCommand::name = "batchrestore |name |name |file";
string SetLanguageCommand::name = "setlang |name";
string ScanUsnCommand::name = "scanusn |name |name";
int main()
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

	// ========== 清理 ==========

	LOG_INFO("Application shutting down...");
	logger.Close();
	CrashHandler::Uninstall();

	return 0;
}
