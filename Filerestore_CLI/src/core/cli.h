#pragma once
#include <string>
#include <vector>
#include <Windows.h>
#include "ImageTable.h"
#include "climodule.h"
#include "CommandMacros.h"

using namespace std;

// 动态模块函数指针定义
typedef string(WINAPI* DLGetName)();
typedef BOOL(WINAPI* DLCheckName)(string);
typedef void(WINAPI* DLExecute)(string);
typedef void(WINAPI* DLAcceptArgs)(vector<LPVOID>);
typedef BOOL(WINAPI* DLHasArgs)();
typedef BOOL(WINAPI* DLGetModuleFlag)();

class CLI
{
private:
    static vector<vector<string>> commands;  // 改用 vector<vector<string>>
    static bool initialized;   // 防止重复注册
    static bool shouldExit;    // 退出标志
    ImageTableAnalyzer* analyzer = new ImageTableAnalyzer();
    vector<LPVOID> argsinstances = vector<LPVOID>();

public:
    void Run(const string& command);
    static vector<string> SplitString(const string& str, char delimiter);
    static void ParseCommands(const string& thecommand, LPVOID instanceptr);
    static void ParseDynamicCommands(const string& modulename);
    static void ParseDynamicCommands();
    static void RegisterFromRegistry();  // 从 CommandRegistry 自动注册命令
    static const vector<vector<string>>& GetCommands() { return commands; }
    static void SetShouldExit(bool value) { shouldExit = value; }
    static bool ShouldExit() { return shouldExit; }

public:
    CLI();
    ~CLI();
};
