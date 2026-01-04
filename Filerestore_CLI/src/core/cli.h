#pragma once
#include <string>
#include <queue>
#include <vector>
#include <Windows.h>
#include "ImageTable.h"
#include "climodule.h"
using namespace std;
//动态模块函数指针定义
typedef string(WINAPI* DLGetName)();
typedef BOOL(WINAPI* DLCheckName)(string);
typedef void(WINAPI* DLExecute)(string);
typedef void(WINAPI* DLAcceptArgs)(vector<LPVOID>);
typedef BOOL(WINAPI* DLHasArgs)();
typedef BOOL(WINAPI* DLGetModuleFlag)();
class CLI
{
private:
    static  vector<queue<string>> commands ;
	static  queue<string> args;
	static  bool initialized;  // 防止重复注册
	static  bool shouldExit;  // 退出标志
	ImageTableAnalyzer* analyzer = new ImageTableAnalyzer();
	vector<LPVOID> argsinstances = vector<LPVOID>();
	public:
		void Run(string& command);
		static queue<string> SplitString(string& str, char delimiter);
	    static void ParseCommands(string& thecommmand,LPVOID instanceptr);
		static void ParseDynamicCommands(string modulename);
		static void ParseDynamicCommands();
		static vector<queue<string>> GetCommands() { return commands; };
		static void SetShouldExit(bool value) { shouldExit = value; }
		static bool ShouldExit() { return shouldExit; }
public:
	CLI();
	~CLI();
};