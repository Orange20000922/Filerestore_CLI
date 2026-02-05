// PEAnalysisCommands.cpp - PE文件分析相关命令实现
// 包含: QueueDLLsCommand, GetProcessFuncAddressCommand

#include "cmd.h"
#include <vector>
#include <iostream>
#include "ImageTable.h"

using namespace std;

// ============================================================================
// QueueDLLsCommand - 列出 DLL 依赖
// ============================================================================
DEFINE_COMMAND_BASE(QueueDLLsCommand, "queuedllsname |file", TRUE)
REGISTER_COMMAND(QueueDLLsCommand);

void QueueDLLsCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}
	ImageTableAnalyzer analyzer;
	if (GET_ARG_COUNT() != 1) {
		cout << "Invalid Args!" << endl;
	}
	else {
		string& pefile = GET_ARG_STRING(0);
		vector<string> dlllist = analyzer.AnalyzeTableForDLL(pefile);
		if (dlllist.size() != 0) {
			for (int i = 0; i < dlllist.size(); i++) {
				cout << "    " + dlllist[i] << endl;
			}
		}
		else {
			cout << "can't find the IAT" << endl;
		}
	}
}

// ============================================================================
// GetProcessFuncAddressCommand - 获取函数名及其地址
// ============================================================================
DEFINE_COMMAND_BASE(GetProcessFuncAddressCommand, "getfuncnameaddr |file |name", TRUE)
REGISTER_COMMAND(GetProcessFuncAddressCommand);

void GetProcessFuncAddressCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}
	ImageTableAnalyzer analyzer;
	if (GET_ARG_COUNT() != 2) {
		cout << "Invalid Args!" << endl;
	}
	else {
		string& funcname = GET_ARG_STRING(1);
		string& pefile = GET_ARG_STRING(0);
		ULONGLONG funcaddress = analyzer.GetFuncaddressByName(funcname, pefile);
		if (funcaddress != 0) {
			cout << "Function Address: 0x" << hex << funcaddress << endl;
		}
		else {
			cout << "can't find the function address" << endl;
		}
	}
}
