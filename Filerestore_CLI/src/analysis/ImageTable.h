#pragma once
#include <map>
#include <vector>
#include <Windows.h>
#include <string>
using namespace std;

// 注意：ZwCreateThreadEx 类型定义已移除 - 只读 PE 分析不需要

class ImageTableAnalyzer
{
public:
	vector<string> dllList = vector<string>();
	map<string, vector<string>> funcList = map<string, vector<string>>();
	ULONGLONG funcAddress = 0;
	LPVOID lpBuffer = NULL;
	HANDLE hFile = NULL;
	HANDLE hFileMapping = NULL;
public:
	// ==================== 只读 PE 分析方法 ====================
	vector<string> AnalyzeTableForDLL(string file);
	map<string, vector<string>> AnalyzeTableForFunctions(string file);
	ULONGLONG GetFuncaddressByName(string name,string file);
	bool IsImagineTable(LPVOID lpBuffer);
	int GetPIDByName(wstring processname);
	DWORD RVAtoFOA(DWORD rva, LPVOID lpBuffer);
	BOOL CheckIsAdmin();

	// ==================== 已移除的方法 ====================
	// 以下方法已从公开版本中移除：
	// - bool IATHooked(string dllfile,int PID);
	// - BOOL ElevatePrivilegeForAdmin(string path);
	// - BOOL ElevatePrivilegeForSystem(wstring Privilege);
	// - BOOL DLLImplantForSystemProcess(string dllfile, int PID);

public:
	ImageTableAnalyzer();
	~ImageTableAnalyzer();
};