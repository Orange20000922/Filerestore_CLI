#pragma once
#include <map>
#include <vector>
#include <Windows.h>
#include <string>
using namespace std;

// Note: ZwCreateThreadEx typedefs removed - not needed for read-only PE analysis

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
	// ==================== Read-only PE Analysis Methods ====================
	vector<string> AnalyzeTableForDLL(string file);
	map<string, vector<string>> AnalyzeTableForFunctions(string file);
	ULONGLONG GetFuncaddressByName(string name,string file);
	bool IsImagineTable(LPVOID lpBuffer);
	int GetPIDByName(wstring processname);
	DWORD RVAtoFOA(DWORD rva, LPVOID lpBuffer);
	BOOL CheckIsAdmin();

	// ==================== REMOVED METHODS ====================
	// The following methods have been removed from public version:
	// - bool IATHooked(string dllfile,int PID);
	// - BOOL ElevatePrivilegeForAdmin(string path);
	// - BOOL ElevatePrivilegeForSystem(wstring Privilege);
	// - BOOL DLLImplantForSystemProcess(string dllfile, int PID);

public:
	ImageTableAnalyzer();
	~ImageTableAnalyzer();
};