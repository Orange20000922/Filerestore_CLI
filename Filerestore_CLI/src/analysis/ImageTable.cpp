#include "ImageTable.h"
#include <vector>
#include <Windows.h>
#include <iostream>
#include <TlHelp32.h>
#include <string>  
#include <map>
#include <sstream>
#include <ShlObj.h>
using namespace std;
//查找函数VA地址
ULONGLONG ImageTableAnalyzer::GetFuncaddressByName(string name,string file)
{
	ImageTableAnalyzer::hFile = CreateFileA(file.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE) {
		cout << "打开文件失败！" << endl;
		return 0;
	}
	ImageTableAnalyzer::hFileMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	if (hFileMapping == NULL) {
		cout << "创建文件映射失败！" << GetLastError() << endl;
		CloseHandle(hFile);
		return 0;
	}
	ImageTableAnalyzer::lpBuffer = MapViewOfFile(hFileMapping, FILE_MAP_READ, 0, 0, 0);
	if (!lpBuffer) {
		cout << "映射文件视图失败！" << endl;
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return 0;
	}
	if (!IsImagineTable(lpBuffer) ){
		cout << "这不是一个有效的 PE 文件！" << endl;
		UnmapViewOfFile(lpBuffer);
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return 0;
	}
	DWORD peOffset = ((PIMAGE_DOS_HEADER)lpBuffer)->e_lfanew;
	PIMAGE_NT_HEADERS pNtHeaders = (PIMAGE_NT_HEADERS)((DWORD_PTR)lpBuffer + peOffset);
	if (pNtHeaders->FileHeader.Machine == IMAGE_FILE_MACHINE_AMD64 || pNtHeaders->FileHeader.Machine == IMAGE_FILE_MACHINE_IA64) {
		PIMAGE_NT_HEADERS64 pNtHeaders64 = (PIMAGE_NT_HEADERS64)pNtHeaders;
		PIMAGE_OPTIONAL_HEADER64 pOptionalHeader64 = &pNtHeaders64->OptionalHeader;
		DWORD importDirRVA = pOptionalHeader64->DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
		if (importDirRVA == 0) {
			cout << "没有导入目录！" << endl;
			UnmapViewOfFile(lpBuffer);
			CloseHandle(hFileMapping);
			CloseHandle(hFile);
			return 0;
		}
	    PIMAGE_IMPORT_DESCRIPTOR pImportDescriptor = (PIMAGE_IMPORT_DESCRIPTOR)((DWORD_PTR)lpBuffer + RVAtoFOA(importDirRVA,lpBuffer));
		while (pImportDescriptor->Name != 0) {
			auto IAT = (PIMAGE_THUNK_DATA64)((DWORD_PTR)lpBuffer + RVAtoFOA(pImportDescriptor->FirstThunk,lpBuffer));
			while (IAT->u1.Ordinal!=0) {
				if ((IAT->u1.AddressOfData & IMAGE_ORDINAL_FLAG64) == 0) {
					PIMAGE_IMPORT_BY_NAME pImportByName = (PIMAGE_IMPORT_BY_NAME)((DWORD_PTR)lpBuffer + RVAtoFOA((DWORD)IAT->u1.AddressOfData,lpBuffer));
					if (name.compare(string((char*)pImportByName->Name))==0) {
						funcAddress = IAT->u1.Function+pOptionalHeader64->ImageBase;
						return funcAddress;
					}
				}
				IAT++;
			}
			pImportDescriptor++;
		}
	}
	return 0;
}
//遍历导入表中的DLL名称
vector<string> ImageTableAnalyzer::AnalyzeTableForDLL(string file)
{
	ImageTableAnalyzer::hFile = CreateFileA(file.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE) {
		cout << "打开文件失败！" << endl;
		return vector<string>();
	}
	ImageTableAnalyzer::hFileMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	if (hFileMapping == NULL) {
		cout << "创建文件映射失败！" << GetLastError() << endl;
		CloseHandle(hFile);
		return vector<string>();
	}
	ImageTableAnalyzer::lpBuffer = MapViewOfFile(hFileMapping, FILE_MAP_READ, 0, 0, 0);
	if (!lpBuffer) {
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return vector<string>();
	}
	if (!IsImagineTable(lpBuffer)) {
		cout << "这不是一个有效的 PE 文件！" << endl;
		UnmapViewOfFile(lpBuffer);
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return vector<string>();
	}
	DWORD peOffset = ((PIMAGE_DOS_HEADER)lpBuffer)->e_lfanew;
	PIMAGE_NT_HEADERS pNtHeaders = (PIMAGE_NT_HEADERS)((DWORD_PTR)lpBuffer + peOffset);

	DWORD importDirRVA = 0;
	if (pNtHeaders->FileHeader.Machine == IMAGE_FILE_MACHINE_AMD64 || pNtHeaders->FileHeader.Machine == IMAGE_FILE_MACHINE_ARM64) {
		// 64位PE
		PIMAGE_NT_HEADERS64 pNtHeaders64 = (PIMAGE_NT_HEADERS64)pNtHeaders;
		importDirRVA = pNtHeaders64->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
	}
	else if (pNtHeaders->FileHeader.Machine == IMAGE_FILE_MACHINE_I386) {
		// 32位PE
		PIMAGE_NT_HEADERS32 pNtHeaders32 = (PIMAGE_NT_HEADERS32)pNtHeaders;
		importDirRVA = pNtHeaders32->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
	}
	else {
		UnmapViewOfFile(lpBuffer);
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return vector<string>();
	}

	if (importDirRVA == 0) {
		cout << "没有导入目录！" << endl;
		UnmapViewOfFile(lpBuffer);
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return vector<string>();
	}
	PIMAGE_IMPORT_DESCRIPTOR pImportDescriptor = (PIMAGE_IMPORT_DESCRIPTOR)((DWORD_PTR)lpBuffer+RVAtoFOA(importDirRVA, lpBuffer));
	while (pImportDescriptor->Name!=0) {
		char* dllName = (char*)(RVAtoFOA(pImportDescriptor->Name,lpBuffer)+(DWORD_PTR)lpBuffer);
		dllList.push_back(string(dllName));
		pImportDescriptor++;
	}
    return dllList;
}
//遍历导入表中的函数名称
map<string, vector<string>> ImageTableAnalyzer::AnalyzeTableForFunctions(string file)
{
	vector<string> funcNames;
	ImageTableAnalyzer::hFile = CreateFileA(file.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE) {
		cout << "打开文件失败！" << endl;
		return map<string, vector<string>>();
	}
	ImageTableAnalyzer::hFileMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	if (hFileMapping == NULL) {
		cout << "创建文件映射失败！" << GetLastError() << endl;
		CloseHandle(hFile);
		return map<string, vector<string>>();
	}
	lpBuffer = MapViewOfFile(hFileMapping, FILE_MAP_READ, 0, 0, 0);
	if (!lpBuffer) {
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return map<string, vector<string>>();
	}
	if (!IsImagineTable(lpBuffer)) {
		cout << "这不是一个有效的 PE 文件！" << endl;
		UnmapViewOfFile(lpBuffer);
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return map<string, vector<string>>();
	}
	DWORD peOffset = ((PIMAGE_DOS_HEADER)lpBuffer)->e_lfanew;
	PIMAGE_NT_HEADERS pNtHeaders = (PIMAGE_NT_HEADERS)((DWORD_PTR)lpBuffer + peOffset);
	DWORD importDirRVA = 0;
	BOOL is64bit = FALSE;

	if (pNtHeaders->FileHeader.Machine == IMAGE_FILE_MACHINE_AMD64 || pNtHeaders->FileHeader.Machine == IMAGE_FILE_MACHINE_ARM64) {
		// 64位PE
		PIMAGE_NT_HEADERS64 pNtHeaders64 = (PIMAGE_NT_HEADERS64)pNtHeaders;
		importDirRVA = pNtHeaders64->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
		is64bit = TRUE;
	}
	else if (pNtHeaders->FileHeader.Machine == IMAGE_FILE_MACHINE_I386) {
		// 32位PE
		PIMAGE_NT_HEADERS32 pNtHeaders32 = (PIMAGE_NT_HEADERS32)pNtHeaders;
		importDirRVA = pNtHeaders32->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
		is64bit = FALSE;
	}
	else {
		UnmapViewOfFile(lpBuffer);
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return map<string, vector<string>>();
	}

	if (importDirRVA == 0) {
		cout << "没有导入目录！" << endl;
		UnmapViewOfFile(lpBuffer);
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return map<string, vector<string>>();
	}
	PIMAGE_IMPORT_DESCRIPTOR pImportDescriptor = (PIMAGE_IMPORT_DESCRIPTOR)((DWORD_PTR)lpBuffer + RVAtoFOA(importDirRVA,lpBuffer));
	while (pImportDescriptor->Name != 0) {
		char* dllName = (char*)((DWORD_PTR)lpBuffer + RVAtoFOA(pImportDescriptor->Name,lpBuffer));
		funcNames.clear();  
		if (is64bit) {
			auto IAT = (PIMAGE_THUNK_DATA64)((DWORD_PTR)lpBuffer + RVAtoFOA(pImportDescriptor->FirstThunk, lpBuffer));
			while (IAT->u1.Ordinal != 0) {
				if ((IAT->u1.AddressOfData & IMAGE_ORDINAL_FLAG64) == 0) {
					PIMAGE_IMPORT_BY_NAME pImportByName = (PIMAGE_IMPORT_BY_NAME)((DWORD_PTR)lpBuffer + RVAtoFOA((DWORD)IAT->u1.AddressOfData, lpBuffer));
					funcNames.push_back(string((char*)pImportByName->Name));
				}
				else {
					ostringstream oss;
					oss << "Ordinal:" << (IAT->u1.AddressOfData & 0xFFFF);
					funcNames.push_back(oss.str());
				}
				IAT++;
			}
		}
		else {
			auto IAT = (PIMAGE_THUNK_DATA32)((DWORD_PTR)lpBuffer + RVAtoFOA(pImportDescriptor->FirstThunk, lpBuffer));
			while (IAT->u1.Ordinal != 0) {
				if ((IAT->u1.AddressOfData & IMAGE_ORDINAL_FLAG32) == 0) {
					PIMAGE_IMPORT_BY_NAME pImportByName = (PIMAGE_IMPORT_BY_NAME)((DWORD_PTR)lpBuffer + RVAtoFOA(IAT->u1.AddressOfData, lpBuffer));
					funcNames.push_back(string((char*)pImportByName->Name));
				}
				else {
					ostringstream oss;
					oss << "Ordinal:" << (IAT->u1.AddressOfData & 0xFFFF);
					funcNames.push_back(oss.str());
				}
				IAT++;
			}
		}
		funcList[string(dllName)] = funcNames;
		pImportDescriptor++;
	}
	return funcList;
}
bool ImageTableAnalyzer::IsImagineTable(LPVOID lpBuffer)
{
	PIMAGE_DOS_HEADER pDosHeader = (PIMAGE_DOS_HEADER)lpBuffer;
	if (pDosHeader->e_magic!=IMAGE_DOS_SIGNATURE) {
		return false;
	}
	DWORD peOffset = pDosHeader->e_lfanew;
	if (peOffset<sizeof(IMAGE_DOS_HEADER)||peOffset>GetFileSize(hFile,NULL) ){
		return false;
	}
	PIMAGE_NT_HEADERS pNtHeaders = (PIMAGE_NT_HEADERS)((DWORD_PTR)lpBuffer + peOffset);
	if (pNtHeaders->Signature != IMAGE_NT_SIGNATURE) {
		return false;
	}
	return true;
}
//根据进程名称获取PID
int ImageTableAnalyzer::GetPIDByName(wstring processname)
{
	DWORD PID = 0;
	HANDLE hshapshot = CreateToolhelp32Snapshot(TH32CS_SNAPALL, 0);
	if (hshapshot == INVALID_HANDLE_VALUE) {
		return 0;
	}
	PROCESSENTRY32 pe;
	ZeroMemory(&pe, sizeof(PROCESSENTRY32));
	pe.dwSize = sizeof(PROCESSENTRY32);
	if (Process32First(hshapshot, &pe)) {
		do {
			if (processname.compare(pe.szExeFile) == 0) {
				PID = pe.th32ProcessID;
				CloseHandle(hshapshot);
				return PID;
			}
		} while (Process32Next(hshapshot, &pe));
	}
	CloseHandle(hshapshot);
	return 0;
}
//RVA转换为FOA
DWORD ImageTableAnalyzer::RVAtoFOA(DWORD rva, LPVOID lpBuffer)
{
	DWORD peOffset = ((PIMAGE_DOS_HEADER)lpBuffer)->e_lfanew;
	PIMAGE_NT_HEADERS pNtHeaders = (PIMAGE_NT_HEADERS)((DWORD_PTR)lpBuffer + peOffset);
	PIMAGE_SECTION_HEADER pSectionHeader = IMAGE_FIRST_SECTION(pNtHeaders);
	for (int i = 0; i < pNtHeaders->FileHeader.NumberOfSections; i++) {
		DWORD sectionVA = pSectionHeader[i].VirtualAddress;
		DWORD sectionSize = pSectionHeader[i].Misc.VirtualSize;
		if (rva >= sectionVA && rva < sectionVA + sectionSize) {
			DWORD delta = rva - sectionVA;
			DWORD foa = pSectionHeader[i].PointerToRawData + delta;
			return foa;
		}
	}
	return 0;
}
// ==================== ElevatePrivilegeForAdmin 已移除 ====================
// 权限提升漏洞利用已从公开版本中移除
// 原实现：使用 fodhelper.exe 进行 UAC 绕过

// ==================== ElevatePrivilegeForSystem 已移除 ====================
// 权限提升功能已从公开版本中移除
// 原实现：AdjustTokenPrivileges

BOOL ImageTableAnalyzer::CheckIsAdmin()
{
	BOOL isAdmin = FALSE;
	HANDLE hToken = NULL;

	// 获取当前进程令牌
	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
		return FALSE;
	}

	// 方法1: 检查令牌是否已提升（UAC提升后）
	TOKEN_ELEVATION elevation;
	DWORD dwSize = sizeof(TOKEN_ELEVATION);
	if (GetTokenInformation(hToken, TokenElevation, &elevation, sizeof(elevation), &dwSize)) {
		isAdmin = elevation.TokenIsElevated;
	}

	// 如果方法1返回FALSE，使用方法2: 检查是否为管理员组成员
	if (!isAdmin) {
		// 创建管理员组 SID
		BYTE adminSID[SECURITY_MAX_SID_SIZE];
		dwSize = sizeof(adminSID);
		if (CreateWellKnownSid(WinBuiltinAdministratorsSid, NULL, adminSID, &dwSize)) {
			// 检查当前令牌是否属于管理员组
			CheckTokenMembership(NULL, adminSID, &isAdmin);
		}
	}

	CloseHandle(hToken);
	return isAdmin;
}

//针对系统进程的IAT注入
// ==================== DLLImplantForSystemProcess 已移除 ====================
// DLL 注入功能已从公开版本中移除
// 原实现：使用 ZwCreateThreadEx 进行远程 DLL 注入

ImageTableAnalyzer::ImageTableAnalyzer() {
}
ImageTableAnalyzer::~ImageTableAnalyzer() {
	CloseHandle(ImageTableAnalyzer::hFileMapping);
	CloseHandle(ImageTableAnalyzer::hFile);
	UnmapViewOfFile(lpBuffer);
}
