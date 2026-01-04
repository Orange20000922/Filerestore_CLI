#include "climodule.h"
#include <Windows.h>
#include <iostream>
#include "cli.h"
#include "Logger.h"
using namespace std;
// 静态成员变量定义
CLIModule::ModuleClassPtr CLIModule::ModulePtr = nullptr;
vector<CLIModule::ModuleClassPtr> CLIModule::moduleclasspointers = vector<CLIModule::ModuleClassPtr>();
std::unordered_map<std::string, CLIModule::DynamicLoadedModulePtr> CLIModule::dynamicloadedmodules = std::unordered_map<std::string, CLIModule::DynamicLoadedModulePtr>();
CLIModule::CLIModule()
{
}
CLIModule::~CLIModule()
{
}
void CLIModule::RegisterModule(string name, LPVOID classptr, BOOL flag)
{
	ModuleClassPtr newModule = new ModuleClass();
	newModule->Name = name;
	newModule->ClassPtr = classptr;
	newModule->Flag = flag;
	moduleclasspointers.push_back(newModule);
}
void CLIModule::UnregisterModules()
{
	// 释放所有动态分配的 ModuleClass 对象
	for (ModuleClassPtr ptr : moduleclasspointers) {
		delete ptr;
	}
	CLIModule::moduleclasspointers.clear();

	// 释放所有动态模块（先卸载 DLL，再清理资源）
	for (pair<string, CLIModule::DynamicLoadedModulePtr> pair : dynamicloadedmodules) {
		if (pair.second != nullptr) {
			// 先卸载 DLL
			if (pair.second->ModuleHandle != NULL) {
				FreeLibrary(pair.second->ModuleHandle);
			}
			// 再清理内存
			delete pair.second->ModulePath;
			delete pair.second;
		}
	}
	dynamicloadedmodules.clear();
}
LPVOID CLIModule::GetModuleClassPtrByName(string name)
{
	for (ModuleClassPtr moduleclassptr:moduleclasspointers) {
		if (moduleclassptr->Name.compare(name)==0) {
			return moduleclassptr->ClassPtr;
		}
	}
	return nullptr;
}
BOOL CLIModule::SetModuleFlagByName(string name, BOOL flag)
{
	for (auto moduleclassptr : moduleclasspointers) {
		if (moduleclassptr->Name.compare(name) == 0) {
			moduleclassptr->Flag = flag;
			return true;
		}
	}
	return false;
}
BOOL CLIModule::GetModuleFlagByName(string name)
{
	for (ModuleClassPtr moduleclassptr : moduleclasspointers) {
		if (moduleclassptr->Name.compare(name) == 0) {
			return moduleclassptr->Flag;
		}
	}
	return false;
}
vector<string> CLIModule::GetAllModuleNames()
{
	vector<string> names = vector<string>();
	for (ModuleClassPtr moduleclassptr : moduleclasspointers) {
		names.push_back(moduleclassptr->Name);
	}
	return names;
}

void CLIModule::LoadDynamicModule(string modulepath)
{
	HMODULE hModule = LoadLibraryA(modulepath.c_str());
	if (hModule == NULL) {
		LOG_ERROR_FMT("Failed to load module: %s (Error: %lu)", modulepath.c_str(), GetLastError());
		return;
	}

	// 填充 DynamicLoadedModule 结构体
	LPVOID dlHasArgsPtr = GetProcAddress(hModule, "HasArgs");
	LPVOID dlCheckNamePtr = GetProcAddress(hModule, "CheckName");
	LPVOID dlGetNamePtr = GetProcAddress(hModule, "GetName");
	LPVOID dlExecutePtr = GetProcAddress(hModule, "Execute");
	LPVOID dlAcceptArgsPtr = GetProcAddress(hModule, "AcceptArgs");
	LPVOID dlGetModuleFlagPtr = GetProcAddress(hModule, "GetModuleFlag");

	// 验证必需的函数是否存在
	if (dlGetNamePtr == NULL || dlCheckNamePtr == NULL || dlExecutePtr == NULL || dlGetModuleFlagPtr == NULL) {
		LOG_ERROR_FMT("Module missing required exports: %s", modulepath.c_str());
		FreeLibrary(hModule);
		return;
	}

	// 获取模块名称
	DLGetName getNameFunc = (DLGetName)dlGetNamePtr;
	string moduleName = getNameFunc();

	// 检查是否已经加载
	if (dynamicloadedmodules.find(moduleName) != dynamicloadedmodules.end()) {
		LOG_WARNING_FMT("Module already loaded: %s", moduleName.c_str());
		FreeLibrary(hModule);
		return;
	}

	// 创建并填充模块信息
	DynamicLoadedModulePtr dlModulePtr = new DynamicLoadedModule();
	dlModulePtr->ModuleHandle = hModule;
	dlModulePtr->DLAcceptArgsPtr = dlAcceptArgsPtr;
	dlModulePtr->DLHasArgsPtr = dlHasArgsPtr;
	dlModulePtr->DLCheckNamePtr = dlCheckNamePtr;
	dlModulePtr->DLGetNamePtr = dlGetNamePtr;
	dlModulePtr->DLExecutePtr = dlExecutePtr;
	dlModulePtr->DLGetModuleFlagPtr = dlGetModuleFlagPtr;
	dlModulePtr->ModulePath = new string(modulepath);

	dynamicloadedmodules[moduleName] = dlModulePtr;
	LOG_INFO_FMT("Loaded dynamic module: %s from %s", moduleName.c_str(), modulepath.c_str());
}
void CLIModule::LoadDynamicModule()
{
	// 此函数可用于加载预定义路径下的所有动态模块
	char modulepath[MAX_PATH];
	GetModuleFileNameA(NULL, modulepath, MAX_PATH);
	string basePath = string(modulepath);
	size_t lastSlash = basePath.find_last_of("\\/");
	if (lastSlash != string::npos) {
		basePath = basePath.substr(0, lastSlash + 1);
	}
	// 假设动态模块存放在 "modules" 子目录下
	string modulesDir = basePath + "modules\\";
	// 这里可以使用文件系统遍历函数来加载该目录下的所有 DLL 文件
	HANDLE hFind;
	WIN32_FIND_DATAA findData;
	hFind = FindFirstFileA((modulesDir + "*.dll").c_str(), &findData);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			string modulePath = modulesDir + findData.cFileName;
			LoadDynamicModule(modulePath);
		} while (FindNextFileA(hFind, &findData));
		FindClose(hFind);
	}

}
