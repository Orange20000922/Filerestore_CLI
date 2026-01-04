#pragma once
#include <string>
#include <vector>
#include <Windows.h>
#include <queue>
#include <unordered_map>
using namespace std;
typedef string(WINAPI* DLGetName)();
class CLIModule
{
public:
	typedef struct ModuleClass{
		BOOL Flag;
		LPVOID ClassPtr;
		string Name;
	}*ModuleClassPtr,ModuleClass;
	typedef struct DynamicLoadedModule{
		HMODULE ModuleHandle;
		LPVOID DLHasArgsPtr;
		LPVOID DLCheckNamePtr;
		LPVOID DLGetNamePtr;
		LPVOID DLExecutePtr;
		LPVOID DLAcceptArgsPtr;
		LPVOID DLGetModuleFlagPtr;
		string* ModulePath;
	}*DynamicLoadedModulePtr, DynamicLoadedModule;
private:
	static ModuleClassPtr ModulePtr;
	static vector<ModuleClassPtr> moduleclasspointers;
	static std::unordered_map<std::string, DynamicLoadedModulePtr> dynamicloadedmodules;
public :
	CLIModule();
	~CLIModule();
	static void RegisterModule(string name, LPVOID classptr,BOOL flag);
	static void UnregisterModules();
	static const std::unordered_map<std::string, DynamicLoadedModulePtr>& GetDynamicLoadedModules() {
		return dynamicloadedmodules;
	};
	LPVOID GetModuleClassPtrByName(string name);
	BOOL SetModuleFlagByName(string name, BOOL flag);
	BOOL GetModuleFlagByName(string name);
	vector<string> GetAllModuleNames();
	void LoadDynamicModule(string modulepath);
	void LoadDynamicModule();
	void UnloadDynamicModules();
};