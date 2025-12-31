#include "climodule.h"
#include <Windows.h>
#include <iostream>
#include "cli.h"
using namespace std;
// 静态成员变量定义
CLIModule::ModuleClassPtr CLIModule::ModulePtr = nullptr;
vector<CLIModule::ModuleClassPtr> CLIModule::moduleclasspointers = vector<CLIModule::ModuleClassPtr>();

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
	CLIModule::moduleclasspointers.clear();
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
