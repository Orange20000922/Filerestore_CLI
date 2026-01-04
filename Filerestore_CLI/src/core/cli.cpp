#include "cli.h"
#include <iostream>
#include "ImageTable.h"
#include <string>
#include <queue>
#include <vector>
#include <Windows.h>
#include "climodule.h"
#include "cmd.h"
#include <sstream>
#include <algorithm>
#include "Logger.h"
using namespace std;
// 命令解析逻辑
vector<queue<string>> CLI::commands = vector<queue<string>>();
queue<string> CLI::args = queue<string>();
bool CLI::initialized = false;
bool CLI::shouldExit = false;
void CLI::Run(string& command)
{
	//解析命令
	queue<string> theargs = SplitString(command, ' ');
	queue<string> theargsCopy = theargs; // 保存参数副本用于后续处理
	vector<queue<string>> thecommands = GetCommands();
	vector<queue<string>> subcommands = vector<queue<string>>();
    int count1 = 0;
	if (thecommands.empty()) {
		return;
	}
    while (!theargs.empty()) {   
        for (auto currentcommand : thecommands) {
            queue<string> tempcommand = currentcommand;
            if (currentcommand.size() >= 2) {
                for (int i = 0; i < count1; i++) {
                    currentcommand.pop();
                }
            }
            if (currentcommand.front().compare(theargs.front()) == 0) {
                subcommands.push_back(tempcommand);
            }
            else {
                if (currentcommand.front().find('|') != string::npos) {
                    if (currentcommand.front().compare("|file") == 0) {
                        HANDLE hFile = CreateFileA(theargs.front().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
                        if (hFile != INVALID_HANDLE_VALUE) {
                            subcommands.push_back(tempcommand);
                            CloseHandle(hFile);
                        }
                    }
                    if (currentcommand.front().compare("|pid") == 0) {
                        try {
                            int pid = stoi(theargs.front());
                            HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, pid);
                            if (hProcess != NULL) {
                                subcommands.push_back(tempcommand);
                                CloseHandle(hProcess);
                            }
                        }
                        catch (std::invalid_argument&) {
                            cout << "invalid argument for PID" << endl;
                        }
                        catch (std::out_of_range&) {
                            cout << "PID out of range" << endl;
                        }
                    }
                    if (currentcommand.front().compare("|name") == 0) {
                        HANDLE hFile = CreateFileA(theargs.front().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
                        if (hFile==INVALID_HANDLE_VALUE) {
                            subcommands.push_back(tempcommand);
                        }
                        else {
                            CloseHandle(hFile);
                        }
                    }
                    if (currentcommand.front().compare("|privilege") == 0) {
                        if (theargs.front().compare("admin") == 0 || theargs.front().compare("user") == 0) {
                            subcommands.push_back(tempcommand);
                        }
                    }
                    if (currentcommand.front().compare("|processprivilege")==0) {
                        if (theargs.front().compare("SeBackupPrivilege") == 0
                            ||theargs.front().compare("SeRestorePrivilege")==0
                            ||theargs.front().compare("SeShutdownPrivilege")==0
                            ||theargs.front().compare("SeDebugPrivilege")==0) {
							subcommands.push_back(tempcommand);
                        }
                    }
                }
            }
        }
          if (thecommands.size()!=1) {
              thecommands.clear();
              for (queue<string> command : subcommands) {
                  thecommands.push_back(command);
              }
              subcommands.clear();
          }
		theargs.pop();
        count1++;
    }
    if (thecommands.size()==1) {
		string currectcommandname = string();
        while (!thecommands[0].empty()) {
            if (!currectcommandname.empty()) {
                currectcommandname += " ";
            }
            currectcommandname += thecommands[0].front();
            if (thecommands[0].front().find('|') != string::npos && !theargsCopy.empty()) {
                if (thecommands[0].front().compare("|file") == 0) {
                    argsinstances.push_back((LPVOID)new string(theargsCopy.front()));
                }
                if (thecommands[0].front().compare("|pid") == 0) {
                    argsinstances.push_back((LPVOID)new string(theargsCopy.front()));
                }
                if (thecommands[0].front().compare("|name") == 0) {
                    argsinstances.push_back((LPVOID)new string(theargsCopy.front()));
                }
                if (thecommands[0].front().compare("|privilege") == 0) {
					argsinstances.push_back((LPVOID)new string(theargsCopy.front()));
                }
                if (thecommands[0].front().compare("|processprivilege") == 0) {
					argsinstances.push_back((LPVOID)new string(theargsCopy.front()));
                }
            }
            if (!theargsCopy.empty()) {
                theargsCopy.pop();
            }
            thecommands[0].pop();
        }  
		CLIModule* climodule = new CLIModule();
		LPVOID commandclassptr = climodule->GetModuleClassPtrByName(currectcommandname);
        if (commandclassptr == nullptr) {
			cout << "Command class pointer is null." << endl;
            delete climodule;
            for (LPVOID ptr : argsinstances) {
                delete (string*)ptr;
            }
            argsinstances.clear();
            return;
        }
        // 根据命令名称执行相应的命令
            if (PrintAllCommand::CheckName(currectcommandname)&&climodule->GetModuleFlagByName(currectcommandname)) {
				PrintAllCommand* printallcommand = (PrintAllCommand*)commandclassptr;
                printallcommand->Execute(currectcommandname);
            }
            if (HelpCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
                HelpCommand* helpcommand = (HelpCommand*)commandclassptr;
				helpcommand->AcceptArgs(argsinstances);
                helpcommand->Execute(currectcommandname);
            }
            if (QueueDLLsCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
				QueueDLLsCommand* queuedllscommand = (QueueDLLsCommand*)commandclassptr;
                queuedllscommand->AcceptArgs(argsinstances);
                queuedllscommand->Execute(currectcommandname);
            }
            if (GetProcessFuncAddressCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
                GetProcessFuncAddressCommand* getprocessfuncaddresscommand = (GetProcessFuncAddressCommand*)commandclassptr;
                getprocessfuncaddresscommand->AcceptArgs(argsinstances);
                getprocessfuncaddresscommand->Execute(currectcommandname);
            }
            if (ExitCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
				ExitCommand* exitcommand = (ExitCommand*)commandclassptr;
				exitcommand->Execute(currectcommandname);
            }
            if (PrintAllFunction::CheckName(currectcommandname)&&climodule->GetModuleFlagByName(currectcommandname)) {
                PrintAllFunction* printfunccommand = (PrintAllFunction*)commandclassptr;
                printfunccommand->AcceptArgs(argsinstances);
                printfunccommand->Execute(currectcommandname);
            }
           
            if (ListDeletedFilesCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
				ListDeletedFilesCommand* listdeletedcommand = (ListDeletedFilesCommand*)commandclassptr;
				listdeletedcommand->AcceptArgs(argsinstances);
				listdeletedcommand->Execute(currectcommandname);
            }
            if (RestoreByRecordCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
				RestoreByRecordCommand* restorebyrecordcommand = (RestoreByRecordCommand*)commandclassptr;
				restorebyrecordcommand->AcceptArgs(argsinstances);
				restorebyrecordcommand->Execute(currectcommandname);
            }
            if (DiagnoseMFTCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
				DiagnoseMFTCommand* diagnosemftcommand = (DiagnoseMFTCommand*)commandclassptr;
				diagnosemftcommand->AcceptArgs(argsinstances);
				diagnosemftcommand->Execute(currectcommandname);
            }
            if (DetectOverwriteCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
				DetectOverwriteCommand* detectoverwritecommand = (DetectOverwriteCommand*)commandclassptr;
				detectoverwritecommand->AcceptArgs(argsinstances);
				detectoverwritecommand->Execute(currectcommandname);
            }
            if (SearchDeletedFilesCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
				SearchDeletedFilesCommand* searchdeletedcommand = (SearchDeletedFilesCommand*)commandclassptr;
				searchdeletedcommand->AcceptArgs(argsinstances);
				searchdeletedcommand->Execute(currectcommandname);
            }
            if (ScanUsnCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
				ScanUsnCommand* scanusncommand = (ScanUsnCommand*)commandclassptr;
				scanusncommand->AcceptArgs(argsinstances);
				scanusncommand->Execute(currectcommandname);
            }
            if (DiagnoseFileCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
				DiagnoseFileCommand* diagnosefilecommand = (DiagnoseFileCommand*)commandclassptr;
				diagnosefilecommand->AcceptArgs(argsinstances);
				diagnosefilecommand->Execute(currectcommandname);
            }
			const std::unordered_map<string, CLIModule::DynamicLoadedModulePtr>& dynamicmodules = CLIModule::GetDynamicLoadedModules();
            try {
                CLIModule::DynamicLoadedModulePtr DLmodule = dynamicmodules.at(currectcommandname);
                if (DLmodule == nullptr) {
                    LOG_WARNING_FMT("Dynamic module '%s' is null.", currectcommandname.c_str());
                }
                else {
					// 类型转换
					DLAcceptArgs dlacceptargsptr = (DLAcceptArgs)(DLmodule->DLAcceptArgsPtr);
					DLExecute dlexecuteptr = (DLExecute)(DLmodule->DLExecutePtr);
					DLCheckName dlchecknameptr = (DLCheckName)(DLmodule->DLCheckNamePtr);
					DLHasArgs dlhasargsptr = (DLHasArgs)(DLmodule->DLHasArgsPtr);
					DLGetModuleFlag dlgetmoduleflagptr = (DLGetModuleFlag)(DLmodule->DLGetModuleFlagPtr);

					// 验证所有必需的函数指针
					if (dlchecknameptr == nullptr || dlexecuteptr == nullptr || dlgetmoduleflagptr == nullptr) {
						LOG_ERROR_FMT("Dynamic module '%s' missing required functions.", currectcommandname.c_str());
					}
					else if (dlchecknameptr(currectcommandname) && dlgetmoduleflagptr()) {
						if (dlhasargsptr != nullptr && dlhasargsptr()) {
							if (dlacceptargsptr != nullptr) {
								dlacceptargsptr(argsinstances);
							}
						}
						dlexecuteptr(currectcommandname);
					}
                }
			}
            catch (const out_of_range&) {
				// 动态模块未找到，这是正常情况
            }
            catch (const exception& e) {
				LOG_ERROR_FMT("Exception in dynamic module execution: %s", e.what());
            }
        delete climodule;  // 释放 CLIModule 对象
    }
    else {
		cout << "Command not found or ambiguous command." << endl;
    }
    cout <<"size:" <<argsinstances.size() << endl;
    // 释放 argsinstances 中的 string 指针
    for (LPVOID ptr : argsinstances) {
        delete (string*)ptr;
    }
	argsinstances.clear();
}
queue<string> CLI::SplitString(string& str, char delimiter)
{
	queue<string> theargs = queue<string>();
    int count = 0;
    while(str.find(delimiter,count)!=string::npos){
		int index = str.find(delimiter, count);
        int tokenLength = index - count;
        if (tokenLength>0) {
            string token = string();
            token = str.substr(count, tokenLength);
            count = str.find(delimiter, count) + 1;
            theargs.push(string(token));
        }
        else {
            break;
        }
    }
	theargs.push(string(str.substr(count, str.length() - count)));
    return theargs;
}
 void CLI::ParseCommands(string& thecommand,LPVOID instanceptr)
{
	queue<string> cmdQueue = SplitString(thecommand, ' ');
	commands.push_back(cmdQueue);
	CLIModule::RegisterModule(thecommand, (LPVOID)instanceptr, TRUE);
}
 void CLI::ParseDynamicCommands(string modulename)
 {
     const std::unordered_map<string, CLIModule::DynamicLoadedModulePtr>& dynamicmodules = CLIModule::GetDynamicLoadedModules();
     auto it = dynamicmodules.find(modulename);
     if (it == dynamicmodules.end() || it->second == nullptr) {
		 LOG_ERROR_FMT("Dynamic module %s not found.", modulename.c_str());
         return;
	 }
	 commands.push_back(SplitString(modulename, ' '));
 }
 void CLI::ParseDynamicCommands()
 {
     const std::unordered_map<string, CLIModule::DynamicLoadedModulePtr>& dynamicmodules = CLIModule::GetDynamicLoadedModules();
     for (const pair<string, CLIModule::DynamicLoadedModulePtr>& pair : dynamicmodules) {
         CLIModule::DynamicLoadedModulePtr dlmoduleptr = pair.second;
         if (dlmoduleptr == nullptr) {
             LOG_ERROR_FMT("Dynamic module %s not found.", pair.first.c_str());
             continue;
         }
         commands.push_back(SplitString(const_cast<string&>(pair.first), ' '));
	 }
 }
 
 CLI::CLI()
 {
     if (!initialized) {
         ParseCommands(PrintAllCommand::name,PrintAllCommand::GetInstancePtr());
         ParseCommands(HelpCommand::name, HelpCommand::GetInstancePtr());
         ParseCommands(QueueDLLsCommand::name, QueueDLLsCommand::GetInstancePtr());
         ParseCommands(GetProcessFuncAddressCommand::name, GetProcessFuncAddressCommand::GetInstancePtr());
         ParseCommands(ExitCommand::name, ExitCommand::GetInstancePtr());
         ParseCommands(PrintAllFunction::name,PrintAllFunction::GetInstancePtr());
         ParseCommands(ListDeletedFilesCommand::name, ListDeletedFilesCommand::GetInstancePtr());
         ParseCommands(RestoreByRecordCommand::name, RestoreByRecordCommand::GetInstancePtr());
         ParseCommands(DiagnoseMFTCommand::name, DiagnoseMFTCommand::GetInstancePtr());
         ParseCommands(DetectOverwriteCommand::name, DetectOverwriteCommand::GetInstancePtr());
         ParseCommands(SearchDeletedFilesCommand::name, SearchDeletedFilesCommand::GetInstancePtr());
         ParseCommands(ScanUsnCommand::name, ScanUsnCommand::GetInstancePtr());
         ParseCommands(DiagnoseFileCommand::name, DiagnoseFileCommand::GetInstancePtr());
         ParseCommands(SearchUsnCommand::name, SearchUsnCommand::GetInstancePtr());
         ParseCommands(FilterSizeCommand::name, FilterSizeCommand::GetInstancePtr());
         ParseCommands(FindRecordCommand::name, FindRecordCommand::GetInstancePtr());
         ParseCommands(FindUserFilesCommand::name, FindUserFilesCommand::GetInstancePtr());
         ParseCommands(BatchRestoreCommand::name, BatchRestoreCommand::GetInstancePtr());
         ParseCommands(SetLanguageCommand::name, SetLanguageCommand::GetInstancePtr());
         initialized = true;
     }
 }
 CLI::~CLI()
 {
	 CLIModule::UnregisterModules();
     delete analyzer;
 }
