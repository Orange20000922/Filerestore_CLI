#include "cli.h"
#include <iostream>
#include "ImageTable.h"
#include <string>
#include <vector>
#include <Windows.h>
#include "climodule.h"
#include "cmd.h"
#include <sstream>
#include <algorithm>
#include "Logger.h"
#include "../tui/TuiProgressTracker.h"

using namespace std;

// 命令解析逻辑
vector<vector<string>> CLI::commands = vector<vector<string>>();
bool CLI::initialized = false;
bool CLI::shouldExit = false;

// ============================================================================
// 迭代筛选命令匹配算法
// 时间复杂度: 平均情况下优于 O(n × m)，因为候选集在每次迭代后缩小
// 空间复杂度: O(n) 用于存储候选索引
// ============================================================================

// 检查单个 token 是否匹配命令模式中对应位置的 token
static bool MatchToken(const string& pattern, const string& arg) {
    // 如果是占位符，进行类型检查
    if (!pattern.empty() && pattern[0] == '|') {
        // |file - 必须是有效的文件路径
        if (pattern == "|file") {
            HANDLE hFile = CreateFileA(arg.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                       NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile != INVALID_HANDLE_VALUE) {
                CloseHandle(hFile);
                return true;
            }
            return false;
        }
        // |pid - 必须是有效的进程ID
        if (pattern == "|pid") {
            try {
                int pid = stoi(arg);
                HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, pid);
                if (hProcess != NULL) {
                    CloseHandle(hProcess);
                    return true;
                }
            } catch (...) {}
            return false;
        }
        // |name - 通用占位符，匹配任意非空字符串
        // 用途：驱动器字母、命令名、搜索模式、扩展名、过滤级别等
        if (pattern == "|name") {
            return !arg.empty();
        }
        // |privilege - 权限级别
        if (pattern == "|privilege") {
            return (arg == "admin" || arg == "user");
        }
        // |processprivilege - 进程权限名称
        if (pattern == "|processprivilege") {
            return (arg == "SeBackupPrivilege" || arg == "SeRestorePrivilege" ||
                    arg == "SeShutdownPrivilege" || arg == "SeDebugPrivilege");
        }
        // 未知占位符类型，默认匹配任意非空字符串
        return !arg.empty();
    }

    // 精确匹配命令名
    return pattern == arg;
}

void CLI::Run(const string& command) {
    // 解析用户输入
    vector<string> args = SplitString(command, ' ');
    if (args.empty()) {
        return;
    }

    const vector<vector<string>>& allCommands = GetCommands();
    if (allCommands.empty()) {
        return;
    }

    // ========== 迭代筛选算法 ==========
    // 使用索引而非复制，避免 queue 的开销
    // 初始化候选集：所有命令的索引
    vector<size_t> candidates;
    candidates.reserve(allCommands.size());
    for (size_t i = 0; i < allCommands.size(); i++) {
        candidates.push_back(i);
    }

    // 逐个 token 迭代筛选，缩小候选集
    vector<size_t> nextCandidates;
    nextCandidates.reserve(allCommands.size());

    for (size_t tokenIdx = 0; tokenIdx < args.size() && candidates.size() > 1; tokenIdx++) {
        const string& userToken = args[tokenIdx];
        nextCandidates.clear();

        for (size_t cmdIdx : candidates) {
            const vector<string>& cmdPattern = allCommands[cmdIdx];

            // 检查命令是否有足够的 token
            if (tokenIdx < cmdPattern.size()) {
                // 检查当前位置的 token 是否匹配
                if (MatchToken(cmdPattern[tokenIdx], userToken)) {
                    nextCandidates.push_back(cmdIdx);
                }
            }
        }

        // 如果筛选后有候选，则更新候选集
        if (!nextCandidates.empty()) {
            swap(candidates, nextCandidates);
        }
        // 如果筛选后无候选，保持原候选集（可能是参数过多的情况）
    }

    // ========== 选择最佳匹配 ==========
    // 从剩余候选中选择最长且完全匹配的命令
    size_t bestMatchIdx = SIZE_MAX;
    size_t bestMatchLength = 0;

    for (size_t cmdIdx : candidates) {
        const vector<string>& cmdPattern = allCommands[cmdIdx];

        // 检查是否完全匹配（用户参数数量 >= 命令模式长度）
        if (args.size() >= cmdPattern.size()) {
            // 验证所有 token 都匹配
            bool allMatch = true;
            for (size_t i = 0; i < cmdPattern.size() && allMatch; i++) {
                if (!MatchToken(cmdPattern[i], args[i])) {
                    allMatch = false;
                }
            }

            if (allMatch && cmdPattern.size() > bestMatchLength) {
                bestMatchIdx = cmdIdx;
                bestMatchLength = cmdPattern.size();
            }
        }
    }

    // 如果没有完全匹配，选择第一个部分匹配
    if (bestMatchIdx == SIZE_MAX && !candidates.empty()) {
        bestMatchIdx = candidates[0];
    }

    if (bestMatchIdx == SIZE_MAX) {
        cout << "Command not found." << endl;
        return;
    }

    // ========== 构建命令名称和收集参数 ==========
    const vector<string>& bestMatch = allCommands[bestMatchIdx];
    string commandName;
    vector<LPVOID> argInstances;

    for (size_t i = 0; i < bestMatch.size(); i++) {
        const string& token = bestMatch[i];

        if (!commandName.empty()) {
            commandName += " ";
        }
        commandName += token;

        // 收集占位符对应的用户参数
        if (!token.empty() && token[0] == '|' && i < args.size()) {
            argInstances.push_back((LPVOID)new string(args[i]));
        }
    }

    // ========== 执行命令 ==========
    CLIModule* climodule = new CLIModule();
    LPVOID commandClassPtr = climodule->GetModuleClassPtrByName(commandName);

    if (commandClassPtr == nullptr) {
        cout << "Command not registered: " << commandName << endl;
        delete climodule;
        for (LPVOID ptr : argInstances) {
            delete (string*)ptr;
        }
        return;
    }

    // ========== TUI 状态更新（在命令执行前）==========
    if (TuiProgressTracker::Instance().IsEnabled()) {
        // 提取驱动器参数（如果有）
        if (!args.empty() && args.size() >= 1) {
            string potentialDrive = args[0];
            // 检查是否是驱动器格式 (C, C:, C:\)
            if (potentialDrive.size() >= 1 && isalpha(potentialDrive[0])) {
                char driveLetter = toupper(potentialDrive[0]);
                TuiProgressTracker::Instance().SetDrive(string(1, driveLetter) + ":");
            }
        }

        // 根据命令类型设置预期状态
        string cmdBase = bestMatch[0]; // 第一个 token（命令基础名）
        if (cmdBase == "listdeleted" || cmdBase == "searchdeleted" || cmdBase == "recover") {
            TuiProgressTracker::Instance().SetMftStatus("Loading...");
            TuiProgressTracker::Instance().SetUsnStatus("Loading...");
        } else if (cmdBase == "carvepool" || cmdBase == "carve") {
            TuiProgressTracker::Instance().SetMftStatus("N/A");
            TuiProgressTracker::Instance().SetUsnStatus("N/A");
        } else if (cmdBase == "usnrecover" || cmdBase == "searchusn") {
            TuiProgressTracker::Instance().SetUsnStatus("Loading...");
        }
    }

    // 使用多态分发执行命令
    if (climodule->GetModuleFlagByName(commandName)) {
        Command* cmd = (Command*)commandClassPtr;
        if (cmd->HasArgs()) {
            cmd->AcceptArgs(argInstances);
        }
        cmd->Execute(commandName);
    }

    // 尝试动态模块
    const auto& dynamicModules = CLIModule::GetDynamicLoadedModules();
    try {
        auto DLmodule = dynamicModules.at(commandName);
        if (DLmodule != nullptr) {
            DLAcceptArgs dlAcceptArgs = (DLAcceptArgs)(DLmodule->DLAcceptArgsPtr);
            DLExecute dlExecute = (DLExecute)(DLmodule->DLExecutePtr);
            DLCheckName dlCheckName = (DLCheckName)(DLmodule->DLCheckNamePtr);
            DLHasArgs dlHasArgs = (DLHasArgs)(DLmodule->DLHasArgsPtr);
            DLGetModuleFlag dlGetModuleFlag = (DLGetModuleFlag)(DLmodule->DLGetModuleFlagPtr);

            if (dlCheckName != nullptr && dlExecute != nullptr && dlGetModuleFlag != nullptr) {
                if (dlCheckName(commandName) && dlGetModuleFlag()) {
                    if (dlHasArgs != nullptr && dlHasArgs() && dlAcceptArgs != nullptr) {
                        dlAcceptArgs(argInstances);
                    }
                    dlExecute(commandName);
                }
            }
        }
    } catch (const out_of_range&) {
        // 动态模块未找到，正常情况
    } catch (const exception& e) {
        LOG_ERROR_FMT("Exception in dynamic module execution: %s", e.what());
    }

    delete climodule;

    // 释放参数实例
    for (LPVOID ptr : argInstances) {
        delete (string*)ptr;
    }
}

vector<string> CLI::SplitString(const string& str, char delimiter) {
    vector<string> tokens;
    size_t start = 0;
    size_t end = 0;

    while ((end = str.find(delimiter, start)) != string::npos) {
        if (end > start) {
            tokens.push_back(str.substr(start, end - start));
        }
        start = end + 1;
    }

    // 添加最后一个 token
    if (start < str.length()) {
        tokens.push_back(str.substr(start));
    }

    return tokens;
}

void CLI::ParseCommands(const string& thecommand, LPVOID instanceptr) {
    vector<string> cmdTokens = SplitString(thecommand, ' ');
    commands.push_back(cmdTokens);
    CLIModule::RegisterModule(thecommand, instanceptr, TRUE);
}

void CLI::ParseDynamicCommands(const string& modulename) {
    const auto& dynamicModules = CLIModule::GetDynamicLoadedModules();
    auto it = dynamicModules.find(modulename);
    if (it == dynamicModules.end() || it->second == nullptr) {
        LOG_ERROR_FMT("Dynamic module %s not found.", modulename.c_str());
        return;
    }
    commands.push_back(SplitString(modulename, ' '));
}

void CLI::ParseDynamicCommands() {
    const auto& dynamicModules = CLIModule::GetDynamicLoadedModules();
    for (const auto& pair : dynamicModules) {
        if (pair.second != nullptr) {
            commands.push_back(SplitString(pair.first, ' '));
        }
    }
}

void CLI::RegisterFromRegistry() {
    const auto& registeredCommands = CommandRegistry::Instance().GetCommands();
    for (const auto& cmdInfo : registeredCommands) {
        Command* cmd = cmdInfo.factory();
        if (cmd != nullptr) {
            ParseCommands(cmdInfo.name, cmd);
        }
    }
}

CLI::CLI() {
    if (!initialized) {
        // 从 CommandRegistry 注册所有使用 REGISTER_COMMAND 宏自动注册的命令
        RegisterFromRegistry();

        // PrintAllFunction 因包含额外成员变量 (ImageTableAnalyzer*)，无法使用标准宏
        // 需要手动注册
        ParseCommands(PrintAllFunction::name, PrintAllFunction::GetInstancePtr());

        initialized = true;
    }
}

CLI::~CLI() {
    CLIModule::UnregisterModules();
    delete analyzer;
}
