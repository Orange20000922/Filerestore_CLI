#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <functional>

using namespace std;

// ============================================================================
// 命令宏系统 - 减少命令类的模板代码
// ============================================================================

// 前向声明
class Command;
class CLI;

// ============================================================================
// 命令注册器 - 用于自动注册命令
// ============================================================================
class CommandRegistry {
public:
    struct CommandInfo {
        string name;
        function<Command*()> factory;
    };

    static CommandRegistry& Instance() {
        static CommandRegistry instance;
        return instance;
    }

    void Register(const string& name, function<Command*()> factory) {
        commands.push_back({ name, factory });
    }

    const vector<CommandInfo>& GetCommands() const {
        return commands;
    }

private:
    CommandRegistry() = default;
    vector<CommandInfo> commands;
};

// 自动注册辅助类
template<typename T>
class CommandRegistrar {
public:
    CommandRegistrar(const string& name) {
        CommandRegistry::Instance().Register(name, []() -> Command* {
            return new T();
        });
    }
};

// ============================================================================
// 命令声明宏 - 用于 cmd.h
// ============================================================================

// 声明一个带参数的命令类
#define DECLARE_COMMAND(ClassName) \
class ClassName : public Command \
{ \
public: \
    static string name; \
    static vector<LPVOID> ArgsList; \
public: \
    ClassName(); \
    ~ClassName(); \
    void AcceptArgs(vector<LPVOID> argslist) override; \
    void Execute(string command) override; \
    BOOL HasArgs() override; \
    static BOOL CheckName(string input); \
    static LPVOID GetInstancePtr() { \
        return new ClassName(); \
    } \
    static string GetName() { \
        return name; \
    } \
}

// 声明一个无参数的命令类
#define DECLARE_COMMAND_NOARGS(ClassName) \
class ClassName : public Command \
{ \
public: \
    static string name; \
public: \
    ClassName(); \
    void AcceptArgs(vector<LPVOID> argslist) override; \
    void Execute(string command) override; \
    BOOL HasArgs() override; \
    static BOOL CheckName(string input); \
    static LPVOID GetInstancePtr() { \
        return new ClassName(); \
    } \
    static string GetName() { \
        return name; \
    } \
}

// ============================================================================
// 命令实现宏 - 用于 cmd.cpp
// ============================================================================

// 定义命令的静态成员和基础方法（带参数版本）
#define DEFINE_COMMAND_BASE(ClassName, CommandName, HasArgsValue) \
    string ClassName::name = CommandName; \
    vector<LPVOID> ClassName::ArgsList = vector<LPVOID>(); \
    ClassName::ClassName() { FlagHasArgs = HasArgsValue; } \
    ClassName::~ClassName() {} \
    void ClassName::AcceptArgs(vector<LPVOID> argslist) { \
        ClassName::ArgsList = argslist; \
    } \
    BOOL ClassName::HasArgs() { return FlagHasArgs; } \
    BOOL ClassName::CheckName(string input) { \
        return input.compare(name) == 0; \
    }

// 定义命令的静态成员和基础方法（无参数版本）
#define DEFINE_COMMAND_BASE_NOARGS(ClassName, CommandName) \
    string ClassName::name = CommandName; \
    ClassName::ClassName() { FlagHasArgs = FALSE; } \
    void ClassName::AcceptArgs(vector<LPVOID> argslist) { } \
    BOOL ClassName::HasArgs() { return FlagHasArgs; } \
    BOOL ClassName::CheckName(string input) { \
        return input.compare(name) == 0; \
    }

// 自动注册宏 - 在静态初始化期间注册命令
#define REGISTER_COMMAND(ClassName) \
    static CommandRegistrar<ClassName> g_##ClassName##_registrar(ClassName::name)

// 组合宏：定义基础 + 自动注册
#define IMPLEMENT_COMMAND(ClassName, CommandName, HasArgsValue) \
    DEFINE_COMMAND_BASE(ClassName, CommandName, HasArgsValue) \
    REGISTER_COMMAND(ClassName)

#define IMPLEMENT_COMMAND_NOARGS(ClassName, CommandName) \
    DEFINE_COMMAND_BASE_NOARGS(ClassName, CommandName) \
    REGISTER_COMMAND(ClassName)

// ============================================================================
// 便捷宏 - 快速定义完整命令
// ============================================================================

// 开始 Execute 方法实现
#define BEGIN_EXECUTE(ClassName) \
    void ClassName::Execute(string command) { \
        if (!CheckName(command)) return;

// 结束 Execute 方法实现
#define END_EXECUTE }

// 获取参数的便捷宏
#define GET_ARG_STRING(index) (*(string*)ArgsList[index])
#define GET_ARG_COUNT() (ArgsList.size())
#define HAS_ARG(index) (ArgsList.size() > (index))

