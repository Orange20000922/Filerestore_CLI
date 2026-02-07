#pragma once
#include <string>
#include <vector>
#include <map>

// TUI 命令辅助类
// 提供命令元数据：名称、描述、使用说明、参数定义
class CommandHelper {
public:
    // 参数信息
    struct ParamInfo {
        std::string name;        // 参数标识符 (drive, type, ...)
        std::string label;       // 显示标签 (Drive, File Type, ...)
        bool required;           // 是否必填
        std::string defaultValue;// 默认值
        std::vector<std::string> options; // 可选值（如文件类型列表）
    };

    struct CommandMetadata {
        std::string name;
        std::string description;
        std::string usage;
        std::vector<ParamInfo> params;
    };

    // 获取所有命令名称
    static std::vector<std::string> GetAllCommandNames();

    // 获取命令描述
    static std::string GetDescription(const std::string& commandName);

    // 获取命令使用说明
    static std::string GetUsage(const std::string& commandName);

    // 获取命令参数定义
    static std::vector<ParamInfo> GetParams(const std::string& commandName);

    // 根据输入前缀匹配命令（用于自动补全）
    static std::vector<std::string> MatchCommands(const std::string& prefix);

    // 组装命令字符串
    static std::string AssembleCommand(const std::string& cmdName,
                                        const std::string paramValues[],
                                        size_t paramCount);

private:
    static std::map<std::string, CommandMetadata> commandMetadata_;
    static void InitializeMetadata();
    static bool initialized_;
};
