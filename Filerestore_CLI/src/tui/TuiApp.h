#pragma once
#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"

#include "CommandHelper.h"

// 前向声明
class OutputCapture;

// TUI 应用主类
class TuiApp {
public:
    TuiApp();
    ~TuiApp();

    void Run();

    static TuiApp* GetInstance() { return instance_; }

    void AppendOutput(const std::string& text);
    void AppendLog(const std::string& text);

    enum class ViewMode {
        Welcome,     // 主菜单
        Output,      // 命令输出
        ParamInput,  // 参数填充表单
        Scan,        // 扫描进度（Phase 3）
        Results,     // 结果表格（Phase 4）
    };
    void SetViewMode(ViewMode mode);

    void ExecuteCommand(const std::string& command);

    std::vector<std::string> GetOutputLines();
    std::vector<std::string> GetLogLines();

private:
    static TuiApp* instance_;

    ftxui::ScreenInteractive screen_;
    std::atomic<ViewMode> currentView_;

    // 输出缓冲
    std::vector<std::string> outputLines_;
    std::vector<std::string> logLines_;
    std::mutex outputMutex_;
    std::mutex logMutex_;

    // 运行状态
    std::atomic<bool> running_;
    std::atomic<bool> commandRunning_;

    // 输出捕获器
    OutputCapture* capture_;

    // 焦点管理: 0=菜单, 1=命令输入, 2=参数表单
    int focusArea_;

    // 命令历史
    std::vector<std::string> commandHistory_;
    int historyIndex_;

    // 命令补全
    std::vector<std::string> autocompleteMatches_;

    // 参数填充
    static const int MAX_PARAMS = 6;
    std::string paramValues_[MAX_PARAMS];
    std::string currentParamCommand_;
    std::vector<CommandHelper::ParamInfo> currentParams_;

    // 进入参数填充模式
    void EnterParamMode(const std::string& cmdName);
};
