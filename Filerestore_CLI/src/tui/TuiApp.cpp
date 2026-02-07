#include "TuiApp.h"
#include "components/OutputCapture.h"
#include "CommandHelper.h"
#include "TuiProgressTracker.h"
#include "cli.h"
#include "Logger.h"

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/component/event.hpp"

#include <Windows.h>
#include <thread>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace ftxui;

TuiApp* TuiApp::instance_ = nullptr;

TuiApp::TuiApp()
    : screen_(ScreenInteractive::Fullscreen()),
      currentView_(ViewMode::Welcome),
      running_(true),
      commandRunning_(false),
      capture_(nullptr),
      focusArea_(0),
      historyIndex_(-1) {
    instance_ = this;
}

TuiApp::~TuiApp() {
    running_ = false;
    if (instance_ == this) {
        instance_ = nullptr;
    }
}

void TuiApp::AppendOutput(const std::string& text) {
    std::lock_guard<std::mutex> lock(outputMutex_);
    outputLines_.push_back(text);
    if (outputLines_.size() > 500) {
        outputLines_.erase(outputLines_.begin());
    }
}

void TuiApp::AppendLog(const std::string& text) {
    std::lock_guard<std::mutex> lock(logMutex_);
    logLines_.push_back(text);
    if (logLines_.size() > 100) {
        logLines_.erase(logLines_.begin());
    }
}

void TuiApp::SetViewMode(ViewMode mode) {
    currentView_ = mode;
}

std::vector<std::string> TuiApp::GetOutputLines() {
    std::lock_guard<std::mutex> lock(outputMutex_);
    return outputLines_;
}

std::vector<std::string> TuiApp::GetLogLines() {
    std::lock_guard<std::mutex> lock(logMutex_);
    return logLines_;
}

void TuiApp::EnterParamMode(const std::string& cmdName) {
    currentParamCommand_ = cmdName;
    currentParams_ = CommandHelper::GetParams(cmdName);
    for (int i = 0; i < MAX_PARAMS; i++) {
        paramValues_[i] = (i < (int)currentParams_.size()) ?
            currentParams_[i].defaultValue : "";
    }
    SetViewMode(ViewMode::ParamInput);
    focusArea_ = 2;
}

void TuiApp::ExecuteCommand(const std::string& command) {
    if (commandRunning_) {
        AppendLog("[WARN] A command is already running.");
        return;
    }

    AppendLog("[CMD] " + command);
    SetViewMode(ViewMode::Output);
    focusArea_ = 1;

    std::thread([this, command]() {
        commandRunning_ = true;
        if (capture_) capture_->BeginCapture();

        try {
            CLI cli;
            cli.Run(command);
        } catch (const std::exception& e) {
            AppendOutput(std::string("Error: ") + e.what());
        } catch (...) {
            AppendOutput("Unknown error occurred.");
        }

        if (capture_) capture_->EndCapture();
        commandRunning_ = false;
        AppendLog("[DONE] " + command);
        screen_.PostEvent(Event::Custom);
    }).detach();
}

void TuiApp::Run() {
    // VT processing（Legacy console 需要）
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(hOut, &mode)) {
            SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }

    // 输出捕获
    OutputCapture capture;
    capture.SetCallback([this](const std::string& line) {
        AppendOutput(line);
        screen_.PostEvent(Event::Custom);
    });
    capture.Install();
    capture_ = &capture;

    // 启用 TUI 进度追踪
    TuiProgressTracker::Instance().Enable();

    AppendLog("[INFO] TUI started. Tab: switch focus | Esc: back | Q: quit");

    // ================================================================
    //  组件 1: 命令输入框
    // ================================================================
    std::string inputContent;
    std::string tempInput;
    auto inputBox = Input(&inputContent, "Type command here...");

    auto commandInputComponent = CatchEvent(inputBox, [&](Event event) {
        // Enter: 检查是否需要参数填充
        if (event == Event::Return && !inputContent.empty()) {
            std::string cmd = inputContent;

            // 提取命令名
            size_t sp = cmd.find(' ');
            std::string cmdName = (sp != std::string::npos) ? cmd.substr(0, sp) : cmd;
            bool hasArgs = (sp != std::string::npos && sp < cmd.size() - 1);

            auto params = CommandHelper::GetParams(cmdName);

            // 只输入了命令名 + 命令有必填参数 → 进入参数填充模式
            bool hasRequired = false;
            for (auto& p : params) { if (p.required) { hasRequired = true; break; } }

            if (!hasArgs && hasRequired) {
                EnterParamMode(cmdName);
                inputContent.clear();
                autocompleteMatches_.clear();
                screen_.PostEvent(Event::Custom);
                return true;
            }

            // 直接执行
            inputContent.clear();
            historyIndex_ = -1;
            autocompleteMatches_.clear();
            if (commandHistory_.empty() || commandHistory_.back() != cmd) {
                commandHistory_.push_back(cmd);
            }
            ExecuteCommand(cmd);
            return true;
        }

        // 上箭头: 命令历史
        if (event == Event::ArrowUp) {
            if (!commandHistory_.empty()) {
                if (historyIndex_ == -1) {
                    tempInput = inputContent;
                    historyIndex_ = (int)commandHistory_.size() - 1;
                } else if (historyIndex_ > 0) {
                    historyIndex_--;
                }
                if (historyIndex_ >= 0 && historyIndex_ < (int)commandHistory_.size()) {
                    inputContent = commandHistory_[historyIndex_];
                }
            }
            return true;
        }

        // 下箭头: 命令历史
        if (event == Event::ArrowDown) {
            if (historyIndex_ != -1) {
                historyIndex_++;
                if (historyIndex_ >= (int)commandHistory_.size()) {
                    historyIndex_ = -1;
                    inputContent = tempInput;
                } else {
                    inputContent = commandHistory_[historyIndex_];
                }
            }
            return true;
        }

        // Tab: 自动补全
        if (event == Event::Tab) {
            if (!autocompleteMatches_.empty()) {
                inputContent = autocompleteMatches_[0];
                autocompleteMatches_.clear();
                return true;
            }
        }

        return false;
    });

    // ================================================================
    //  组件 2: 主菜单
    // ================================================================
    std::vector<std::string> menuEntries = {
        "Smart Recovery (USN-based)",
        "Scan for Deleted Files",
        "Deep Scan (Signature Carving)",
        "Repair Corrupted Files",
        "Browse Previous Results",
        "Advanced (Command Line)",
    };
    int menuSelected = 0;
    auto mainMenu = Menu(&menuEntries, &menuSelected);

    // 菜单命令映射
    const std::string menuCommands[] = {
        "recover",      // 0: Smart Recovery (USN + 签名扫描联合)
        "listdeleted",  // 1: Scan Deleted
        "carvepool",    // 2: Deep Scan
        "repair",       // 3: Repair
        "",             // 4: Browse Results (直接执行)
        "",             // 5: Advanced
    };

    auto menuWithAction = CatchEvent(mainMenu, [&](Event event) {
        if (event == Event::Return) {
            if (menuSelected >= 0 && menuSelected <= 3) {
                // 有参数的命令 → 进入参数填充模式
                EnterParamMode(menuCommands[menuSelected]);
                screen_.PostEvent(Event::Custom);
            } else if (menuSelected == 4) {
                ExecuteCommand("carvelist");
            } else if (menuSelected == 5) {
                SetViewMode(ViewMode::Output);
                focusArea_ = 1;
                AppendOutput("Command mode. Type commands below.");
                screen_.PostEvent(Event::Custom);
            }
            return true;
        }
        return false;
    });

    // ================================================================
    //  组件 3: 参数填充表单
    // ================================================================
    auto paramInput0 = Input(&paramValues_[0], "...");
    auto paramInput1 = Input(&paramValues_[1], "...");
    auto paramInput2 = Input(&paramValues_[2], "...");
    auto paramInput3 = Input(&paramValues_[3], "...");
    auto paramInput4 = Input(&paramValues_[4], "...");
    auto paramInput5 = Input(&paramValues_[5], "...");

    auto paramContainer = Container::Vertical({
        paramInput0, paramInput1, paramInput2,
        paramInput3, paramInput4, paramInput5,
    });

    // 参数输入指针数组（用于渲染）
    Component paramInputPtrs[] = {
        paramInput0, paramInput1, paramInput2,
        paramInput3, paramInput4, paramInput5,
    };

    auto paramWithEvents = CatchEvent(paramContainer, [&](Event event) {
        // Enter: 验证并执行
        if (event == Event::Return) {
            // 验证必填参数
            for (size_t i = 0; i < currentParams_.size(); i++) {
                if (currentParams_[i].required && paramValues_[i].empty()) {
                    AppendLog("[WARN] Missing: " + currentParams_[i].label);
                    screen_.PostEvent(Event::Custom);
                    return true;
                }
            }
            // 组装命令
            std::string fullCmd = CommandHelper::AssembleCommand(
                currentParamCommand_, paramValues_, currentParams_.size());

            if (commandHistory_.empty() || commandHistory_.back() != fullCmd) {
                commandHistory_.push_back(fullCmd);
            }
            ExecuteCommand(fullCmd);
            return true;
        }

        // Esc: 取消
        if (event == Event::Escape) {
            focusArea_ = 1;
            SetViewMode(ViewMode::Output);
            screen_.PostEvent(Event::Custom);
            return true;
        }

        return false;
    });

    // ================================================================
    //  焦点管理: Container::Tab
    // ================================================================
    auto focusContainer = Container::Tab({
        menuWithAction,        // 0: 菜单
        commandInputComponent, // 1: 命令输入
        paramWithEvents,       // 2: 参数表单
    }, &focusArea_);

    // ================================================================
    //  渲染
    // ================================================================
    auto layoutRenderer = Renderer(focusContainer, [&]() {
        // 更新自动补全
        if (focusArea_ == 1 && !inputContent.empty() && historyIndex_ == -1) {
            autocompleteMatches_ = CommandHelper::MatchCommands(inputContent);
            if (autocompleteMatches_.size() > 8) autocompleteMatches_.resize(8);
        } else {
            autocompleteMatches_.clear();
        }

        ViewMode mode = currentView_.load();

        // === 标题栏 ===
        std::string focusLabel = (focusArea_ == 0) ? " MENU " :
                                 (focusArea_ == 1) ? " CMD " : " PARAMS ";
        auto titleBar = hbox({
            text(" Filerestore CLI v0.3.2 ") | bold,
            filler(),
            text(commandRunning_ ? " [Running...] " : " [Ready] "),
            text(focusLabel) | bgcolor(Color::DarkGreen),
            text(" "),
        }) | color(Color::White) | bgcolor(Color::Blue);

        // === 左侧状态面板 ===
        auto statusData = TuiProgressTracker::Instance().GetStatus();
        auto statusPanel = vbox({
            text(" STATUS ") | bold | center,
            separator(),
            text(" Drive:  " + statusData.drive),
            text(" MFT:    " + statusData.mftStatus),
            text(" USN:    " + statusData.usnStatus),
            text(" Cache:  " + statusData.cacheStatus),
            separator(),
            text(" KEYS ") | bold | center,
            separator(),
            text(" [Tab] Focus"),
            text(" [Esc] Back"),
            text(" [Q]   Quit"),
            filler(),
        }) | size(WIDTH, EQUAL, 16) | border;

        // === 主视图 ===
        Element mainContent;

        if (mode == ViewMode::Welcome) {
            // ---- 主菜单 ----
            mainContent = vbox({
                text(""),
                text(" What do you want to do?") | bold,
                text(""),
                menuWithAction->Render(),
                text(""),
                text(" [Enter] Select  [Tab] Switch to Command Input") | dim,
            }) | border;

        } else if (mode == ViewMode::ParamInput) {
            // ---- 参数填充表单 ----
            Elements paramElements;
            paramElements.push_back(
                hbox({
                    text(" " + currentParamCommand_ + " ") | bold |
                        bgcolor(Color::DarkBlue) | color(Color::White),
                    text("  " + CommandHelper::GetDescription(currentParamCommand_)) | dim,
                })
            );
            paramElements.push_back(separator());

            for (size_t i = 0; i < currentParams_.size() && i < (size_t)MAX_PARAMS; i++) {
                auto& p = currentParams_[i];

                // 标签
                std::string label = p.label + ":";
                auto labelElem = text((p.required ? " * " : "   ") + label) | bold |
                    size(WIDTH, EQUAL, 22);

                // 输入框
                auto inputElem = paramInputPtrs[i]->Render() | flex;

                // 选项提示
                Element optHint = text("");
                if (!p.options.empty()) {
                    std::string opts;
                    for (size_t j = 0; j < p.options.size() && j < 8; j++) {
                        if (j > 0) opts += "/";
                        opts += p.options[j];
                    }
                    if (p.options.size() > 8) opts += "/...";
                    optHint = text(" (" + opts + ")") | dim;
                } else if (!p.required) {
                    optHint = text(" (optional)") | dim;
                }

                paramElements.push_back(hbox({labelElem, inputElem, optHint}));
            }

            paramElements.push_back(separator());
            paramElements.push_back(
                text(" [Enter] Execute  [Esc] Cancel  [Up/Down] Navigate Fields") | dim
            );
            mainContent = vbox(paramElements) | border;

        } else {
            // ---- 命令输出 ----
            auto lines = GetOutputLines();
            Elements outputElements;

            // 进度条渲染（如果有活跃进度）
            auto progressData = TuiProgressTracker::Instance().GetProgress();
            if (progressData.active && progressData.total > 0) {
                // 计算百分比
                double percentage = (double)progressData.current / progressData.total * 100.0;
                int barWidth = 50;
                int filled = (int)((double)progressData.current / progressData.total * barWidth);

                // 标签
                if (!progressData.label.empty()) {
                    outputElements.push_back(text(" " + progressData.label) | bold);
                }

                // 进度条
                std::string bar = "[";
                for (int i = 0; i < barWidth; i++) {
                    if (i < filled) bar += "=";
                    else bar += " ";
                }
                bar += "]";

                // 进度信息
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(1) << percentage << "% ";
                oss << progressData.current << "/" << progressData.total;
                if (progressData.extra > 0) {
                    oss << " | Found: " << progressData.extra;
                }

                outputElements.push_back(
                    hbox({
                        text(bar) | color(Color::Green),
                        text(" " + oss.str()) | dim,
                    })
                );
                outputElements.push_back(separator());
            }

            // 输出文本
            int startLine = (int)lines.size() > 25 ? (int)lines.size() - 25 : 0;
            for (int i = startLine; i < (int)lines.size(); i++) {
                outputElements.push_back(text(lines[i]));
            }
            if (outputElements.empty() && !progressData.active) {
                outputElements.push_back(text(" (No output yet)") | dim);
            }
            mainContent = vbox(outputElements) | frame | border;
        }

        // === 底部面板 ===
        Elements bottomParts;

        // 命令输入栏（参数模式时不显示）
        if (mode != ViewMode::ParamInput) {
            auto cmdPrompt = text(" Command> ") | bold;
            if (focusArea_ == 1) cmdPrompt = cmdPrompt | color(Color::Green);

            bottomParts.push_back(
                hbox({cmdPrompt, commandInputComponent->Render() | flex}) | border
            );
        }

        // 自动补全（仅在命令输入模式）
        if (focusArea_ == 1 && !autocompleteMatches_.empty()) {
            Elements acElems;
            for (size_t i = 0; i < autocompleteMatches_.size() && i < 5; i++) {
                std::string desc = CommandHelper::GetDescription(autocompleteMatches_[i]);
                auto nameElem = text((i == 0 ? " > " : "   ") + autocompleteMatches_[i]);
                if (i == 0) nameElem = nameElem | color(Color::Green);
                acElems.push_back(
                    hbox({
                        nameElem | size(WIDTH, EQUAL, 25),
                        text(desc) | dim,
                    })
                );
            }
            acElems.push_back(text(" [Tab] complete") | dim);
            bottomParts.push_back(vbox(acElems) | border);
        }

        // 使用说明（仅在命令输入模式 + 精确匹配时）
        if (focusArea_ == 1 && !inputContent.empty()) {
            std::string cmdName;
            size_t sp = inputContent.find(' ');
            cmdName = (sp != std::string::npos) ? inputContent.substr(0, sp) : inputContent;
            auto matches = CommandHelper::MatchCommands(cmdName);
            if (matches.size() == 1 && matches[0] == cmdName) {
                auto params = CommandHelper::GetParams(cmdName);
                bool hasRequired = false;
                for (auto& p : params) if (p.required) { hasRequired = true; break; }

                Elements usageElems;
                usageElems.push_back(hbox({
                    text(" Usage: ") | bold,
                    text(CommandHelper::GetUsage(cmdName)) | color(Color::Cyan),
                }));
                if (hasRequired) {
                    usageElems.push_back(
                        text(" Press Enter to fill parameters interactively.") | dim
                    );
                }
                bottomParts.push_back(vbox(usageElems) | border);
            }
        }

        // 日志
        auto logs = GetLogLines();
        Elements logElems;
        int logStart = (int)logs.size() > 2 ? (int)logs.size() - 2 : 0;
        for (int i = logStart; i < (int)logs.size(); i++) {
            logElems.push_back(text(" " + logs[i]) | dim);
        }
        if (logElems.empty()) logElems.push_back(text(" Ready.") | dim);
        bottomParts.push_back(vbox(logElems) | size(HEIGHT, EQUAL, 3) | border);

        // === 组合 ===
        Elements layout;
        layout.push_back(titleBar);
        layout.push_back(
            hbox({statusPanel, mainContent | flex}) | flex
        );
        for (auto& e : bottomParts) layout.push_back(e);

        return vbox(layout);
    });

    // ================================================================
    //  全局事件处理
    // ================================================================
    auto appComponent = CatchEvent(layoutRenderer, [&](Event event) {
        // Tab: 焦点切换 (仅菜单模式 → 命令输入)
        if (event == Event::Tab) {
            if (focusArea_ == 0) {
                focusArea_ = 1;
                if (currentView_ == ViewMode::Welcome) SetViewMode(ViewMode::Output);
                return true;
            }
            // focusArea_==1 时 Tab 传递给 commandInputComponent (自动补全)
            // focusArea_==2 时 Tab 传递给 paramContainer (切换字段)
            return false;
        }

        // Esc: 返回上一级
        if (event == Event::Escape) {
            if (focusArea_ == 2) {
                focusArea_ = 1;
                SetViewMode(ViewMode::Output);
                return true;
            } else if (focusArea_ == 1) {
                focusArea_ = 0;
                SetViewMode(ViewMode::Welcome);
                inputContent.clear();
                autocompleteMatches_.clear();
                return true;
            }
            return false;
        }

        // Q: 退出（仅菜单模式 + Welcome 视图）
        if (event == Event::Character('q') || event == Event::Character('Q')) {
            if (focusArea_ == 0 && !commandRunning_ && currentView_ == ViewMode::Welcome) {
                screen_.Exit();
                return true;
            }
        }

        return false;
    });

    // ================================================================
    //  异步刷新
    // ================================================================
    std::thread refreshThread([this]() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (commandRunning_) {
                screen_.PostEvent(Event::Custom);
            }
        }
    });
    refreshThread.detach();

    // ================================================================
    //  主循环
    // ================================================================
    screen_.Loop(appComponent);

    running_ = false;
    capture_ = nullptr;
    capture.Uninstall();

    // 禁用 TUI 进度追踪
    TuiProgressTracker::Instance().Disable();
}
