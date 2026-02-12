#pragma once
#include <string>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>

// TUI 输入桥接 - 命令线程与 TUI 之间的输入通道
//
// 设计模式与 OutputCapture 对称:
//   OutputCapture: cout (命令线程) → TUI 输出面板
//   TuiInputBridge: TUI 输入框 → getline (命令线程)
//
// 命令线程调用 GetLine() 阻塞等待用户输入
// TUI 线程检测到请求后切换输入框为"响应模式"，用户输入后唤醒命令线程
class TuiInputBridge {
public:
    static TuiInputBridge& Instance();

    // ==================== 命令线程调用 ====================

    // 替代 getline(cin, ...) — 自动适配 CLI/TUI 模式
    // CLI 模式: 直接调用 getline(cin, ...)
    // TUI 模式: 发送请求到 TUI，阻塞等待响应
    // 返回: true=正常输入, false=用户取消(Esc)
    bool GetLine(const std::string& prompt, std::string& result);

    // ==================== TUI 线程调用 ====================

    // 检查是否有待处理的输入请求
    bool HasPendingRequest();

    // 获取当前请求的提示文本
    std::string GetPrompt();

    // 提交用户的输入响应（唤醒命令线程）
    void SubmitResponse(const std::string& response);

    // 取消输入请求（唤醒命令线程，GetLine 返回 false）
    void CancelRequest();

    // ==================== 模式管理 ====================

    void EnableTuiMode();
    void DisableTuiMode();
    bool IsTuiMode() const { return tuiMode_; }

private:
    TuiInputBridge();
    ~TuiInputBridge() = default;

    TuiInputBridge(const TuiInputBridge&) = delete;
    TuiInputBridge& operator=(const TuiInputBridge&) = delete;

    std::atomic<bool> tuiMode_;

    // 请求/响应同步
    std::mutex mutex_;
    std::condition_variable cv_;
    bool hasPendingRequest_;
    std::string currentPrompt_;
    std::string currentResponse_;
    bool responseReady_;
    bool cancelled_;
};
