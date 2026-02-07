#pragma once
#include <string>
#include <sstream>
#include <mutex>
#include <functional>
#include <iostream>
#include <atomic>
#include <thread>

// 输出捕获器 - 重定向 cout 到 TUI
// 使用 std::streambuf 拦截所有 cout 输出
// 关键设计：FTXUI 输出转发到终端，命令输出捕获到 TUI 面板
class OutputCapture : public std::streambuf {
public:
    OutputCapture();
    ~OutputCapture();

    // 安装捕获（重定向 cout）
    void Install();

    // 卸载捕获（恢复 cout）
    void Uninstall();

    // 设置输出回调
    using OutputCallback = std::function<void(const std::string&)>;
    void SetCallback(OutputCallback callback) { callback_ = callback; }

    // 启用命令输出捕获（在命令线程中调用）
    void BeginCapture();

    // 结束命令输出捕获
    void EndCapture();

protected:
    // streambuf 重写：逐字符处理
    int overflow(int c) override;
    std::streamsize xsputn(const char* s, std::streamsize count) override;
    int sync() override;

private:
    std::streambuf* originalBuf_;  // 原始 cout 的 streambuf
    bool installed_;
    std::string lineBuffer_;       // 行缓冲
    std::mutex mutex_;
    OutputCallback callback_;

    // 命令输出捕获状态
    std::atomic<bool> capturing_;      // 是否正在捕获命令输出
    std::thread::id captureThreadId_;  // 命令线程 ID

    // 刷新行缓冲
    void FlushLine();
};
