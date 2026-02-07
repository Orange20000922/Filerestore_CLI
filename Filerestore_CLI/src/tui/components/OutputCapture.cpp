#include "OutputCapture.h"
#include <functional>

OutputCapture::OutputCapture()
    : originalBuf_(nullptr), installed_(false), capturing_(false) {
}

OutputCapture::~OutputCapture() {
    Uninstall();
}

void OutputCapture::Install() {
    if (installed_) return;
    originalBuf_ = std::cout.rdbuf(this);
    installed_ = true;
}

void OutputCapture::Uninstall() {
    if (!installed_) return;
    std::cout.rdbuf(originalBuf_);
    installed_ = false;

    // 刷新残留内容
    if (!lineBuffer_.empty()) {
        FlushLine();
    }
}

void OutputCapture::BeginCapture() {
    captureThreadId_ = std::this_thread::get_id();
    capturing_ = true;
}

void OutputCapture::EndCapture() {
    capturing_ = false;
    // 刷新残留内容
    std::lock_guard<std::mutex> lock(mutex_);
    if (!lineBuffer_.empty()) {
        FlushLine();
    }
}

int OutputCapture::overflow(int c) {
    // 判断是否为命令线程输出
    bool isCommandThread = capturing_ && std::this_thread::get_id() == captureThreadId_;

    if (isCommandThread) {
        // 命令输出：仅捕获到 TUI 面板，不转发到终端
        std::lock_guard<std::mutex> lock(mutex_);
        if (c == '\n') {
            FlushLine();
        } else if (c != '\r') {
            lineBuffer_ += static_cast<char>(c);
        }
    } else {
        // FTXUI 或其他输出：转发到终端，不捕获
        if (originalBuf_) {
            originalBuf_->sputc(c);
        }
    }
    return c;
}

std::streamsize OutputCapture::xsputn(const char* s, std::streamsize count) {
    bool isCommandThread = capturing_ && std::this_thread::get_id() == captureThreadId_;

    if (isCommandThread) {
        // 命令输出：捕获
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::streamsize i = 0; i < count; i++) {
            if (s[i] == '\n') {
                FlushLine();
            } else if (s[i] != '\r') {
                lineBuffer_ += s[i];
            }
        }
    } else {
        // FTXUI 输出：转发
        if (originalBuf_) {
            originalBuf_->sputn(s, count);
        }
    }
    return count;
}

int OutputCapture::sync() {
    // 始终转发 flush 到原始 streambuf（FTXUI 需要）
    if (originalBuf_) {
        return originalBuf_->pubsync();
    }
    return 0;
}

void OutputCapture::FlushLine() {
    if (callback_) {
        callback_(lineBuffer_);
    }
    lineBuffer_.clear();
}
