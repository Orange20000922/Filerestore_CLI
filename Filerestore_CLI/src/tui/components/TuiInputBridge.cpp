#include "TuiInputBridge.h"

TuiInputBridge::TuiInputBridge()
    : tuiMode_(false),
      hasPendingRequest_(false),
      responseReady_(false),
      cancelled_(false) {
}

TuiInputBridge& TuiInputBridge::Instance() {
    static TuiInputBridge instance;
    return instance;
}

void TuiInputBridge::EnableTuiMode() {
    tuiMode_ = true;
}

void TuiInputBridge::DisableTuiMode() {
    tuiMode_ = false;
    // 如果有阻塞的请求，取消它
    CancelRequest();
}

bool TuiInputBridge::GetLine(const std::string& prompt, std::string& result) {
    if (!tuiMode_) {
        // CLI 模式：直接使用标准输入
        std::cout << prompt;
        std::getline(std::cin, result);
        return true;
    }

    // TUI 模式：发送请求到 TUI，阻塞等待响应
    {
        std::lock_guard<std::mutex> lock(mutex_);
        currentPrompt_ = prompt;
        currentResponse_.clear();
        responseReady_ = false;
        cancelled_ = false;
        hasPendingRequest_ = true;
    }

    // 阻塞等待 TUI 线程提交响应
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return responseReady_ || cancelled_; });

        hasPendingRequest_ = false;

        if (cancelled_) {
            result.clear();
            return false;
        }

        result = currentResponse_;
        return true;
    }
}

bool TuiInputBridge::HasPendingRequest() {
    std::lock_guard<std::mutex> lock(mutex_);
    return hasPendingRequest_;
}

std::string TuiInputBridge::GetPrompt() {
    std::lock_guard<std::mutex> lock(mutex_);
    return currentPrompt_;
}

void TuiInputBridge::SubmitResponse(const std::string& response) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hasPendingRequest_) return;
    currentResponse_ = response;
    responseReady_ = true;
    cv_.notify_one();
}

void TuiInputBridge::CancelRequest() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hasPendingRequest_) return;
    cancelled_ = true;
    cv_.notify_one();
}
