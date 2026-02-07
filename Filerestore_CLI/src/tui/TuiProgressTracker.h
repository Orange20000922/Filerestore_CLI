#pragma once
#include <string>
#include <mutex>
#include <atomic>
#include <chrono>

// TUI 进度追踪器 - 全局单例
// 拦截 ProgressBar 的更新，转发给 TUI 界面
class TuiProgressTracker {
public:
    struct ProgressData {
        std::string label;           // 进度标签（如 "Scanning MFT records"）
        unsigned long long current;  // 当前进度
        unsigned long long total;    // 总数
        unsigned long long extra;    // 额外信息（如找到的文件数）
        bool active;                 // 是否有活跃的进度
        std::chrono::steady_clock::time_point startTime;  // 开始时间
    };

    struct StatusData {
        std::string drive;
        std::string mftStatus;   // "Loaded" / "Loading..." / "-"
        std::string usnStatus;
        std::string cacheStatus;
    };

    static TuiProgressTracker& Instance();

    // 进度更新（供 ProgressBar 调用）
    void UpdateProgress(const std::string& label, unsigned long long current,
                        unsigned long long total, unsigned long long extra = 0);

    void StartProgress(const std::string& label, unsigned long long total);
    void FinishProgress();

    // 状态更新（供各模块调用）
    void SetDrive(const std::string& drive);
    void SetMftStatus(const std::string& status);
    void SetUsnStatus(const std::string& status);
    void SetCacheStatus(const std::string& status);

    // 获取数据（供 TUI 渲染）
    ProgressData GetProgress();
    StatusData GetStatus();

    // 启用/禁用 TUI 模式
    void Enable();
    void Disable();
    bool IsEnabled() const { return enabled_; }

private:
    TuiProgressTracker();
    ~TuiProgressTracker();

    std::atomic<bool> enabled_;
    ProgressData progress_;
    StatusData status_;
    std::mutex mutex_;
};
