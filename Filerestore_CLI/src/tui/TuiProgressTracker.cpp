#include "TuiProgressTracker.h"

TuiProgressTracker& TuiProgressTracker::Instance() {
    static TuiProgressTracker instance;
    return instance;
}

TuiProgressTracker::TuiProgressTracker() : enabled_(false) {
    progress_.label = "";
    progress_.current = 0;
    progress_.total = 0;
    progress_.extra = 0;
    progress_.active = false;

    status_.drive = "-";
    status_.mftStatus = "-";
    status_.usnStatus = "-";
    status_.cacheStatus = "-";
}

TuiProgressTracker::~TuiProgressTracker() {}

void TuiProgressTracker::UpdateProgress(const std::string& label,
                                        unsigned long long current,
                                        unsigned long long total,
                                        unsigned long long extra) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);
    progress_.label = label;
    progress_.current = current;
    progress_.total = total;
    progress_.extra = extra;
    progress_.active = true;
}

void TuiProgressTracker::StartProgress(const std::string& label,
                                        unsigned long long total) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);
    progress_.label = label;
    progress_.current = 0;
    progress_.total = total;
    progress_.extra = 0;
    progress_.active = true;
    progress_.startTime = std::chrono::steady_clock::now();
}

void TuiProgressTracker::FinishProgress() {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);
    progress_.active = false;
}

void TuiProgressTracker::SetDrive(const std::string& drive) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    status_.drive = drive;
}

void TuiProgressTracker::SetMftStatus(const std::string& status) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    status_.mftStatus = status;
}

void TuiProgressTracker::SetUsnStatus(const std::string& status) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    status_.usnStatus = status;
}

void TuiProgressTracker::SetCacheStatus(const std::string& status) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    status_.cacheStatus = status;
}

TuiProgressTracker::ProgressData TuiProgressTracker::GetProgress() {
    std::lock_guard<std::mutex> lock(mutex_);
    return progress_;
}

TuiProgressTracker::StatusData TuiProgressTracker::GetStatus() {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

void TuiProgressTracker::Enable() {
    enabled_ = true;
}

void TuiProgressTracker::Disable() {
    enabled_ = false;
}
