#include "ProgressBar.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;

ProgressBar::ProgressBar(ULONGLONG totalItems, int width)
    : total(totalItems), current(0), extraInfo(0), barWidth(width), isVisible(true) {
    startTime = chrono::steady_clock::now();
    lastUpdate = startTime;
}

ProgressBar::~ProgressBar() {
    if (isVisible && current > 0) {
        cout << endl;  // 确保进度条后换行
    }
}

void ProgressBar::Update(ULONGLONG currentItems, ULONGLONG extra) {
    current = currentItems;
    extraInfo = extra;

    // 限制更新频率(每100ms更新一次)
    auto now = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - lastUpdate).count();

    if (elapsed < 100 && current < total) {
        return;  // 避免过于频繁的更新
    }

    lastUpdate = now;

    if (isVisible) {
        Render();
    }
}

void ProgressBar::Finish() {
    current = total;
    if (isVisible) {
        Render();
        cout << endl;
    }
}

void ProgressBar::Show() {
    isVisible = true;
}

void ProgressBar::Hide() {
    isVisible = false;
}

void ProgressBar::Render() {
    // 计算百分比
    double percentage = (total > 0) ? (double)current / total * 100.0 : 0.0;

    // 计算进度条填充数量
    int filled = (total > 0) ? (int)((double)current / total * barWidth) : 0;

    // 计算速度
    ULONGLONG speed = GetSpeed();

    // 构建进度条字符串
    cout << "\r";  // 回到行首

    // 百分比
    cout << fixed << setprecision(1) << setw(5) << percentage << "% ";

    // 进度条 [=========>     ]
    cout << "[";
    for (int i = 0; i < barWidth; i++) {
        if (i < filled - 1) {
            cout << "=";
        } else if (i == filled - 1 && filled < barWidth) {
            cout << ">";
        } else {
            cout << " ";
        }
    }
    cout << "] ";

    // 数量统计
    cout << FormatNumber(current) << "/" << FormatNumber(total);

    // 额外信息(如找到的文件数)
    if (extraInfo > 0) {
        cout << " | Found: " << FormatNumber(extraInfo);
    }

    // 速度
    if (speed > 0) {
        cout << " | " << FormatNumber(speed) << " rec/s";
    }

    // 清除行尾多余字符
    cout << "   " << flush;
}

string ProgressBar::FormatNumber(ULONGLONG num) {
    stringstream ss;
    ss.imbue(locale(""));  // 使用系统locale添加千位分隔符
    ss << num;
    return ss.str();
}

ULONGLONG ProgressBar::GetSpeed() {
    auto now = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - startTime).count();

    if (elapsed < 1000 || current == 0) {
        return 0;  // 前1秒不显示速度
    }

    // 计算每秒处理数量
    return (ULONGLONG)((double)current / elapsed * 1000.0);
}
