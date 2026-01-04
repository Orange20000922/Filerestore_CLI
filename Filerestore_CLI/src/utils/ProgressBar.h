#pragma once
#include <Windows.h>
#include <string>
#include <chrono>

using namespace std;

// 控制台进度条工具类
class ProgressBar
{
private:
    ULONGLONG total;           // 总数
    ULONGLONG current;         // 当前进度
    ULONGLONG extraInfo;       // 额外信息(如找到的文件数)
    int barWidth;              // 进度条宽度
    chrono::steady_clock::time_point startTime;  // 开始时间
    chrono::steady_clock::time_point lastUpdate; // 上次更新时间
    bool isVisible;            // 是否显示

public:
    ProgressBar(ULONGLONG totalItems, int width = 40);
    ~ProgressBar();

    // 更新进度
    void Update(ULONGLONG currentItems, ULONGLONG extra = 0);

    // 完成进度条
    void Finish();

    // 显示/隐藏
    void Show();
    void Hide();

private:
    // 内部渲染方法
    void Render();

    // 格式化大数字(添加千位分隔符)
    string FormatNumber(ULONGLONG num);

    // 计算速度(每秒处理数量)
    ULONGLONG GetSpeed();
};
