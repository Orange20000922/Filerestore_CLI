#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include "CarvedFileTypes.h"

using namespace std;

// 内存映射配置
constexpr size_t MMAP_THRESHOLD_COUNT = 100000;     // 超过10万条记录时使用内存映射
constexpr size_t MMAP_VIEW_SIZE = 64 * 1024 * 1024; // 64MB 视图大小
constexpr size_t MMAP_RECORDS_PER_VIEW = 200000;    // 每个视图大约容纳的记录数

// 内存映射的 Carved 文件结果
// 功能：使用 Windows 内存映射文件处理超大结果集
// 优势：按需加载数据，不占用大量物理内存
class MemoryMappedResults {
private:
    HANDLE hFile;               // 文件句柄
    HANDLE hMapping;            // 映射对象句柄
    LPVOID pMappedView;         // 当前映射的视图
    string filePath;            // 映射文件路径
    ULONGLONG fileSize;         // 文件大小
    ULONGLONG totalRecords;     // 总记录数
    ULONGLONG currentViewStart; // 当前视图起始偏移
    ULONGLONG currentViewSize;  // 当前视图大小
    bool isValid;               // 是否有效
    char driveLetter;           // 驱动器字母

    // 映射特定范围到内存
    bool MapViewRange(ULONGLONG offset, ULONGLONG size);

    // 取消当前视图映射
    void UnmapCurrentView();

    // 计算记录在文件中的偏移
    ULONGLONG CalculateRecordOffset(size_t index);

public:
    MemoryMappedResults();
    ~MemoryMappedResults();

    // 从缓存文件创建内存映射
    bool OpenFromCache(const string& cachePath);

    // 关闭内存映射
    void Close();

    // 获取单个记录（自动映射所需区域）
    bool GetRecord(size_t index, CarvedFileInfo& outInfo);

    // 获取一批记录（分页）
    bool GetRecordBatch(size_t startIndex, size_t count, vector<CarvedFileInfo>& outResults);

    // 获取总记录数
    ULONGLONG GetTotalRecords() const { return totalRecords; }

    // 检查是否有效
    bool IsValid() const { return isValid; }

    // 获取驱动器字母
    char GetDriveLetter() const { return driveLetter; }

    // 预取记录（提前映射）
    void PrefetchRange(size_t startIndex, size_t count);

    // 静态方法：检查是否应该使用内存映射
    static bool ShouldUseMemoryMapping(size_t recordCount) {
        return recordCount >= MMAP_THRESHOLD_COUNT;
    }
};
