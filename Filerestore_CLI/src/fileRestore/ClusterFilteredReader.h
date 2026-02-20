#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <memory>

class MFTReader;
class OverwriteDetector;

// 簇健康报告（所有恢复路径共用）
struct ClusterHealthReport {
    ULONGLONG totalClusters = 0;
    ULONGLONG goodClusters = 0;
    ULONGLONG overwrittenClusters = 0;
    double healthPercentage = 100.0;     // goodClusters / totalClusters * 100
    double detectionTimeMs = 0.0;
    std::vector<bool> clusterIsGood;     // 索引=文件内簇顺序，true=有效
    bool formatTruncated = false;        // 是否执行了格式截断
    ULONGLONG truncatedBytes = 0;        // 格式截断移除的字节数
};

// 集中式簇过滤读取器
class ClusterFilteredReader {
public:
    explicit ClusterFilteredReader(MFTReader* reader);
    ~ClusterFilteredReader();

    // Data Runs 读取（UsnTargetedRecovery 路径）
    bool ReadFromDataRuns(
        const std::vector<std::pair<ULONGLONG, ULONGLONG>>& dataRuns,
        ULONGLONG fileSize,
        const std::wstring& fileName,
        std::vector<BYTE>& fileData,
        ClusterHealthReport& report);

    // 连续 LCN 读取（FileCarverRecovery / FileRestoreAPI 路径）
    bool ReadContiguous(
        ULONGLONG startLCN,
        ULONGLONG startOffset,
        ULONGLONG fileSize,
        const std::wstring& fileName,
        std::vector<BYTE>& fileData,
        ClusterHealthReport& report);

private:
    MFTReader* reader_;
    std::unique_ptr<OverwriteDetector> detector_;  // 延迟创建，复用 $Bitmap 缓存

    OverwriteDetector& GetDetector();

    ClusterHealthReport BuildHealthReport(
        const std::vector<std::pair<ULONGLONG, ULONGLONG>>& dataRuns);

    bool ReadWithFilter(
        const std::vector<std::pair<ULONGLONG, ULONGLONG>>& dataRuns,
        ULONGLONG fileSize,
        const ClusterHealthReport& report,
        std::vector<BYTE>& fileData);

    ULONGLONG TryFormatTruncation(
        std::vector<BYTE>& fileData,
        const std::wstring& fileName);
};
