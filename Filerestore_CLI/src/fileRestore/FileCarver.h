#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "MFTReader.h"

using namespace std;

// 文件签名定义
struct FileSignature {
    string extension;           // 文件扩展名（如 "zip", "pdf"）
    vector<BYTE> header;        // 文件头签名
    vector<BYTE> footer;        // 文件尾签名（可选，如PDF的%%EOF）
    ULONGLONG maxSize;          // 最大文件大小（字节）
    ULONGLONG minSize;          // 最小文件大小
    bool hasFooter;             // 是否有明确的文件尾
    string description;         // 描述
    BYTE firstByte;             // 签名第一个字节（用于快速查找）
};

// Carving 结果
struct CarvedFileInfo {
    ULONGLONG startLCN;         // 起始逻辑簇号
    ULONGLONG startOffset;      // 簇内偏移
    ULONGLONG fileSize;         // 文件大小（估计或精确）
    string extension;           // 文件类型
    string description;         // 文件类型描述
    bool hasValidFooter;        // 是否找到有效的文件尾
    double confidence;          // 置信度 (0.0-1.0)
};

// 扫描模式
enum CarvingMode {
    CARVE_QUICK,        // 快速扫描：只扫描未分配的簇
    CARVE_FULL,         // 完整扫描：扫描整个磁盘
    CARVE_SMART         // 智能扫描：跳过系统区域和已知文件
};

// 扫描统计信息
struct CarvingStats {
    ULONGLONG totalClusters;
    ULONGLONG scannedClusters;
    ULONGLONG skippedClusters;      // 跳过的空簇
    ULONGLONG filesFound;
    ULONGLONG bytesRead;
    DWORD elapsedMs;                // 耗时（毫秒）
    double readSpeedMBps;           // 读取速度 MB/s
    double scanSpeedMBps;           // 扫描速度 MB/s
    double ioBusyPercent;           // I/O 忙碌百分比
    double cpuBusyPercent;          // CPU 忙碌百分比
};

// 双缓冲区结构（用于异步I/O）
struct ScanBuffer {
    vector<BYTE> data;              // 数据缓冲区
    ULONGLONG startLCN;             // 起始簇号
    ULONGLONG clusterCount;         // 簇数量
    bool ready;                     // 数据是否就绪
    bool isEmpty;                   // 是否为空块
    bool isLast;                    // 是否为最后一块
};

class FileCarver {
private:
    MFTReader* reader;
    map<string, FileSignature> signatures;      // 文件签名数据库

    // 优化：按首字节分组的签名索引
    unordered_map<BYTE, vector<const FileSignature*>> signatureIndex;

    // 要扫描的签名集合（用于选择性扫描）
    set<string> activeSignatures;

    // 统计信息
    CarvingStats stats;
    atomic<bool> shouldStop;                    // 中断标志（线程安全）

    // ==================== 异步I/O 双缓冲 ====================
    // 双缓冲区
    ScanBuffer buffers[2];
    int currentReadBuffer;                      // 当前正在读取的缓冲区
    int currentScanBuffer;                      // 当前正在扫描的缓冲区

    // 同步原语
    mutex bufferMutex;
    condition_variable bufferReadyCV;           // 缓冲区就绪通知
    condition_variable bufferConsumedCV;        // 缓冲区已消费通知

    // 异步模式开关
    bool useAsyncIO;

    // 性能统计（线程安全）
    atomic<ULONGLONG> ioWaitTimeMs;             // I/O等待时间
    atomic<ULONGLONG> scanTimeMs;               // 扫描处理时间
    atomic<ULONGLONG> totalIoTimeMs;            // 总I/O时间
    atomic<ULONGLONG> totalScanTimeMs;          // 总扫描时间

    // I/O 读取线程函数
    void IOReaderThread(ULONGLONG startLCN, ULONGLONG endLCN,
                       ULONGLONG bufferClusters, ULONGLONG bytesPerCluster,
                       CarvingMode mode);

    // 扫描工作线程函数
    void ScanWorkerThread(vector<CarvedFileInfo>& results,
                         ULONGLONG bytesPerCluster, ULONGLONG maxResults);

    // 初始化签名数据库
    void InitializeSignatures();

    // 构建签名索引（按首字节分组）
    void BuildSignatureIndex();

    // 构建活动签名索引
    void BuildActiveSignatureIndex();

    // 匹配签名
    bool MatchSignature(const BYTE* data, size_t dataSize,
                       const vector<BYTE>& signature);

    // 查找文件尾
    ULONGLONG FindFooter(const BYTE* data, size_t dataSize,
                        const vector<BYTE>& footer, ULONGLONG maxSearch);

    // 估算文件大小（当没有明确文件尾时）
    ULONGLONG EstimateFileSize(const BYTE* data, size_t dataSize,
                              const FileSignature& sig,
                              ULONGLONG* outFooterPos = nullptr);

    // 验证文件有效性（基于内容）- 优化版本，避免重复计算
    double ValidateFileOptimized(const BYTE* data, size_t dataSize,
                                const FileSignature& sig,
                                bool signatureAlreadyMatched,
                                ULONGLONG footerPos);

    // 提取文件数据
    bool ExtractFile(ULONGLONG startLCN, ULONGLONG startOffset,
                    ULONGLONG fileSize, vector<BYTE>& fileData);

    // 检查缓冲区是否为空（全零或接近全零）
    bool IsEmptyBuffer(const BYTE* data, size_t size);

    // 单次扫描核心函数 - 扫描一个缓冲区，检查所有活动签名
    void ScanBufferMultiSignature(const BYTE* data, size_t dataSize,
                                  ULONGLONG baseLCN, ULONGLONG bytesPerCluster,
                                  vector<CarvedFileInfo>& results,
                                  ULONGLONG maxResults);

public:
    FileCarver(MFTReader* mftReader);
    ~FileCarver();

    // 扫描特定类型的文件
    vector<CarvedFileInfo> ScanForFileType(const string& fileType,
                                          CarvingMode mode = CARVE_SMART,
                                          ULONGLONG maxResults = 1000);

    // 扫描多种类型（单次扫描，多签名匹配）
    vector<CarvedFileInfo> ScanForFileTypes(const vector<string>& fileTypes,
                                           CarvingMode mode = CARVE_SMART,
                                           ULONGLONG maxResults = 1000);

    // 扫描所有支持的文件类型（优化：单次扫描）
    vector<CarvedFileInfo> ScanAllTypes(CarvingMode mode = CARVE_SMART,
                                       ULONGLONG maxResults = 1000);

    // 恢复 carved 文件
    bool RecoverCarvedFile(const CarvedFileInfo& info,
                          const string& outputPath);

    // 获取支持的文件类型列表
    vector<string> GetSupportedTypes();

    // 获取扫描进度
    double GetProgress() const;

    // 停止扫描
    void StopScanning();

    // 获取统计信息
    const CarvingStats& GetStats() const { return stats; }
    ULONGLONG GetScannedClusters() const { return stats.scannedClusters; }
    ULONGLONG GetFilesFound() const { return stats.filesFound; }

    // 异步I/O设置
    void SetAsyncMode(bool enabled) { useAsyncIO = enabled; }
    bool IsAsyncMode() const { return useAsyncIO; }

    // 异步扫描（双缓冲 + 生产者-消费者模式）
    vector<CarvedFileInfo> ScanForFileTypesAsync(const vector<string>& fileTypes,
                                                 CarvingMode mode = CARVE_SMART,
                                                 ULONGLONG maxResults = 1000);
};
