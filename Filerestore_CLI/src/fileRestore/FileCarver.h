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
#include <memory>
#include "MFTReader.h"
#include "SignatureScanThreadPool.h"
#include "CarvedFileTypes.h"
#include "MLClassifier.h"
#include "FileFormatUtils.h"

using namespace std;

// FileSignature 已移至 FileFormatUtils.h

// TimestampSource 和 CarvedFileInfo 已移至 CarvedFileTypes.h

// 扫描模式
enum CarvingMode {
    CARVE_QUICK,        // 快速扫描：只扫描未分配的簇
    CARVE_FULL,         // 完整扫描：扫描整个磁盘
    CARVE_SMART         // 智能扫描：跳过系统区域和已知文件
};

// 混合扫描配置（签名 + ML）
struct HybridScanConfig {
    bool enableSignatureScan = true;     // 启用签名扫描
    bool enableMLScan = true;            // 启用 ML 扫描
    float mlConfidenceThreshold = 0.75f; // ML 置信度阈值
    size_t mlScanStep = 512;             // ML 扫描步长（字节）
    size_t mlBatchSize = 64;             // ML 批量推理大小
    bool prefilterEmpty = true;          // 预过滤空簇（熵检测）
    set<string> mlOnlyTypes;             // 仅用 ML 扫描的类型（txt, html, xml）

    // 默认构造函数：设置无签名类型
    HybridScanConfig() {
        mlOnlyTypes = {"txt", "html", "xml"};
    }
};

// 扫描统计信息
struct CarvingStats {
    atomic<ULONGLONG> totalClusters{0};
    atomic<ULONGLONG> scannedClusters{0};
    atomic<ULONGLONG> skippedClusters{0};      // 跳过的空簇
    atomic<ULONGLONG> filesFound{0};
    atomic<ULONGLONG> bytesRead{0};
    atomic<DWORD> elapsedMs{0};                // 耗时（毫秒）
    atomic<double> readSpeedMBps{0.0};         // 读取速度 MB/s
    atomic<double> scanSpeedMBps{0.0};         // 扫描速度 MB/s
    atomic<double> ioBusyPercent{0.0};         // I/O 忙碌百分比
    atomic<double> cpuBusyPercent{0.0};        // CPU 忙碌百分比

    // 重置方法
    void reset() {
        totalClusters = 0;
        scannedClusters = 0;
        skippedClusters = 0;
        filesFound = 0;
        bytesRead = 0;
        elapsedMs = 0;
        readSpeedMBps = 0.0;
        scanSpeedMBps = 0.0;
        ioBusyPercent = 0.0;
        cpuBusyPercent = 0.0;
    }
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

    // ==================== 线程池相关 ====================
    unique_ptr<SignatureScanThreadPool> scanThreadPool;
    bool useThreadPool;
    ScanThreadPoolConfig threadPoolConfig;

    // 将缓冲区分块并提交给线程池
    void SubmitBufferToThreadPool(const BYTE* buffer, size_t bufferSize,
                                   ULONGLONG baseLCN, ULONGLONG bytesPerCluster,
                                   int& taskIdCounter);

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

    // 获取签名数据库（供 FileCarverRecovery 使用）
    const map<string, FileSignature>& GetSignatures() const { return signatures; }

    // 异步扫描（双缓冲 + 生产者-消费者模式）
    vector<CarvedFileInfo> ScanForFileTypesAsync(const vector<string>& fileTypes,
                                                 CarvingMode mode = CARVE_SMART,
                                                 ULONGLONG maxResults = 1000);

    // ==================== 线程池扫描（新增） ====================

    // 线程池设置
    void SetThreadPoolMode(bool enabled) { useThreadPool = enabled; }
    bool IsThreadPoolMode() const { return useThreadPool; }

    // SIMD 设置（用于基准测试）
    void SetSimdEnabled(bool enabled);
    bool IsSimdEnabled() const;
    std::string GetSimdInfo() const;

    // 设置线程池配置
    void SetThreadPoolConfig(const ScanThreadPoolConfig& config);
    const ScanThreadPoolConfig& GetThreadPoolConfig() const { return threadPoolConfig; }

    // 使用线程池并行扫描（阶段1实现）
    // 特点：I/O线程读取 + 多Worker线程并行扫描
    vector<CarvedFileInfo> ScanForFileTypesThreadPool(const vector<string>& fileTypes,
                                                       CarvingMode mode = CARVE_SMART,
                                                       ULONGLONG maxResults = 10000);

    // ==================== 混合扫描模式（签名 + ML）====================

    // 混合扫描：签名扫描 + ML 扫描
    // - 有签名的类型（jpg, png, pdf...）使用签名扫描
    // - 无签名的类型（txt, html, xml）使用 ML 扫描
    // - 结果自动融合去重
    vector<CarvedFileInfo> ScanHybridMode(const vector<string>& fileTypes,
                                          const HybridScanConfig& config,
                                          CarvingMode mode = CARVE_SMART,
                                          ULONGLONG maxResults = 10000);

    // 估算文本文件大小（转发到 FileFormatUtils）
    static ULONGLONG EstimateFileSizeML(const BYTE* data, size_t maxSize, const string& type) {
        return FileFormatUtils::EstimateFileSizeML(data, maxSize, type);
    }

    // 快速熵计算（转发到 FileFormatUtils）
    static float QuickEntropy(const BYTE* data, size_t size) {
        return FileFormatUtils::QuickEntropy(data, size);
    }

    // 获取签名索引（供线程池使用）
    const unordered_map<BYTE, vector<const FileSignature*>>& GetSignatureIndex() const {
        return signatureIndex;
    }

    // 获取活动签名集合（供线程池使用）
    const set<string>& GetActiveSignatures() const { return activeSignatures; }

    // ==================== ML 分类（新增） ====================

    // 启用/禁用 ML 分类
    void SetMLClassification(bool enabled) { mlClassificationEnabled = enabled; }
    bool IsMLClassificationEnabled() const { return mlClassificationEnabled; }

    // 加载 ML 模型
    bool LoadMLModel(const wstring& modelPath);

    // 检查 ML 模型是否可用
    bool IsMLModelLoaded() const;

    // 使用 ML 分类单个文件片段
    // 返回: 预测的文件类型和置信度
    optional<ML::ClassificationResult> ClassifyWithML(
        const BYTE* data, size_t dataSize);

    // 为扫描结果补充 ML 分类
    // 当签名检测置信度较低时，使用 ML 提供辅助判断
    void EnhanceResultsWithML(vector<CarvedFileInfo>& results, bool showProgress = true);

    // 使用纯 ML 扫描（无签名扫描，直接使用 ML 分类）
    // 适用于检测签名库不支持的文件类型
    vector<CarvedFileInfo> ScanWithMLOnly(CarvingMode mode = CARVE_SMART,
                                          ULONGLONG maxResults = 1000,
                                          float minConfidence = 0.7f);

    // 获取 ML 支持的文件类型
    vector<string> GetMLSupportedTypes() const;

    // 获取默认ML分类模型搜索路径
    static vector<wstring> GetDefaultMLModelPaths();

    // 获取默认ML修复模型搜索路径
    static vector<wstring> GetDefaultRepairModelPaths();

private:
    // ML 分类相关成员
    bool mlClassificationEnabled;
    unique_ptr<ML::MLClassifier> mlClassifier;

    // 自动加载ML模型（在构造函数中调用）
    void AutoLoadMLModel();
};
