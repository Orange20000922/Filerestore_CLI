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
#include "TimestampExtractor.h"
#include "MFTLCNIndex.h"
#include "FileIntegrityValidator.h"
#include "CarvedFileTypes.h"
#include "MLClassifier.h"

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

    // 查找文件尾（从前向后搜索）
    ULONGLONG FindFooter(const BYTE* data, size_t dataSize,
                        const vector<BYTE>& footer, ULONGLONG maxSearch);

    // 查找文件尾（从后向前搜索，适用于文件尾在末尾的格式如 ZIP/PDF）
    ULONGLONG FindFooterReverse(const BYTE* data, size_t dataSize,
                               const vector<BYTE>& footer, ULONGLONG maxSearch);

    // 查找 ZIP 文件尾（EOCD，处理注释字段）
    ULONGLONG FindZipEndOfCentralDirectory(const BYTE* data, size_t dataSize);

    // 通过遍历 PNG chunk 结构查找文件末尾
    ULONGLONG FindPngEndByChunks(const BYTE* data, size_t dataSize);

    // 检测 ZIP 是否为 OOXML Office 文档 (DOCX/XLSX/PPTX)
    // 返回: "docx", "xlsx", "pptx", "ooxml"(通用), 或 ""(非 Office)
    string DetectOOXMLType(const BYTE* data, size_t dataSize);

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

    // ==================== 线程池相关 ====================
    unique_ptr<SignatureScanThreadPool> scanThreadPool;
    bool useThreadPool;
    ScanThreadPoolConfig threadPoolConfig;

    // 将缓冲区分块并提交给线程池
    void SubmitBufferToThreadPool(const BYTE* buffer, size_t bufferSize,
                                   ULONGLONG baseLCN, ULONGLONG bytesPerCluster,
                                   int& taskIdCounter);

    // ==================== 时间戳提取相关 ====================
    unique_ptr<MFTLCNIndex> lcnIndex;
    bool timestampExtractionEnabled;
    bool mftIndexBuilt;

    // 为单个文件提取时间戳
    void ExtractTimestampForFile(CarvedFileInfo& info, const BYTE* fileData, size_t dataSize);

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

    // ==================== 线程池扫描（新增） ====================

    // 线程池设置
    void SetThreadPoolMode(bool enabled) { useThreadPool = enabled; }
    bool IsThreadPoolMode() const { return useThreadPool; }

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

    // 估算文本文件大小（用于 ML 检测的无签名文件）
    static ULONGLONG EstimateFileSizeML(const BYTE* data, size_t maxSize, const string& type);

    // 快速熵计算（用于预过滤）
    static float QuickEntropy(const BYTE* data, size_t size);

    // 获取签名索引（供线程池使用）
    const unordered_map<BYTE, vector<const FileSignature*>>& GetSignatureIndex() const {
        return signatureIndex;
    }

    // 获取活动签名集合（供线程池使用）
    const set<string>& GetActiveSignatures() const { return activeSignatures; }

    // ==================== 静态辅助函数（供线程池使用，线程安全）====================

    // 查找 ZIP 文件尾（EOCD）- 静态版本，线程安全
    // 返回 EOCD 结束位置（即 ZIP 文件大小），未找到返回 0
    static ULONGLONG FindZipEndOfCentralDirectoryStatic(const BYTE* data, size_t dataSize);

    // 通过遍历 PNG chunk 结构查找文件末尾 - 静态版本，线程安全
    static ULONGLONG FindPngEndByChunksStatic(const BYTE* data, size_t dataSize);

    // 检测 ZIP 是否为 OOXML Office 文档 - 静态版本，线程安全
    // 返回: "docx", "xlsx", "pptx", "ooxml"(通用), 或 ""(非 Office)
    static string DetectOOXMLTypeStatic(const BYTE* data, size_t dataSize);

    // 通过遍历 Local File Headers 估算 ZIP 大小 - 静态版本，线程安全
    // 当找不到 EOCD 时使用此方法获得更准确的估计
    // 返回值: 估算大小, outIsComplete 表示是否在数据范围内找到了完整结构
    static ULONGLONG EstimateZipSizeByHeaders(const BYTE* data, size_t dataSize,
                                               bool* outIsComplete = nullptr);

    // 查找文件尾（正向搜索）- 静态版本，线程安全
    // 适用于 JPEG, PNG, GIF 等文件尾在文件中间的格式
    static ULONGLONG FindFooterStatic(const BYTE* data, size_t dataSize,
                                      const vector<BYTE>& footer, ULONGLONG maxSearch);

    // 查找文件尾（反向搜索）- 静态版本，线程安全
    // 适用于 PDF 等文件尾在文件末尾的格式
    static ULONGLONG FindFooterReverseStatic(const BYTE* data, size_t dataSize,
                                             const vector<BYTE>& footer, ULONGLONG maxSearch);

    // 估算文件大小 - 静态版本，线程安全
    // 综合处理各种格式（ZIP EOCD, PDF EOF, BMP/AVI/WAV 头部大小等）
    // outIsComplete: 输出参数，表示是否找到了完整的文件结构（对于大文件跨chunk处理有用）
    static ULONGLONG EstimateFileSizeStatic(const BYTE* data, size_t dataSize,
                                            const FileSignature& sig,
                                            ULONGLONG* outFooterPos = nullptr,
                                            bool* outIsComplete = nullptr);

    // ==================== 时间戳提取（新增） ====================

    // 启用/禁用时间戳提取
    void SetTimestampExtraction(bool enabled) { timestampExtractionEnabled = enabled; }
    bool IsTimestampExtractionEnabled() const { return timestampExtractionEnabled; }

    // 构建 MFT LCN 索引（用于时间戳交叉匹配）
    // 在扫描前调用一次，可显著提高时间戳匹配效率
    bool BuildMFTIndex(bool includeActiveFiles = false);

    // 为扫描结果批量提取时间戳
    // 会自动尝试：1. 内嵌元数据提取  2. MFT 记录匹配
    void ExtractTimestampsForResults(vector<CarvedFileInfo>& results, bool showProgress = true);

    // 获取 MFT 索引状态
    bool IsMFTIndexBuilt() const { return mftIndexBuilt; }
    size_t GetMFTIndexSize() const { return lcnIndex ? lcnIndex->GetIndexSize() : 0; }

    // ==================== 完整性验证（新增） ====================

    // 启用/禁用完整性验证
    void SetIntegrityValidation(bool enabled) { integrityValidationEnabled = enabled; }
    bool IsIntegrityValidationEnabled() const { return integrityValidationEnabled; }

    // 为单个文件验证完整性
    FileIntegrityScore ValidateFileIntegrity(const CarvedFileInfo& info);

    // 为扫描结果批量验证完整性
    void ValidateIntegrityForResults(vector<CarvedFileInfo>& results, bool showProgress = true);

    // 过滤出可能损坏的文件
    vector<CarvedFileInfo> FilterCorruptedFiles(const vector<CarvedFileInfo>& results,
                                                 double minIntegrityScore = 0.5);

    // 获取完整性验证统计
    size_t GetValidatedCount() const { return validatedCount; }
    size_t GetCorruptedCount() const { return corruptedCount; }

    // ==================== 删除状态检查（新增） ====================

    // 检查单个文件的删除状态
    // 通过 MFT LCN 索引交叉验证，判断文件是否为已删除状态
    void CheckDeletionStatus(CarvedFileInfo& info);

    // 批量检查删除状态
    void CheckDeletionStatusForResults(vector<CarvedFileInfo>& results, bool showProgress = true);

    // 过滤出已删除的文件
    vector<CarvedFileInfo> FilterDeletedOnly(const vector<CarvedFileInfo>& results);

    // 过滤出活动文件（未删除）
    vector<CarvedFileInfo> FilterActiveOnly(const vector<CarvedFileInfo>& results);

    // 获取已删除文件数量统计
    size_t CountDeletedFiles(const vector<CarvedFileInfo>& results);
    size_t CountActiveFiles(const vector<CarvedFileInfo>& results);

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

    // ==================== 基于结构的大文件恢复（推荐）====================

    // ZIP 文件恢复配置
    struct ZipRecoveryConfig {
        ULONGLONG maxSize = 50ULL * 1024 * 1024 * 1024;  // 最大搜索大小 (默认 50GB)
        ULONGLONG expectedSize = 0;                      // 用户预期大小 (0 = 不限制)
        ULONGLONG expectedSizeTolerance = 0;             // 大小容差 (0 = 自动 10%)
        bool verifyCRC = true;                           // 恢复后验证 CRC
        bool stopOnFirstEOCD = true;                     // 找到第一个 EOCD 就停止
        bool allowFragmented = false;                    // 允许碎片化文件（跳过无效块）
    };

    // ZIP 恢复结果
    struct ZipRecoveryResult {
        bool success = false;           // 是否成功找到 EOCD
        ULONGLONG actualSize = 0;       // 实际文件大小
        ULONGLONG bytesWritten = 0;     // 写入的字节数
        bool crcValid = false;          // CRC 是否有效
        int totalFiles = 0;             // ZIP 内文件数量
        int corruptedFiles = 0;         // 损坏的文件数量
        bool isFragmented = false;      // 是否检测到碎片化
        string diagnosis;               // 诊断信息
    };

    // 扫描到 EOCD 恢复 ZIP 文件（推荐方法）
    // 从 startLCN 开始扫描，直到找到有效的 EOCD 或达到 maxSize
    ZipRecoveryResult RecoverZipWithEOCDScan(
        ULONGLONG startLCN,
        const string& outputPath,
        const ZipRecoveryConfig& config = ZipRecoveryConfig()
    );

    // 验证已恢复的 ZIP 文件完整性
    // 检查: EOCD 结构, Central Directory, 各文件 CRC
    static ZipRecoveryResult ValidateZipFile(const string& filePath);

    // 验证 ZIP 数据完整性（从内存）
    static ZipRecoveryResult ValidateZipData(const BYTE* data, size_t dataSize);

private:
    // 完整性验证相关成员
    bool integrityValidationEnabled;
    size_t validatedCount;
    size_t corruptedCount;

    // ML 分类相关成员
    bool mlClassificationEnabled;
    unique_ptr<ML::MLClassifier> mlClassifier;

    // 自动加载ML模型（在构造函数中调用）
    void AutoLoadMLModel();
};
