#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "CarvedFileTypes.h"
#include "MFTReader.h"
#include "MFTParser.h"
#include "MFTLCNIndex.h"
#include "UsnJournalReader.h"

using namespace std;

// ============================================================================
// 验证级别枚举
// ============================================================================
enum ValidationLevel {
    VAL_NONE = 0,           // 无法验证
    VAL_SIGNATURE_ONLY,     // 仅签名（最低置信度）
    VAL_MFT_SIGNATURE,      // MFT + 签名
    VAL_USN_SIGNATURE,      // USN + 签名
    VAL_USN_MFT,            // USN + MFT（数据可能被覆盖）
    VAL_TRIPLE              // 三角验证（最高置信度）
};

// ============================================================================
// 数据运行结构（LCN + 簇数）
// ============================================================================
struct DataRun {
    ULONGLONG lcn;          // 逻辑簇号
    ULONGLONG clusterCount; // 簇数量

    DataRun() : lcn(0), clusterCount(0) {}
    DataRun(ULONGLONG l, ULONGLONG c) : lcn(l), clusterCount(c) {}
};

// ============================================================================
// 三角验证结果
// ============================================================================
struct TripleValidationResult {
    // ========== 来源标识 ==========
    bool hasUsnSource;              // 是否有USN来源
    bool hasMftSource;              // 是否有MFT来源
    bool hasCarvedSource;           // 是否有签名扫描来源

    // ========== 关联的记录 ==========
    ULONGLONG mftRecordNumber;      // MFT记录号
    ULONGLONG startLCN;             // 起始LCN
    WORD expectedSequence;          // USN中的期望序列号
    WORD actualSequence;            // MFT中的实际序列号

    // ========== 验证状态 ==========
    bool sequenceValid;             // MFT序列号验证通过
    bool signatureValid;            // 文件头签名验证通过
    bool typeMatched;               // 文件类型匹配（扩展名 vs 签名检测）
    bool timestampMatched;          // 时间戳一致（容差1分钟）
    bool sizeMatched;               // 文件大小一致（容差5%）
    bool lcnMatched;                // LCN位置匹配

    // ========== 置信度 ==========
    double confidence;              // 综合置信度 0.0 - 1.0
    ValidationLevel level;          // 验证级别

    // ========== 最优恢复参数（综合三个来源的最可靠数据）==========
    ULONGLONG exactFileSize;        // 精确文件大小（优先MFT）
    vector<DataRun> dataRuns;       // 数据运行列表（支持碎片重组）
    wstring fileName;               // 文件名（优先USN）
    string detectedExtension;       // 签名检测的扩展名
    FILETIME creationTime;
    FILETIME modificationTime;
    FILETIME accessTime;

    // ========== 诊断信息 ==========
    string diagnosis;               // 诊断描述
    bool canRecover;                // 是否可恢复
    bool isFragmented;              // 是否碎片化

    // 构造函数
    TripleValidationResult() :
        hasUsnSource(false), hasMftSource(false), hasCarvedSource(false),
        mftRecordNumber(0), startLCN(0), expectedSequence(0), actualSequence(0),
        sequenceValid(false), signatureValid(false), typeMatched(false),
        timestampMatched(false), sizeMatched(false), lcnMatched(false),
        confidence(0.0), level(VAL_NONE), exactFileSize(0),
        canRecover(false), isFragmented(false) {
        ZeroMemory(&creationTime, sizeof(FILETIME));
        ZeroMemory(&modificationTime, sizeof(FILETIME));
        ZeroMemory(&accessTime, sizeof(FILETIME));
    }
};

// ============================================================================
// USN删除记录（简化版，用于验证）
// ============================================================================
struct UsnDeletedRecord {
    ULONGLONG fileReferenceNumber;  // 包含MFT记录号和序列号
    wstring fileName;
    ULONGLONG fileSize;             // 如果可用
    ULONGLONG parentDirectory;
    LARGE_INTEGER timestamp;        // USN时间戳
    vector<DataRun> dataRuns;       // 从MFT提取的数据运行

    // 提取MFT记录号（低48位）
    ULONGLONG GetMftRecordNumber() const {
        return fileReferenceNumber & 0x0000FFFFFFFFFFFFULL;
    }

    // 提取序列号（高16位）
    WORD GetExpectedSequence() const {
        return (WORD)(fileReferenceNumber >> 48);
    }
};

// ============================================================================
// 三角验证器类
// ============================================================================
class TripleValidator {
private:
    MFTReader* reader;                          // MFT读取器（不拥有）
    MFTParser* parser;                          // MFT解析器（不拥有）
    unique_ptr<MFTLCNIndex> lcnIndex;           // LCN索引（拥有）

    // USN删除记录缓存（按MFT记录号索引）
    map<ULONGLONG, UsnDeletedRecord> usnRecordsByMft;

    // USN删除记录缓存（按起始LCN索引）
    multimap<ULONGLONG, ULONGLONG> usnRecordsByLcn;  // LCN -> MFT记录号

    // 签名扫描结果缓存（按起始LCN索引）
    multimap<ULONGLONG, CarvedFileInfo*> carvedByLcn;

    // 配置参数
    double timestampToleranceSeconds;           // 时间戳容差（秒）
    double sizeTolerance;                       // 大小容差（百分比）
    ULONGLONG bytesPerCluster;                  // 每簇字节数

    // 私有方法
    bool ParseMftDataRuns(ULONGLONG recordNumber, vector<DataRun>& outRuns,
                          ULONGLONG& outFileSize, bool& outIsResident);
    bool ValidateSignature(const BYTE* data, size_t size, const string& expectedType,
                          string& detectedType, double& signatureConfidence);
    double CalculateConfidence(const TripleValidationResult& result);
    ValidationLevel DetermineLevel(const TripleValidationResult& result);
    bool CompareTimestamps(const FILETIME& ft1, const FILETIME& ft2, double toleranceSeconds);
    wstring GetExtensionFromFileName(const wstring& fileName);

public:
    TripleValidator(MFTReader* mftReader, MFTParser* mftParser);
    ~TripleValidator();

    // ========== 初始化 ==========

    // 构建MFT LCN索引（扫描所有/仅删除记录）
    bool BuildLcnIndex(bool deletedOnly = true, bool showProgress = true);

    // 加载USN删除记录
    bool LoadUsnDeletedRecords(const vector<UsnDeletedFileInfo>& usnRecords);

    // 加载签名扫描结果
    bool LoadCarvedResults(vector<CarvedFileInfo>& carvedResults);

    // ========== 三角验证 ==========

    // 验证单个签名扫描结果
    TripleValidationResult ValidateCarvedFile(CarvedFileInfo& carved);

    // 验证单个USN删除记录
    TripleValidationResult ValidateUsnRecord(const UsnDeletedRecord& usn);

    // 批量验证签名扫描结果
    vector<TripleValidationResult> ValidateCarvedFiles(vector<CarvedFileInfo>& carvedFiles,
                                                       bool showProgress = true);

    // 批量验证USN删除记录
    vector<TripleValidationResult> ValidateUsnRecords(bool showProgress = true);

    // ========== 交叉匹配 ==========

    // 查找与签名扫描结果匹配的USN记录
    bool FindMatchingUsn(ULONGLONG lcn, UsnDeletedRecord& outUsn);

    // 查找与USN记录匹配的签名扫描结果
    bool FindMatchingCarved(ULONGLONG mftRecordNumber, CarvedFileInfo*& outCarved);

    // 查找LCN对应的MFT记录信息
    bool FindMftInfoByLcn(ULONGLONG lcn, LCNMappingInfo& outInfo);

    // ========== 统计 ==========

    size_t GetUsnRecordCount() const { return usnRecordsByMft.size(); }
    size_t GetCarvedResultCount() const { return carvedByLcn.size(); }
    size_t GetLcnIndexSize() const { return lcnIndex ? lcnIndex->GetIndexSize() : 0; }

    // ========== 配置 ==========

    void SetTimestampTolerance(double seconds) { timestampToleranceSeconds = seconds; }
    void SetSizeTolerance(double percent) { sizeTolerance = percent; }

    // ========== 工具方法 ==========

    static string ValidationLevelToString(ValidationLevel level);
    static string FormatConfidence(double confidence);
};
