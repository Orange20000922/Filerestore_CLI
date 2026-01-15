#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <functional>
#include <optional>
#include "UsnJournalReader.h"
#include "MFTReader.h"
#include "MFTParser.h"
#include "CarvedFileTypes.h"

using namespace std;

// ============================================================================
// USN 定点恢复 - 结合 USN 日志和签名验证的精准文件恢复
// ============================================================================

// 定点恢复验证结果
enum class UsnRecoveryStatus {
    SUCCESS,                    // 成功恢复
    MFT_RECORD_REUSED,          // MFT 记录已被重用（序列号不匹配），数据不可信
    MFT_REUSED_DATA_VALID,      // MFT 记录被重用，但签名验证通过，数据可能有效
    MFT_RECORD_NOT_FOUND,       // MFT 记录不存在
    NO_DATA_ATTRIBUTE,          // 没有数据属性（可能是目录或特殊文件）
    NO_DATA_SIGNATURE_SCAN,     // 无 MFT 数据，但签名扫描找到匹配
    DATA_OVERWRITTEN,           // 数据已被覆盖（签名不匹配）
    SIGNATURE_MISMATCH,         // 签名不匹配
    PARTIAL_RECOVERY,           // 部分恢复（碎片化文件部分数据丢失）
    READ_ERROR,                 // 读取错误
    WRITE_ERROR,                // 写入错误
    RESIDENT_DATA,              // 常驻数据（小文件，数据在 MFT 记录内）
    UNKNOWN_ERROR               // 未知错误
};

// USN 定点恢复结果
struct UsnTargetedRecoveryResult {
    // USN 元数据
    UsnDeletedFileInfo usnInfo;

    // MFT 解析结果
    ULONGLONG mftRecordNumber;                          // MFT 记录号
    WORD expectedSequence;                              // 期望的序列号（来自 USN）
    WORD actualSequence;                                // 实际的序列号（来自 MFT）
    bool sequenceMatched;                               // 序列号是否匹配

    // 数据定位
    vector<pair<ULONGLONG, ULONGLONG>> dataRuns;        // Data Runs: (LCN, 簇数)
    ULONGLONG totalClusters;                            // 总簇数
    ULONGLONG fileSize;                                 // 文件大小（来自 MFT）
    bool isResident;                                    // 是否为常驻数据
    vector<BYTE> residentData;                          // 常驻数据内容

    // 签名验证
    bool signatureMatched;                              // 签名是否匹配
    string expectedType;                                // 期望的文件类型（根据扩展名）
    string detectedType;                                // 检测到的文件类型
    double confidence;                                  // 置信度 (0.0 - 1.0)

    // 恢复状态
    UsnRecoveryStatus status;                           // 恢复状态
    string statusMessage;                               // 状态描述
    bool canRecover;                                    // 是否可恢复

    // 恢复后信息
    wstring recoveredPath;                              // 恢复后的文件路径
    ULONGLONG recoveredSize;                            // 恢复的字节数

    // 构造函数
    UsnTargetedRecoveryResult() :
        mftRecordNumber(0), expectedSequence(0), actualSequence(0),
        sequenceMatched(false), totalClusters(0), fileSize(0),
        isResident(false), signatureMatched(false), confidence(0.0),
        status(UsnRecoveryStatus::UNKNOWN_ERROR), canRecover(false),
        recoveredSize(0) {}
};

// USN 文件列表项（带验证状态）
struct UsnFileListItem {
    UsnDeletedFileInfo usnInfo;                         // USN 信息
    bool validated;                                     // 是否已验证
    bool canRecover;                                    // 是否可恢复
    string detectedType;                                // 检测到的类型
    double confidence;                                  // 置信度
    UsnRecoveryStatus status;                           // 状态
    string statusMessage;                               // 状态消息

    UsnFileListItem() : validated(false), canRecover(false),
                       confidence(0.0), status(UsnRecoveryStatus::UNKNOWN_ERROR) {}
};

// 进度回调类型
using UsnRecoveryProgressCallback = function<void(size_t current, size_t total, const wstring& fileName)>;

// ============================================================================
// USN 定点恢复类
// ============================================================================
class UsnTargetedRecovery {
private:
    MFTReader* reader;
    MFTParser* parser;
    UsnJournalReader usnReader;

    // 签名数据库（简化版，用于快速验证）
    struct SimpleSignature {
        string extension;
        vector<BYTE> header;
        size_t minSize;
    };
    vector<SimpleSignature> signatures;

    // 初始化签名
    void InitializeSignatures();

    // 从扩展名获取期望的签名
    optional<SimpleSignature> GetSignatureForExtension(const wstring& fileName);

    // 验证数据签名
    bool ValidateSignature(const BYTE* data, size_t dataSize,
                          const SimpleSignature& sig, double& confidence);

    // 检测文件类型
    string DetectFileType(const BYTE* data, size_t dataSize, double& confidence);

    // 解析 MFT 记录获取数据运行
    bool ParseMFTRecordForDataRuns(ULONGLONG recordNumber,
                                   vector<pair<ULONGLONG, ULONGLONG>>& dataRuns,
                                   ULONGLONG& fileSize,
                                   bool& isResident,
                                   vector<BYTE>& residentData,
                                   WORD& sequenceNumber);

    // 从 Data Runs 读取文件数据
    bool ReadFileFromDataRuns(const vector<pair<ULONGLONG, ULONGLONG>>& dataRuns,
                              ULONGLONG fileSize, vector<BYTE>& fileData);

    // 保存文件
    bool SaveFile(const vector<BYTE>& data, const wstring& outputPath);

    // 获取状态消息
    string GetStatusMessage(UsnRecoveryStatus status);

public:
    UsnTargetedRecovery(MFTReader* mftReader, MFTParser* mftParser);
    ~UsnTargetedRecovery();

    // ==================== 核心功能 ====================

    // 验证单个 USN 记录的可恢复性
    UsnTargetedRecoveryResult Validate(const UsnDeletedFileInfo& usnInfo);

    // 恢复单个文件
    UsnTargetedRecoveryResult Recover(const UsnDeletedFileInfo& usnInfo,
                                      const wstring& outputDir,
                                      bool forceRecover = false);

    // 批量验证
    vector<UsnFileListItem> ValidateBatch(const vector<UsnDeletedFileInfo>& usnFiles,
                                          UsnRecoveryProgressCallback progressCallback = nullptr);

    // 批量恢复
    vector<UsnTargetedRecoveryResult> RecoverBatch(const vector<UsnDeletedFileInfo>& usnFiles,
                                                   const wstring& outputDir,
                                                   bool forceRecover = false,
                                                   UsnRecoveryProgressCallback progressCallback = nullptr);

    // ==================== USN 搜索辅助 ====================

    // 搜索最近删除的文件并验证
    vector<UsnFileListItem> SearchAndValidate(char driveLetter,
                                               int maxTimeHours = 24,
                                               const wstring& namePattern = L"",
                                               size_t maxResults = 1000);

    // 根据文件名搜索并恢复
    UsnTargetedRecoveryResult SearchAndRecover(char driveLetter,
                                                const wstring& fileName,
                                                const wstring& outputDir,
                                                bool forceRecover = false);

    // ==================== 工具方法 ====================

    // 获取文件扩展名
    static wstring GetExtension(const wstring& fileName);

    // 宽字符转窄字符
    static string WideToNarrow(const wstring& wide);

    // 窄字符转宽字符
    static wstring NarrowToWide(const string& narrow);

    // 格式化文件大小
    static wstring FormatFileSize(ULONGLONG size);

    // 格式化时间戳
    static wstring FormatTimestamp(const LARGE_INTEGER& timestamp);

    // 获取状态字符串
    static string GetStatusString(UsnRecoveryStatus status);
};
