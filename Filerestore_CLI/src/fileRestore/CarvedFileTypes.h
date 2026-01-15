#pragma once
#include <Windows.h>
#include <string>

using namespace std;

// ============================================================================
// Carved File Types - 独立的类型定义，避免循环包含
// ============================================================================

// 时间戳来源
enum TimestampSource {
    TS_NONE_1 = 0,            // 未获取到时间戳
    TS_EMBEDDED,            // 从文件内嵌元数据提取
    TS_MFT_MATCH,           // 从MFT记录交叉匹配
    TS_BOTH                 // 两种来源都有
};

// Carving 结果
struct CarvedFileInfo {
    ULONGLONG startLCN;         // 起始逻辑簇号
    ULONGLONG startOffset;      // 簇内偏移
    ULONGLONG fileSize;         // 文件大小（估计或精确）
    string extension;           // 文件类型
    string description;         // 文件类型描述
    bool hasValidFooter;        // 是否找到有效的文件尾
    bool sizeIsEstimated;       // 大小是否为估计值（可能实际更大，需要进一步处理）
    double confidence;          // 置信度 (0.0-1.0)
    double validationScore;     // 验证评分 (0.0-1.0)

    // 时间戳信息
    FILETIME creationTime;      // 创建时间
    FILETIME modificationTime;  // 修改时间
    FILETIME accessTime;        // 访问时间
    TimestampSource tsSource;   // 时间戳来源
    ULONGLONG matchedMftRecord; // 匹配的MFT记录号（如果有）
    string embeddedInfo;        // 内嵌元数据信息（如EXIF相机型号）

    // 完整性验证信息
    double integrityScore;      // 完整性评分 (0-1)
    bool integrityValidated;    // 是否已验证
    string integrityDiagnosis;  // 完整性诊断信息

    // 删除状态检查
    bool isDeleted;             // 是否为已删除文件
    bool deletionChecked;       // 是否已检查删除状态
    bool isActiveFile;          // 是否为活动文件（未删除，可通过文件系统访问）

    // ML 分类信息
    string mlClassification;    // ML 预测的文件类型
    float mlConfidence;         // ML 预测置信度 (0-1)

    CarvedFileInfo() : startLCN(0), startOffset(0), fileSize(0),
                       hasValidFooter(false), sizeIsEstimated(false),
                       confidence(0.0), validationScore(0.0),
                       tsSource(TS_NONE_1), matchedMftRecord(0),
                       integrityScore(0.0), integrityValidated(false),
                       isDeleted(false), deletionChecked(false), isActiveFile(false),
                       mlConfidence(0.0f) {
        creationTime.dwLowDateTime = 0;
        creationTime.dwHighDateTime = 0;
        modificationTime.dwLowDateTime = 0;
        modificationTime.dwHighDateTime = 0;
        accessTime.dwLowDateTime = 0;
        accessTime.dwHighDateTime = 0;
    }
};
