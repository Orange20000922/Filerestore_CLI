#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <optional>

using namespace std;

// ============================================================================
// ML 文件修复 - 使用机器学习修复轻微损坏的文件
// ============================================================================

// 损坏类型
enum class DamageType {
    NONE,                   // 无损坏
    HEADER_CORRUPTED,       // 头部损坏
    STRUCTURE_CORRUPTED,    // 结构损坏（索引、目录等）
    CONTENT_CORRUPTED,      // 内容损坏
    UNKNOWN                 // 未知损坏
};

// 修复类型
enum class RepairType {
    HEADER_REBUILD,         // 头部重建
    STRUCTURE_REPAIR,       // 结构修复
    CONTENT_INTERPOLATE,    // 内容插值
    AUTO                    // 自动选择
};

// 修复结果
enum class RepairResult {
    SUCCESS,                // 完全修复
    PARTIAL,                // 部分修复
    FAILED,                 // 无法修复
    NOT_APPLICABLE          // 不适用（文件类型不支持）
};

// 损坏分析结果
struct DamageAnalysis {
    DamageType type;                        // 损坏类型
    double severity;                        // 严重程度 (0.0-1.0)
    bool isRepairable;                      // 是否可修复
    string description;                     // 损坏描述
    vector<pair<size_t, size_t>> damagedRanges; // 损坏区域 [(offset, length)]

    DamageAnalysis() : type(DamageType::UNKNOWN), severity(0.0),
                      isRepairable(false) {}
};

// 修复报告
struct RepairReport {
    RepairResult result;                    // 修复结果
    double confidence;                      // 修复置信度 (0.0-1.0)
    vector<string> repairActions;           // 执行的修复操作
    size_t bytesModified;                   // 修改的字节数
    vector<pair<size_t, size_t>> modifiedRanges; // 修改区域
    string message;                         // 详细信息

    RepairReport() : result(RepairResult::NOT_APPLICABLE), confidence(0.0),
                    bytesModified(0) {}
};

// ============================================================================
// 文件类型修复器基类
// ============================================================================
class FileTypeRepairer {
public:
    virtual ~FileTypeRepairer() = default;

    // 分析损坏情况
    virtual DamageAnalysis AnalyzeDamage(const vector<BYTE>& data) = 0;

    // 尝试修复
    virtual RepairReport TryRepair(vector<BYTE>& data, RepairType type) = 0;

    // 获取支持的文件类型
    virtual vector<string> GetSupportedTypes() const = 0;
};

// ============================================================================
// ML 文件修复主类
// ============================================================================
class MLFileRepair {
private:
    // 文件类型修复器映射
    map<string, unique_ptr<FileTypeRepairer>> repairers;

    // 注册修复器
    void RegisterRepairer(unique_ptr<FileTypeRepairer> repairer);

public:
    MLFileRepair();
    ~MLFileRepair();

    // 初始化（加载模型）
    bool Initialize(const string& modelsDir);

    // 分析文件损坏情况
    DamageAnalysis AnalyzeDamage(const vector<BYTE>& data,
                                  const string& expectedType);

    // 尝试修复文件
    RepairReport TryRepair(vector<BYTE>& data,
                           const string& expectedType,
                           RepairType repairType = RepairType::AUTO);

    // 检查是否支持某个文件类型
    bool IsSupportedType(const string& fileType) const;

    // 获取所有支持的文件类型
    vector<string> GetSupportedTypes() const;

    // 静态工具方法

    // 检测文件是否损坏
    static bool IsFileCorrupted(const vector<BYTE>& data, const string& fileType);

    // 获取文件类型扩展名
    static string GetExtension(const string& fileType);
};
