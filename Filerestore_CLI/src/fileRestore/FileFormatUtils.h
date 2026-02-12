#pragma once
#include <Windows.h>
#include <string>
#include <vector>

using namespace std;

// ============================================================================
// 文件签名定义（从 FileCarver.h 提取，供多模块共用）
// ============================================================================
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

// ============================================================================
// FileFormatUtils — 纯静态文件格式解析工具
// 无状态，线程安全，供 FileCarver、SignatureScanThreadPool 等共用
// ============================================================================
class FileFormatUtils {
public:
    // 快速熵计算（Shannon entropy，用于预过滤空簇）
    static float QuickEntropy(const BYTE* data, size_t size);

    // 查找文件尾（正向搜索）— 适用于 JPEG EOI 等文件尾在中间的格式
    static ULONGLONG FindFooterStatic(const BYTE* data, size_t dataSize,
                                      const vector<BYTE>& footer, ULONGLONG maxSearch);

    // 查找文件尾（反向搜索）— 适用于 PDF %%EOF 等文件尾在末尾的格式
    static ULONGLONG FindFooterReverseStatic(const BYTE* data, size_t dataSize,
                                             const vector<BYTE>& footer, ULONGLONG maxSearch);

    // 通过遍历 PNG chunk 结构查找 IEND，返回文件结束位置
    static ULONGLONG FindPngEndByChunksStatic(const BYTE* data, size_t dataSize);

    // 检测 ZIP 是否为 OOXML Office 文档 (DOCX/XLSX/PPTX)
    // 返回: "docx", "xlsx", "pptx", "ooxml"(通用), 或 ""(非 Office)
    static string DetectOOXMLTypeStatic(const BYTE* data, size_t dataSize);

    // 查找 ZIP EOCD（End of Central Directory），返回 EOCD 结束位置
    static ULONGLONG FindZipEndOfCentralDirectoryStatic(const BYTE* data, size_t dataSize);

    // 通过遍历 Local File Headers 估算 ZIP 大小（EOCD 不可用时的备选方案）
    static ULONGLONG EstimateZipSizeByHeaders(const BYTE* data, size_t dataSize,
                                               bool* outIsComplete = nullptr);

    // 综合估算文件大小 — 处理各种格式（ZIP EOCD, PDF EOF, BMP/AVI/WAV 头部等）
    // outIsComplete: 输出是否找到完整文件结构
    static ULONGLONG EstimateFileSizeStatic(const BYTE* data, size_t dataSize,
                                            const FileSignature& sig,
                                            ULONGLONG* outFooterPos = nullptr,
                                            bool* outIsComplete = nullptr);

    // 估算文本文件大小（用于 ML 检测的无签名文件，扫描到 NULL 序列）
    static ULONGLONG EstimateFileSizeML(const BYTE* data, size_t maxSize, const string& type);

    // 验证 ZIP 数据完整性（从内存）— 检查 EOCD、CD、CRC
    struct ZipValidationResult {
        bool success = false;
        ULONGLONG actualSize = 0;
        bool crcValid = false;
        int totalFiles = 0;
        int corruptedFiles = 0;
        string diagnosis;
    };
    static ZipValidationResult ValidateZipData(const BYTE* data, size_t dataSize);
};
