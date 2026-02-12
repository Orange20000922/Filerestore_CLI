#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <memory>
#include "CarvedFileTypes.h"
#include "FileFormatUtils.h"
#include "MFTLCNIndex.h"
#include "FileIntegrityValidator.h"

using namespace std;

class MFTReader;

// ============================================================================
// CarvedResultEnricher — 扫描结果后处理增强
// 对已扫描结果进行元数据增强（时间戳、完整性、删除状态）
// ============================================================================
class CarvedResultEnricher {
public:
    CarvedResultEnricher(MFTReader* reader);

    // ==================== 时间戳提取 ====================

    void SetTimestampExtraction(bool enabled) { timestampExtractionEnabled_ = enabled; }
    bool IsTimestampExtractionEnabled() const { return timestampExtractionEnabled_; }

    bool BuildMFTIndex(bool includeActiveFiles = false);
    bool IsMFTIndexBuilt() const { return mftIndexBuilt_; }
    size_t GetMFTIndexSize() const { return lcnIndex_ ? lcnIndex_->GetIndexSize() : 0; }

    void ExtractTimestampsForResults(vector<CarvedFileInfo>& results, bool showProgress = true);

    // ==================== 完整性验证 ====================

    void SetIntegrityValidation(bool enabled) { integrityValidationEnabled_ = enabled; }
    bool IsIntegrityValidationEnabled() const { return integrityValidationEnabled_; }

    FileIntegrityScore ValidateFileIntegrity(const CarvedFileInfo& info);
    void ValidateIntegrityForResults(vector<CarvedFileInfo>& results, bool showProgress = true);

    vector<CarvedFileInfo> FilterCorruptedFiles(const vector<CarvedFileInfo>& results,
                                                 double minIntegrityScore = 0.5);

    size_t GetValidatedCount() const { return validatedCount_; }
    size_t GetCorruptedCount() const { return corruptedCount_; }

    // ==================== 删除状态检查 ====================

    void CheckDeletionStatus(CarvedFileInfo& info);
    void CheckDeletionStatusForResults(vector<CarvedFileInfo>& results, bool showProgress = true);

    vector<CarvedFileInfo> FilterDeletedOnly(const vector<CarvedFileInfo>& results);
    vector<CarvedFileInfo> FilterActiveOnly(const vector<CarvedFileInfo>& results);

    size_t CountDeletedFiles(const vector<CarvedFileInfo>& results);
    size_t CountActiveFiles(const vector<CarvedFileInfo>& results);

private:
    MFTReader* reader_;

    // MFT 索引
    unique_ptr<MFTLCNIndex> lcnIndex_;
    bool timestampExtractionEnabled_;
    bool mftIndexBuilt_;

    // 完整性验证
    bool integrityValidationEnabled_;
    size_t validatedCount_;
    size_t corruptedCount_;

    // 为单个文件提取时间戳
    void ExtractTimestampForFile(CarvedFileInfo& info, const BYTE* fileData, size_t dataSize);

    // 提取文件数据
    bool ExtractFile(ULONGLONG startLCN, ULONGLONG startOffset,
                     ULONGLONG fileSize, vector<BYTE>& fileData);
};
