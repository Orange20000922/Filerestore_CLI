#pragma once
#include <Windows.h>
#include <vector>
#include <string>
#include <optional>
#include "MFTStructures.h"
#include "MFTReader.h"

using namespace std;

// ============================================================================
// MFT 文件数据信息 - 从 MFT 记录提取的完整数据定位信息
// ============================================================================
struct FileDataInfo {
    vector<pair<ULONGLONG, ULONGLONG>> dataRuns;  // (LCN, clusterCount)
    ULONGLONG fileSize;
    bool isResident;
    vector<BYTE> residentData;
    WORD sequenceNumber;

    FileDataInfo() : fileSize(0), isResident(false), sequenceNumber(0) {}
};

// MFT 解析器类 - 负责解析 MFT 记录内容
class MFTParser
{
private:
    MFTReader* reader;

public:
    MFTParser(MFTReader* mftReader);
    ~MFTParser();

    // 属性解析
    bool ParseDataRuns(BYTE* dataRun, vector<pair<ULONGLONG, ULONGLONG>>& runs);
    bool ExtractFileData(vector<BYTE>& mftRecord, vector<BYTE>& fileData);
    bool GetIndexRoot(vector<BYTE>& mftRecord, vector<BYTE>& indexData);

    // 从 MFT 记录缓冲区提取完整文件数据信息（Data Runs、文件大小、驻留状态等）
    // 只解析未命名的 $DATA 属性（跳过 ADS）
    // 返回 nullopt 如果记录无效或没有 $DATA 属性
    optional<FileDataInfo> ExtractFileDataInfo(BYTE* recordBuffer, size_t recordSize);

    // 文件信息提取
    wstring GetFileNameFromRecord(vector<BYTE>& mftRecord, ULONGLONG& parentDir, bool enableDebug = false);
    wstring GetFileNameFromAttribute(BYTE* attr);

    // 数据检查
    bool CheckDataAvailable(vector<BYTE>& mftRecord);
};
