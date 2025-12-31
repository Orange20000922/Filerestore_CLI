#pragma once
#include <Windows.h>
#include <vector>
#include <string>
#include "MFTStructures.h"
#include "MFTReader.h"

using namespace std;

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

    // 文件信息提取
    wstring GetFileNameFromRecord(vector<BYTE>& mftRecord, ULONGLONG& parentDir, bool enableDebug = false);
    wstring GetFileNameFromAttribute(BYTE* attr);

    // 数据检查
    bool CheckDataAvailable(vector<BYTE>& mftRecord);
};
