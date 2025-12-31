#pragma once
#include <Windows.h>
#include <string>

using namespace std;

// MFT文件记录头结构
#pragma pack(push, 1)
typedef struct _FILE_RECORD_HEADER {
    DWORD Signature;           // "FILE"
    WORD UpdateSequenceOffset;
    WORD UpdateSequenceSize;
    ULONGLONG LogFileSequenceNumber;
    WORD SequenceNumber;
    WORD HardLinkCount;
    WORD FirstAttributeOffset;
    WORD Flags;                // 0x00 = 已删除, 0x01 = 使用中, 0x02 = 目录
    DWORD UsedSize;
    DWORD AllocatedSize;
    ULONGLONG FileReference;
    WORD NextAttributeId;
} FILE_RECORD_HEADER, *PFILE_RECORD_HEADER;

// 属性头结构
typedef struct _ATTRIBUTE_HEADER {
    DWORD Type;                // 属性类型
    DWORD Length;              // 属性长度
    BYTE NonResident;          // 0=常驻, 1=非常驻
    BYTE NameLength;
    WORD NameOffset;
    WORD Flags;
    WORD AttributeId;
} ATTRIBUTE_HEADER, *PATTRIBUTE_HEADER;

// 常驻属性
typedef struct _RESIDENT_ATTRIBUTE {
    DWORD ValueLength;
    WORD ValueOffset;
    BYTE Flags;
    BYTE Reserved;
} RESIDENT_ATTRIBUTE, *PRESIDENT_ATTRIBUTE;

// 非常驻属性
typedef struct _NONRESIDENT_ATTRIBUTE {
    ULONGLONG StartVCN;
    ULONGLONG EndVCN;
    WORD DataRunOffset;
    WORD CompressionUnit;
    DWORD Reserved;
    ULONGLONG AllocatedSize;
    ULONGLONG RealSize;
    ULONGLONG InitializedSize;
} NONRESIDENT_ATTRIBUTE, *PNONRESIDENT_ATTRIBUTE;

// 文件名属性
typedef struct _FILE_NAME_ATTRIBUTE {
    ULONGLONG ParentDirectory;
    ULONGLONG CreationTime;
    ULONGLONG ModificationTime;
    ULONGLONG MFTChangeTime;
    ULONGLONG LastAccessTime;
    ULONGLONG AllocatedSize;
    ULONGLONG RealSize;
    DWORD FileAttributes;
    DWORD ReparseTag;
    BYTE FileNameLength;
    BYTE NameType;
    WCHAR FileName[1];  // 可变长度
} FILE_NAME_ATTRIBUTE, *PFILE_NAME_ATTRIBUTE;

// 索引根属性
typedef struct _INDEX_ROOT {
    DWORD Type;
    DWORD CollationRule;
    DWORD IndexBlockSize;
    BYTE ClustersPerIndexBlock;
    BYTE Reserved[3];
    DWORD IndexEntryOffset;
    DWORD TotalEntrySize;
    DWORD AllocatedEntrySize;
    BYTE Flags;
    BYTE Reserved2[3];
} INDEX_ROOT, *PINDEX_ROOT;

// 索引条目
typedef struct _INDEX_ENTRY {
    ULONGLONG FileReference;
    WORD EntryLength;
    WORD StreamLength;
    BYTE Flags;
    BYTE Reserved[3];
    // 后面跟着FILE_NAME_ATTRIBUTE
} INDEX_ENTRY, *PINDEX_ENTRY;
#pragma pack(pop)

// 属性类型枚举
enum ATTRIBUTE_TYPE {
    AttributeStandardInformation = 0x10,
    AttributeFileName = 0x30,
    AttributeData = 0x80,
    AttributeIndexRoot = 0x90,
    AttributeIndexAllocation = 0xA0,
    AttributeEndOfList = 0xFFFFFFFF
};

// 索引条目标志
#define INDEX_ENTRY_FLAG_SUBNODE 0x01
#define INDEX_ENTRY_FLAG_LAST 0x02

// 已删除文件信息结构
struct DeletedFileInfo {
    ULONGLONG recordNumber;        // MFT记录号
    wstring fileName;              // 文件名
    wstring filePath;              // 完整路径（尽力重建）
    ULONGLONG fileSize;            // 文件大小
    ULONGLONG parentDirectory;     // 父目录记录号
    FILETIME deletionTime;         // 删除时间（修改时间）
    bool isDirectory;              // 是否为目录
    bool dataAvailable;            // 数据是否可用（未被覆盖）

    // 覆盖检测信息（扩展）
    bool overwriteDetected;        // 是否检测到覆盖
    double overwritePercentage;    // 覆盖百分比 (0.0-100.0)
    ULONGLONG totalClusters;       // 总簇数
    ULONGLONG availableClusters;   // 可用簇数
    ULONGLONG overwrittenClusters; // 被覆盖的簇数

    // 构造函数，初始化新增字段
    DeletedFileInfo() : recordNumber(0), fileSize(0), parentDirectory(0),
                       isDirectory(false), dataAvailable(false),
                       overwriteDetected(false), overwritePercentage(0.0),
                       totalClusters(0), availableClusters(0), overwrittenClusters(0) {
        deletionTime.dwLowDateTime = 0;
        deletionTime.dwHighDateTime = 0;
    }
};
