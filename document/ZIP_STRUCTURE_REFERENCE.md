# ZIP 文件尾部结构详解

## 1. End of Central Directory Record (EOCD)

**签名**: `50 4B 05 06` (0x06054B50)

**搜索方法**: 从文件末尾向前搜索（最多65535+22字节，因为comment最长65535）

```
偏移  大小  字段名                    说明
────────────────────────────────────────────────────────────
0     4     signature                 0x06054B50 (50 4B 05 06)
4     2     disk_number               当前磁盘编号（通常为0）
6     2     disk_cd_start             Central Directory开始的磁盘编号
8     2     cd_entries_this_disk      本磁盘上的CD条目数
10    2     cd_entries_total          CD条目总数
12    4     cd_size                   Central Directory总大小（字节）
16    4     cd_offset                 Central Directory相对于文件开头的偏移
20    2     comment_length            ZIP文件注释长度
22    var   comment                   ZIP文件注释（可选）
────────────────────────────────────────────────────────────
总计: 22 + comment_length 字节
```

### C++ 结构体

```cpp
#pragma pack(push, 1)
struct EOCD {
    uint32_t signature;           // 0x06054B50
    uint16_t diskNumber;          // 当前磁盘号
    uint16_t diskCDStart;         // CD开始的磁盘号
    uint16_t cdEntriesThisDisk;   // 本磁盘CD条目数
    uint16_t cdEntriesTotal;      // CD条目总数
    uint32_t cdSize;              // CD总大小
    uint32_t cdOffset;            // CD偏移（从文件开头）
    uint16_t commentLength;       // 注释长度
    // comment[commentLength] 紧随其后
};
#pragma pack(pop)

static_assert(sizeof(EOCD) == 22, "EOCD size must be 22 bytes");
```

---

## 2. Central Directory File Header

**签名**: `50 4B 01 02` (0x02014B50)

**位置**: 由EOCD的`cd_offset`指定

```
偏移  大小  字段名                    说明
────────────────────────────────────────────────────────────
0     4     signature                 0x02014B50 (50 4B 01 02)
4     2     version_made_by           创建版本
6     2     version_needed            解压所需最低版本
8     2     flags                     通用标志位
10    2     compression               压缩方法（0=存储, 8=Deflate）
12    2     mod_time                  最后修改时间（DOS格式）
14    2     mod_date                  最后修改日期（DOS格式）
16    4     crc32                     CRC-32校验值
20    4     compressed_size           压缩后大小
24    4     uncompressed_size         原始大小
28    2     filename_length           文件名长度
30    2     extra_length              扩展字段长度
32    2     comment_length            文件注释长度
34    2     disk_start                文件开始的磁盘号
36    2     internal_attr             内部文件属性
38    4     external_attr             外部文件属性
42    4     local_header_offset       对应Local Header的偏移
────────────────────────────────────────────────────────────
46    var   filename                  文件名
+     var   extra_field               扩展字段
+     var   comment                   文件注释
────────────────────────────────────────────────────────────
总计: 46 + filename_length + extra_length + comment_length 字节
```

### C++ 结构体

```cpp
#pragma pack(push, 1)
struct CentralDirectoryHeader {
    uint32_t signature;           // 0x02014B50
    uint16_t versionMadeBy;       // 创建版本
    uint16_t versionNeeded;       // 解压所需版本
    uint16_t flags;               // 通用标志
    uint16_t compression;         // 压缩方法
    uint16_t modTime;             // 修改时间
    uint16_t modDate;             // 修改日期
    uint32_t crc32;               // CRC校验
    uint32_t compressedSize;      // 压缩后大小
    uint32_t uncompressedSize;    // 原始大小
    uint16_t filenameLength;      // 文件名长度
    uint16_t extraLength;         // 扩展字段长度
    uint16_t commentLength;       // 注释长度
    uint16_t diskStart;           // 起始磁盘号
    uint16_t internalAttr;        // 内部属性
    uint32_t externalAttr;        // 外部属性
    uint32_t localHeaderOffset;   // Local Header偏移
    // filename[filenameLength]
    // extra[extraLength]
    // comment[commentLength]
};
#pragma pack(pop)

static_assert(sizeof(CentralDirectoryHeader) == 46, "CDH size must be 46 bytes");
```

---

## 3. Local File Header

**签名**: `50 4B 03 04` (0x04034B50)

**位置**: 由Central Directory的`local_header_offset`指定

```
偏移  大小  字段名                    说明
────────────────────────────────────────────────────────────
0     4     signature                 0x04034B50 (50 4B 03 04)
4     2     version_needed            解压所需版本
6     2     flags                     通用标志位
8     2     compression               压缩方法
10    2     mod_time                  修改时间
12    2     mod_date                  修改日期
14    4     crc32                     CRC-32（可能为0，见Data Descriptor）
18    4     compressed_size           压缩后大小（可能为0）
22    4     uncompressed_size         原始大小（可能为0）
26    2     filename_length           文件名长度
28    2     extra_length              扩展字段长度
────────────────────────────────────────────────────────────
30    var   filename                  文件名
+     var   extra_field               扩展字段
+     var   file_data                 压缩后的文件数据
+     (opt) data_descriptor           数据描述符（如果flags bit 3设置）
────────────────────────────────────────────────────────────
```

### C++ 结构体

```cpp
#pragma pack(push, 1)
struct LocalFileHeader {
    uint32_t signature;           // 0x04034B50
    uint16_t versionNeeded;       // 解压所需版本
    uint16_t flags;               // 通用标志
    uint16_t compression;         // 压缩方法
    uint16_t modTime;             // 修改时间
    uint16_t modDate;             // 修改日期
    uint32_t crc32;               // CRC校验
    uint32_t compressedSize;      // 压缩后大小
    uint32_t uncompressedSize;    // 原始大小
    uint16_t filenameLength;      // 文件名长度
    uint16_t extraLength;         // 扩展字段长度
    // filename[filenameLength]
    // extra[extraLength]
    // file_data[compressedSize]
};
#pragma pack(pop)

static_assert(sizeof(LocalFileHeader) == 30, "LFH size must be 30 bytes");
```

---

## 4. ZIP64 扩展（大文件支持）

当文件大于4GB或条目超过65535时使用。

### ZIP64 End of Central Directory Record

**签名**: `50 4B 06 06` (0x06064B50)

```
偏移  大小  字段名                    说明
────────────────────────────────────────────────────────────
0     4     signature                 0x06064B50
4     8     record_size               本记录剩余大小
12    2     version_made_by           创建版本
14    2     version_needed            所需版本
16    4     disk_number               当前磁盘号
20    4     disk_cd_start             CD开始磁盘号
24    8     cd_entries_this_disk      本磁盘CD条目数（64位）
32    8     cd_entries_total          CD总条目数（64位）
40    8     cd_size                   CD大小（64位）
48    8     cd_offset                 CD偏移（64位）
56    var   extensible_data           扩展数据
────────────────────────────────────────────────────────────
```

### ZIP64 End of Central Directory Locator

**签名**: `50 4B 06 07` (0x07064B50)

**位置**: 紧接在EOCD之前

```
偏移  大小  字段名                    说明
────────────────────────────────────────────────────────────
0     4     signature                 0x07064B50
4     4     disk_zip64_eocd           ZIP64 EOCD所在磁盘号
8     8     zip64_eocd_offset         ZIP64 EOCD偏移（64位）
16    4     total_disks               磁盘总数
────────────────────────────────────────────────────────────
总计: 20 字节
```

---

## 5. 恢复算法示例

```cpp
struct ZipRecoveryInfo {
    bool hasValidEOCD;
    bool hasValidCD;
    bool isZip64;

    uint64_t cdOffset;        // Central Directory位置
    uint64_t cdSize;          // Central Directory大小
    uint32_t fileCount;       // 文件数量

    struct FileEntry {
        std::string filename;
        uint64_t localHeaderOffset;
        uint64_t compressedSize;
        uint64_t uncompressedSize;
        uint32_t crc32;
        bool isValid;
    };
    std::vector<FileEntry> entries;
};

class ZipStructureParser {
public:
    // 从文件末尾搜索EOCD
    bool FindEOCD(const uint8_t* data, size_t size, EOCD& eocd, size_t& eocdOffset) {
        // 签名: 50 4B 05 06
        const uint8_t sig[] = {0x50, 0x4B, 0x05, 0x06};

        // 从末尾向前搜索（考虑comment最长65535）
        size_t maxSearch = std::min(size, (size_t)(65535 + 22));

        for (size_t i = size - 22; i >= size - maxSearch; i--) {
            if (memcmp(data + i, sig, 4) == 0) {
                memcpy(&eocd, data + i, sizeof(EOCD));
                eocdOffset = i;

                // 验证: 注释长度应该匹配剩余字节
                size_t expectedEnd = i + 22 + eocd.commentLength;
                if (expectedEnd == size) {
                    return true;
                }
            }
        }
        return false;
    }

    // 解析Central Directory
    bool ParseCentralDirectory(const uint8_t* data, size_t size,
                               const EOCD& eocd, ZipRecoveryInfo& info) {
        if (eocd.cdOffset + eocd.cdSize > size) {
            // CD超出文件范围，可能是碎片化
            info.hasValidCD = false;
            return false;
        }

        const uint8_t* cdPtr = data + eocd.cdOffset;
        size_t offset = 0;

        for (uint16_t i = 0; i < eocd.cdEntriesTotal && offset < eocd.cdSize; i++) {
            CentralDirectoryHeader cdh;
            memcpy(&cdh, cdPtr + offset, sizeof(cdh));

            if (cdh.signature != 0x02014B50) {
                break;  // 无效签名
            }

            ZipRecoveryInfo::FileEntry entry;
            entry.localHeaderOffset = cdh.localHeaderOffset;
            entry.compressedSize = cdh.compressedSize;
            entry.uncompressedSize = cdh.uncompressedSize;
            entry.crc32 = cdh.crc32;

            // 读取文件名
            entry.filename.assign(
                (char*)(cdPtr + offset + 46),
                cdh.filenameLength
            );

            // 验证Local Header是否可访问
            entry.isValid = (cdh.localHeaderOffset < size);

            info.entries.push_back(entry);

            offset += 46 + cdh.filenameLength + cdh.extraLength + cdh.commentLength;
        }

        info.hasValidCD = (info.entries.size() == eocd.cdEntriesTotal);
        return info.hasValidCD;
    }

    // 计算预期文件大小
    uint64_t CalculateExpectedSize(const ZipRecoveryInfo& info, const EOCD& eocd) {
        // 预期大小 = CD偏移 + CD大小 + EOCD大小
        return eocd.cdOffset + eocd.cdSize + 22 + eocd.commentLength;
    }

    // 检测碎片/损坏
    std::vector<std::pair<uint64_t, uint64_t>> DetectGaps(
        const ZipRecoveryInfo& info, size_t actualSize) {

        std::vector<std::pair<uint64_t, uint64_t>> gaps;

        // 按偏移排序
        auto entries = info.entries;
        std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) {
                return a.localHeaderOffset < b.localHeaderOffset;
            });

        uint64_t expectedOffset = 0;
        for (const auto& entry : entries) {
            if (entry.localHeaderOffset > expectedOffset) {
                // 有间隙
                gaps.push_back({expectedOffset, entry.localHeaderOffset});
            }
            // 下一个预期位置 = 当前位置 + header + data
            expectedOffset = entry.localHeaderOffset + 30 +
                             entry.filename.size() + entry.compressedSize;
        }

        return gaps;
    }
};
```

---

## 6. 文件恢复流程图

```
                    ┌─────────────────┐
                    │  读取文件末尾   │
                    │  搜索EOCD签名   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  找到EOCD?      │
                    └────────┬────────┘
                    Yes      │      No
              ┌──────────────┴──────────────┐
              ▼                              ▼
     ┌────────────────┐            ┌────────────────┐
     │ 解析EOCD       │            │ 搜索CD签名     │
     │ 获取CD偏移/大小 │            │ (50 4B 01 02)  │
     └───────┬────────┘            └────────────────┘
             │
     ┌───────▼────────┐
     │ 验证CD偏移     │
     │ 是否在文件范围内│
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │ 解析所有CD条目  │
     │ 获取文件列表    │
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │ 验证每个文件的  │
     │ Local Header    │
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │ 检测间隙/损坏   │
     │ 报告恢复状态    │
     └────────────────┘
```

---

## 7. 关键签名速查表

| 签名 (hex) | 签名 (LE uint32) | 含义 |
|------------|------------------|------|
| 50 4B 03 04 | 0x04034B50 | Local File Header |
| 50 4B 01 02 | 0x02014B50 | Central Directory Header |
| 50 4B 05 06 | 0x06054B50 | End of Central Directory |
| 50 4B 06 06 | 0x06064B50 | ZIP64 EOCD Record |
| 50 4B 06 07 | 0x07064B50 | ZIP64 EOCD Locator |
| 50 4B 07 08 | 0x08074B50 | Data Descriptor |

---

## 8. 常见问题处理

### 8.1 EOCD被覆盖
- 从末尾找不到EOCD
- **策略**: 全盘搜索CD签名(50 4B 01 02)，重建文件列表

### 8.2 CD被覆盖但EOCD完好
- EOCD指向的CD位置数据损坏
- **策略**: 全盘搜索Local Header(50 4B 03 04)，逐个解析

### 8.3 ZIP64格式
- cdOffset/cdSize为0xFFFFFFFF
- **策略**: 先解析ZIP64 EOCD Locator，再读取ZIP64 EOCD

### 8.4 文件碎片化
- Local Header存在但数据不连续
- **策略**: 利用CRC32验证，搜索正确的数据块
