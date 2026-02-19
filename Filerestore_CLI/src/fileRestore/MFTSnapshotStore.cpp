#include "MFTSnapshotStore.h"
#include "Logger.h"
#include <iostream>

using namespace std;

// ============================================================================
// 构造和析构
// ============================================================================
MFTSnapshotStore::MFTSnapshotStore() : driveLetter(0) {}
MFTSnapshotStore::~MFTSnapshotStore() {}

string MFTSnapshotStore::GenerateStorePath(char drive) {
    char tempPath[MAX_PATH];
    GetTempPathA(MAX_PATH, tempPath);
    return string(tempPath) + "mft_snapshot_" + drive + ".dat";
}

// ============================================================================
// 快照管理
// ============================================================================
void MFTSnapshotStore::AddSnapshot(const MFTSnapshot& snapshot) {
    lock_guard<mutex> lock(mtx);
    snapshots[snapshot.recordNumber].push_back(snapshot);
}

const MFTSnapshot* MFTSnapshotStore::FindByRecord(ULONGLONG recordNum, WORD seqNum) const {
    lock_guard<mutex> lock(mtx);

    auto it = snapshots.find(recordNum);
    if (it == snapshots.end()) return nullptr;

    for (const auto& snap : it->second) {
        if (snap.sequenceNumber == seqNum) {
            return &snap;
        }
    }
    return nullptr;
}

const MFTSnapshot* MFTSnapshotStore::FindLatestByRecord(ULONGLONG recordNum) const {
    lock_guard<mutex> lock(mtx);

    auto it = snapshots.find(recordNum);
    if (it == snapshots.end() || it->second.empty()) return nullptr;

    // 返回最后一个（最新添加的）
    return &it->second.back();
}

vector<const MFTSnapshot*> MFTSnapshotStore::SearchByName(const wstring& pattern) const {
    lock_guard<mutex> lock(mtx);
    vector<const MFTSnapshot*> results;

    wstring lowerPattern = pattern;
    transform(lowerPattern.begin(), lowerPattern.end(), lowerPattern.begin(), ::towlower);

    for (const auto& [recordNum, snapList] : snapshots) {
        for (const auto& snap : snapList) {
            wstring lowerName = snap.fileName;
            transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::towlower);
            if (lowerName.find(lowerPattern) != wstring::npos) {
                results.push_back(&snap);
            }
        }
    }
    return results;
}

// ============================================================================
// 持久化
// ============================================================================
void MFTSnapshotStore::SerializeSnapshot(ofstream& out, const MFTSnapshot& snap) {
    out.write((char*)&snap.recordNumber, sizeof(ULONGLONG));
    out.write((char*)&snap.sequenceNumber, sizeof(WORD));
    out.write((char*)&snap.fileSize, sizeof(ULONGLONG));
    out.write((char*)&snap.captureTime, sizeof(FILETIME));
    out.write((char*)&snap.deleteTime, sizeof(FILETIME));
    out.write((char*)&snap.parentRecord, sizeof(ULONGLONG));

    BYTE flags = snap.isResident ? 0x01 : 0x00;
    out.write((char*)&flags, sizeof(BYTE));

    // 文件名
    WORD nameLen = (WORD)snap.fileName.length();
    out.write((char*)&nameLen, sizeof(WORD));
    if (nameLen > 0) {
        out.write((char*)snap.fileName.data(), nameLen * sizeof(WCHAR));
    }

    // Data Runs
    DWORD runCount = (DWORD)snap.dataRuns.size();
    out.write((char*)&runCount, sizeof(DWORD));
    for (const auto& [lcn, count] : snap.dataRuns) {
        out.write((char*)&lcn, sizeof(ULONGLONG));
        out.write((char*)&count, sizeof(ULONGLONG));
    }

    // 常驻数据
    DWORD resDataLen = (DWORD)snap.residentData.size();
    out.write((char*)&resDataLen, sizeof(DWORD));
    if (resDataLen > 0) {
        out.write((char*)snap.residentData.data(), resDataLen);
    }
}

bool MFTSnapshotStore::DeserializeSnapshot(ifstream& in, MFTSnapshot& snap) {
    in.read((char*)&snap.recordNumber, sizeof(ULONGLONG));
    in.read((char*)&snap.sequenceNumber, sizeof(WORD));
    in.read((char*)&snap.fileSize, sizeof(ULONGLONG));
    in.read((char*)&snap.captureTime, sizeof(FILETIME));
    in.read((char*)&snap.deleteTime, sizeof(FILETIME));
    in.read((char*)&snap.parentRecord, sizeof(ULONGLONG));

    BYTE flags;
    in.read((char*)&flags, sizeof(BYTE));
    snap.isResident = (flags & 0x01) != 0;

    // 文件名
    WORD nameLen;
    in.read((char*)&nameLen, sizeof(WORD));
    if (nameLen > 0 && nameLen < 512) {
        snap.fileName.resize(nameLen);
        in.read((char*)snap.fileName.data(), nameLen * sizeof(WCHAR));
    }

    // Data Runs
    DWORD runCount;
    in.read((char*)&runCount, sizeof(DWORD));
    if (runCount > 0 && runCount < 10000) {
        snap.dataRuns.resize(runCount);
        for (DWORD j = 0; j < runCount; j++) {
            in.read((char*)&snap.dataRuns[j].first, sizeof(ULONGLONG));
            in.read((char*)&snap.dataRuns[j].second, sizeof(ULONGLONG));
        }
    }

    // 常驻数据
    DWORD resDataLen;
    in.read((char*)&resDataLen, sizeof(DWORD));
    if (resDataLen > 0 && resDataLen < 4096) {  // MFT 常驻数据最大 ~3.5KB
        snap.residentData.resize(resDataLen);
        in.read((char*)snap.residentData.data(), resDataLen);
    }

    return in.good();
}

bool MFTSnapshotStore::SaveToFile(const string& path) {
    lock_guard<mutex> lock(mtx);

    ofstream out(path, ios::binary);
    if (!out) {
        LOG_ERROR("无法创建快照存储文件");
        return false;
    }

    // 计算总快照数
    size_t totalCount = 0;
    for (const auto& [recordNum, snapList] : snapshots) {
        totalCount += snapList.size();
    }

    // 写入头
    MFTSnapshotHeader header;
    header.magic = MFT_SNAPSHOT_MAGIC;
    header.version = MFT_SNAPSHOT_VERSION;
    header.snapshotCount = totalCount;
    header.driveLetter = driveLetter;
    memset(header.padding, 0, sizeof(header.padding));
    GetSystemTimeAsFileTime(&header.lastUpdateTime);
    header.createTime = header.lastUpdateTime;

    out.write((char*)&header, sizeof(header));

    // 写入快照
    for (const auto& [recordNum, snapList] : snapshots) {
        for (const auto& snap : snapList) {
            SerializeSnapshot(out, snap);
        }
    }

    out.close();
    LOG_INFO_FMT("快照存储已保存: %zu 个快照到 %s", totalCount, path.c_str());
    return true;
}

bool MFTSnapshotStore::LoadFromFile(const string& path) {
    lock_guard<mutex> lock(mtx);
    snapshots.clear();

    ifstream in(path, ios::binary);
    if (!in) {
        return false;
    }

    // 读取头
    MFTSnapshotHeader header;
    in.read((char*)&header, sizeof(header));

    if (header.magic != MFT_SNAPSHOT_MAGIC || header.version != MFT_SNAPSHOT_VERSION) {
        LOG_ERROR("快照存储文件格式无效或版本不匹配");
        return false;
    }

    driveLetter = header.driveLetter;

    // 读取快照
    for (ULONGLONG i = 0; i < header.snapshotCount; i++) {
        MFTSnapshot snap;
        if (!DeserializeSnapshot(in, snap)) {
            LOG_ERROR("快照存储文件损坏");
            snapshots.clear();
            return false;
        }
        snapshots[snap.recordNumber].push_back(snap);
    }

    in.close();
    LOG_INFO_FMT("快照存储已加载: %llu 个快照从 %s", header.snapshotCount, path.c_str());
    return true;
}

// ============================================================================
// 清理和统计
// ============================================================================
size_t MFTSnapshotStore::PurgeOlderThan(int days) {
    lock_guard<mutex> lock(mtx);

    FILETIME now;
    GetSystemTimeAsFileTime(&now);

    ULARGE_INTEGER nowLI;
    nowLI.LowPart = now.dwLowDateTime;
    nowLI.HighPart = now.dwHighDateTime;

    // 100 纳秒 * 10^7 = 1 秒, * 86400 = 1 天
    ULONGLONG cutoff = nowLI.QuadPart - (ULONGLONG)days * 864000000000ULL;

    size_t purgedCount = 0;

    for (auto it = snapshots.begin(); it != snapshots.end(); ) {
        auto& snapList = it->second;

        auto removeIt = remove_if(snapList.begin(), snapList.end(),
            [cutoff, &purgedCount](const MFTSnapshot& snap) {
                ULARGE_INTEGER snapTime;
                snapTime.LowPart = snap.captureTime.dwLowDateTime;
                snapTime.HighPart = snap.captureTime.dwHighDateTime;
                if (snapTime.QuadPart < cutoff) {
                    purgedCount++;
                    return true;
                }
                return false;
            });
        snapList.erase(removeIt, snapList.end());

        if (snapList.empty()) {
            it = snapshots.erase(it);
        } else {
            ++it;
        }
    }

    return purgedCount;
}

size_t MFTSnapshotStore::GetCount() const {
    lock_guard<mutex> lock(mtx);
    size_t count = 0;
    for (const auto& [recordNum, snapList] : snapshots) {
        count += snapList.size();
    }
    return count;
}

void MFTSnapshotStore::Clear() {
    lock_guard<mutex> lock(mtx);
    snapshots.clear();
}
