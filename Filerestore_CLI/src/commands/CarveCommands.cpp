// CarveCommands.cpp - 文件雕刻(File Carving)相关命令实现
// 包含: CarveTypesCommand, CarveCommand, CarveRecoverCommand, CarveCommandThreadPool,
//       CarveTimestampCommand, CarveListCommand, CarveValidateCommand, CarveIntegrityCommand

#include "cmd.h"
#include "CommandUtils.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <Windows.h>
#include <algorithm>
#include <cmath>
#include "MFTReader.h"
#include "FileCarver.h"
#include "FileIntegrityValidator.h"
#include "CarvedResultsCache.h"
#include "MemoryMappedResults.h"

using namespace std;

// 静态变量存储扫描结果（内存模式）
static vector<CarvedFileInfo> lastCarveResults;
static char lastCarveDrive = 0;

// 缓存和内存映射管理（大数据集模式）
static CarvedResultsCache* resultsCache = nullptr;
static MemoryMappedResults* mappedResults = nullptr;
static bool useMemoryMapping = false;
static size_t currentPageIndex = 0;
static size_t pageSize = 50;  // 每页显示50条记录

// 辅助函数：尝试从磁盘缓存初始化（程序重启后恢复）
static bool TryInitFromDiskCache() {
    if (!resultsCache) {
        resultsCache = new CarvedResultsCache();
    }

    if (resultsCache->InitFromAnyExistingCache()) {
        // 获取驱动器字母
        ULONGLONG count;
        char drive;
        if (resultsCache->GetCacheInfo(count, drive)) {
            lastCarveDrive = drive;
        }
        return true;
    }

    return false;
}

// 辅助函数：获取总记录数
static size_t GetTotalResultCount() {
    if (useMemoryMapping && mappedResults && mappedResults->IsValid()) {
        return (size_t)mappedResults->GetTotalRecords();
    } else if (resultsCache && resultsCache->IsValid()) {
        ULONGLONG count;
        char drive;
        if (resultsCache->GetCacheInfo(count, drive)) {
            return (size_t)count;
        }
    }

    // 如果内存中没有结果，尝试从磁盘缓存加载
    if (lastCarveResults.empty()) {
        if (TryInitFromDiskCache()) {
            ULONGLONG count;
            char drive;
            if (resultsCache->GetCacheInfo(count, drive)) {
                return (size_t)count;
            }
        }
    }

    return lastCarveResults.size();
}

// 辅助函数：加载指定页的结果
static bool LoadResultPage(size_t pageIndex, vector<CarvedFileInfo>& outResults) {
    size_t startIndex = pageIndex * pageSize;

    if (useMemoryMapping && mappedResults && mappedResults->IsValid()) {
        return mappedResults->GetRecordBatch(startIndex, pageSize, outResults);
    } else if (resultsCache && resultsCache->IsValid()) {
        char drive;
        return resultsCache->LoadResultRange(outResults, startIndex, pageSize, drive);
    } else {
        // 内存模式
        outResults.clear();
        size_t endIndex = min(startIndex + pageSize, lastCarveResults.size());
        for (size_t i = startIndex; i < endIndex; i++) {
            outResults.push_back(lastCarveResults[i]);
        }
        return !outResults.empty();
    }
}

// 辅助函数：获取单个结果（支持缓存/内存映射/内存三种模式）
static bool GetSingleResult(size_t index, CarvedFileInfo& outResult, char& outDrive) {
    size_t totalCount = GetTotalResultCount();
    if (index >= totalCount) {
        return false;
    }

    if (useMemoryMapping && mappedResults && mappedResults->IsValid()) {
        // 内存映射模式
        vector<CarvedFileInfo> batch;
        if (mappedResults->GetRecordBatch(index, 1, batch) && !batch.empty()) {
            outResult = batch[0];
            outDrive = lastCarveDrive;
            return true;
        }
    } else if (resultsCache && resultsCache->IsValid()) {
        // 缓存模式
        vector<CarvedFileInfo> batch;
        if (resultsCache->LoadResultRange(batch, index, 1, outDrive) && !batch.empty()) {
            outResult = batch[0];
            return true;
        }
    } else if (index < lastCarveResults.size()) {
        // 内存模式
        outResult = lastCarveResults[index];
        outDrive = lastCarveDrive;
        return true;
    }

    return false;
}

// 辅助函数：将 FILETIME 转换为字符串
static string FileTimeToString(const FILETIME& ft) {
	if (ft.dwHighDateTime == 0 && ft.dwLowDateTime == 0) {
		return "N/A";
	}

	SYSTEMTIME st;
	FileTimeToSystemTime(&ft, &st);

	char buffer[32];
	sprintf_s(buffer, "%04d-%02d-%02d %02d:%02d:%02d",
		st.wYear, st.wMonth, st.wDay,
		st.wHour, st.wMinute, st.wSecond);
	return string(buffer);
}

// 辅助函数：获取时间戳来源描述
static string GetTimestampSourceStr(TimestampSource source) {
	switch (source) {
	case TS_EMBEDDED: return "Embedded";
	case TS_MFT_MATCH: return "MFT";
	case TS_BOTH: return "Both";
	default: return "None";
	}
}

// 比较函数：优先置信度高，其次日期早
static bool CompareCarvedFiles(const CarvedFileInfo& a, const CarvedFileInfo& b) {
	// 首先按置信度降序
	if (abs(a.confidence - b.confidence) > 0.05) {
		return a.confidence > b.confidence;
	}

	// 置信度相近时，按修改时间升序（日期早的优先）
	ULARGE_INTEGER timeA, timeB;
	timeA.LowPart = a.modificationTime.dwLowDateTime;
	timeA.HighPart = a.modificationTime.dwHighDateTime;
	timeB.LowPart = b.modificationTime.dwLowDateTime;
	timeB.HighPart = b.modificationTime.dwHighDateTime;

	// 如果都没有时间戳，保持原顺序
	if (timeA.QuadPart == 0 && timeB.QuadPart == 0) {
		return a.confidence > b.confidence;
	}

	// 有时间戳的优先于没有的
	if (timeA.QuadPart == 0) return false;
	if (timeB.QuadPart == 0) return true;

	// 日期早的优先
	return timeA.QuadPart < timeB.QuadPart;
}

// ============================================================================
// CarveTypesCommand - 列出支持的文件类型
// ============================================================================
DEFINE_COMMAND_BASE_NOARGS(CarveTypesCommand, "carvetypes")
REGISTER_COMMAND(CarveTypesCommand);

void CarveTypesCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	cout << "\n=== Supported File Types for Carving ===" << endl;
	cout << "\nThe following file types can be recovered using signature search:\n" << endl;

	cout << "  zip     - ZIP Archive (including DOCX, XLSX, PPTX, JAR, APK)" << endl;
	cout << "  pdf     - PDF Document" << endl;
	cout << "  jpg     - JPEG Image" << endl;
	cout << "  png     - PNG Image" << endl;
	cout << "  gif     - GIF Image" << endl;
	cout << "  bmp     - Bitmap Image" << endl;
	cout << "  7z      - 7-Zip Archive" << endl;
	cout << "  rar     - RAR Archive" << endl;
	cout << "  mp3     - MP3 Audio (with ID3 tag)" << endl;
	cout << "  mp4     - MP4/MOV Video" << endl;
	cout << "  avi     - AVI Video" << endl;
	cout << "  exe     - Windows Executable" << endl;
	cout << "  sqlite  - SQLite Database" << endl;
	cout << "  wav     - WAV Audio" << endl;

	cout << "\nUsage:" << endl;
	cout << "  carve <drive> <file_type> <output_directory>" << endl;
	cout << "  Example: carve C zip D:\\recovered\\" << endl;
	cout << "\n  Use 'carve <drive> all <output_directory>' to scan all types" << endl;
}

// ============================================================================
// CarveCommand - 签名搜索扫描
// ============================================================================
DEFINE_COMMAND_BASE(CarveCommand, "carve |name |name |file |name", TRUE)
REGISTER_COMMAND(CarveCommand);

void CarveCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 3) {
		cout << "Usage: carve <drive> <type|types|all> <output_dir> [async|sync]" << endl;
		cout << "Examples:" << endl;
		cout << "  carve C zip D:\\recovered\\" << endl;
		cout << "  carve C jpg,png,gif D:\\recovered\\" << endl;
		cout << "  carve D all D:\\recovered\\ async" << endl;
		cout << "\nModes:" << endl;
		cout << "  async - Use dual-buffer async I/O (default, faster)" << endl;
		cout << "  sync  - Use synchronous I/O (simpler)" << endl;
		cout << "\nUse 'carvetypes' to see supported file types." << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& fileTypeArg = GET_ARG_STRING(1);
		string& outputDir = GET_ARG_STRING(2);

		// 检查是否指定了同步/异步模式
		bool useAsync = true;  // 默认使用异步
		if (HAS_ARG(3)) {
			string& modeStr = GET_ARG_STRING(3);
			if (modeStr == "sync" || modeStr == "s") {
				useAsync = false;
			}
		}

		char driveLetter;
		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		// 创建输出目录
		if (!CommandUtils::CreateOutputDirectory(outputDir)) {
			cout << "Warning: Could not create output directory" << endl;
		}

		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "Failed to open volume " << driveLetter << ":" << endl;
			return;
		}

		FileCarver carver(&reader);
		carver.SetAsyncMode(useAsync);

		vector<CarvedFileInfo> results;

		// 解析文件类型 - 使用 CommandUtils
		vector<string> types = CommandUtils::ParseFileTypes(fileTypeArg, carver.GetSupportedTypes());

		// 执行扫描
		if (useAsync) {
			results = carver.ScanForFileTypesAsync(types, CARVE_SMART, 500);
		} else {
			if (types.size() == 1) {
				results = carver.ScanForFileType(types[0], CARVE_SMART, 500);
			} else {
				results = carver.ScanForFileTypes(types, CARVE_SMART, 500);
			}
		}

		// 保存结果供后续恢复使用
		lastCarveDrive = driveLetter;

		if (results.empty()) {
			cout << "\nNo files found." << endl;
			return;
		}

		// 检查是否应该使用缓存/内存映射
		if (results.size() >= 1000) {  // 超过1000条记录使用持久化
			cout << "\n=== Saving " << results.size() << " results to cache ===" << endl;

			if (!resultsCache) {
				resultsCache = new CarvedResultsCache();
			}

			if (resultsCache->SaveResults(results, driveLetter)) {
				cout << "Results saved to cache file: " << resultsCache->GetCachePath() << endl;
				cout << "Cache size: " << (resultsCache->GetCacheSize(driveLetter) / (1024 * 1024)) << " MB" << endl;

				// 清空内存中的结果
				lastCarveResults.clear();

				// 检查是否应该使用内存映射
				if (MemoryMappedResults::ShouldUseMemoryMapping(results.size())) {
					cout << "\nUsing memory-mapped files for large result set..." << endl;

					if (!mappedResults) {
						mappedResults = new MemoryMappedResults();
					}

					if (mappedResults->OpenFromCache(resultsCache->GetCachePath())) {
						useMemoryMapping = true;
						cout << "Memory mapping enabled (" << results.size() << " records)" << endl;
					}
				}

				currentPageIndex = 0;  // 重置分页
			} else {
				cout << "Failed to save cache, keeping results in memory" << endl;
				lastCarveResults = results;
			}
		} else {
			// 小数据集，直接保存到内存
			lastCarveResults = results;
			useMemoryMapping = false;
		}

		cout << "\n=== Found " << results.size() << " file(s) ===" << endl;
		cout << "Use 'carverecover <index> <output_path>' to recover a specific file." << endl;
		cout << "Use 'carvelist' to browse results with pagination." << endl;
		cout << "\nFile List (first 50 records):" << endl;

		// 显示前50条记录
		size_t displayCount = min(results.size(), (size_t)50);
		for (size_t i = 0; i < displayCount; i++) {
			const auto& info = results[i];
			cout << "[" << i << "] " << info.extension << " | ";
			cout << "LCN: " << info.startLCN << " | ";
			cout << "Size: " << (info.fileSize / 1024) << " KB | ";
			cout << "Confidence: " << (int)(info.confidence * 100) << "%" << endl;
		}

		if (results.size() > 50) {
			cout << "\n... and " << (results.size() - 50) << " more files" << endl;
			cout << "Use 'carvelist <page>' to view more results" << endl;
		}

		// 检查删除状态并过滤
		cout << "\n=== Checking Deletion Status ===" << endl;
		carver.CheckDeletionStatusForResults(results, true);

		size_t deletedCount = carver.CountDeletedFiles(results);
		size_t activeCount = carver.CountActiveFiles(results);

		cout << "\nDeleted files (recoverable): " << deletedCount << endl;
		cout << "Active files (already exist): " << activeCount << " (will be skipped)" << endl;

		// 自动恢复高置信度的已删除文件
		cout << "\n=== Auto-Recovering High Confidence Deleted Files ===" << endl;

		size_t recoveredCount = 0;
		size_t skippedActiveCount = 0;
		for (size_t i = 0; i < results.size(); i++) {
			const auto& info = results[i];

			// 跳过活动文件（未删除）
			if (info.deletionChecked && info.isActiveFile) {
				skippedActiveCount++;
				continue;
			}

			if (info.confidence >= 0.8) {  // 只自动恢复高置信度文件
				string outputPath = outputDir;
				if (outputPath.back() != '\\' && outputPath.back() != '/') {
					outputPath += '\\';
				}
				outputPath += "carved_" + to_string(i) + "." + info.extension;

				if (carver.RecoverCarvedFile(info, outputPath)) {
					recoveredCount++;
				}

				if (recoveredCount >= 20) {  // 最多自动恢复20个
					cout << "\nReached auto-recovery limit (20 files)." << endl;
					break;
				}
			}
		}

		// 显示扫描统计
		const CarvingStats& stats = carver.GetStats();
		cout << "\n=== Recovery Summary ===" << endl;
		cout << "Total files found: " << results.size() << endl;
		cout << "Deleted files: " << deletedCount << endl;
		cout << "Active files skipped: " << skippedActiveCount << endl;
		cout << "Auto-recovered: " << recoveredCount << " files" << endl;
		cout << "Output directory: " << outputDir << endl;
		cout << "\nUse 'carverecover <index> <output_path>' to recover additional files." << endl;
		cout << "Use 'carvelist deleted' to show only deleted files." << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// CarveRecoverCommand - 恢复指定的 carved 文件
// ============================================================================
DEFINE_COMMAND_BASE(CarveRecoverCommand, "carverecover |name |file", TRUE)
REGISTER_COMMAND(CarveRecoverCommand);

void CarveRecoverCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 2) {
		cout << "Usage: carverecover <index> <output_path>" << endl;
		cout << "Example: carverecover 0 D:\\recovered\\file.zip" << endl;
		cout << "         carverecover 5 C:\\Users\\Me\\Desktop\\recovered.pdf" << endl;
		cout << "\nNOTE: Use full absolute path (e.g., D:\\folder\\file.ext)." << endl;
		cout << "      If ACCESS_DENIED error occurs, run as Administrator." << endl;
		cout << "\nRun 'carve' or 'carvepool' first to scan for files." << endl;
		return;
	}

	try {
		string& indexStr = GET_ARG_STRING(0);
		string outputPath = GET_ARG_STRING(1);

		// 检查是否为绝对路径
		if (outputPath.length() < 3 || outputPath[1] != ':') {
			cout << "[WARNING] Output path appears to be relative: " << outputPath << endl;
			// 尝试将相对路径转为绝对路径
			char currentDir[MAX_PATH];
			if (GetCurrentDirectoryA(MAX_PATH, currentDir)) {
				outputPath = string(currentDir) + "\\" + outputPath;
				cout << "          Resolved to: " << outputPath << endl;
			}
		}

		size_t index = stoull(indexStr);

		// 使用统一的结果计数（支持缓存/内存映射/内存三种模式）
		size_t totalCount = GetTotalResultCount();
		if (totalCount == 0) {
			cout << "No carving results available. Run 'carve' or 'carvepool' first." << endl;
			return;
		}

		if (index >= totalCount) {
			cout << "Invalid index. Valid range: 0-" << (totalCount - 1) << endl;
			return;
		}

		// 获取单个结果（支持缓存/内存映射/内存三种模式）
		CarvedFileInfo info;
		char driveLetter;
		if (!GetSingleResult(index, info, driveLetter)) {
			cout << "Failed to retrieve file info at index " << index << endl;
			return;
		}

		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "Failed to open volume " << driveLetter << ":" << endl;
			return;
		}

		FileCarver carver(&reader);

		cout << "Recovering file #" << index << "..." << endl;
		cout << "  Type: " << info.description << endl;
		cout << "  Size: " << info.fileSize << " bytes" << endl;

		if (carver.RecoverCarvedFile(info, outputPath)) {
			cout << "\n=== Recovery Successful ===" << endl;
			cout << "File saved to: " << outputPath << endl;
		} else {
			cout << "\n=== Recovery Failed ===" << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// CarvePoolCommand - 线程池并行签名搜索扫描（带时间戳排序）
// ============================================================================
DEFINE_COMMAND_BASE(CarveCommandThreadPool, "carvepool |name |name |file |name |name", TRUE)
REGISTER_COMMAND(CarveCommandThreadPool);

void CarveCommandThreadPool::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 3) {
		cout << "Usage: carvepool <drive> <type|types|all> <output_dir> [threads] [options...]" << endl;
		cout << "\nThread Pool Parallel Signature Scanner with Timestamp Sorting" << endl;
		cout << "Files are sorted by: 1) Confidence (high first) 2) Date (earlier first)\n" << endl;
		cout << "Examples:" << endl;
		cout << "  carvepool C zip D:\\recovered\\" << endl;
		cout << "  carvepool C jpg,png,txt D:\\recovered\\ 8" << endl;
		cout << "  carvepool D all D:\\recovered\\ 0 hybrid" << endl;
		cout << "  carvepool D txt,html,xml D:\\recovered\\ 0 ml" << endl;
		cout << "\nOptions:" << endl;
		cout << "  [threads]     - Worker threads (0 = auto-detect)" << endl;
		cout << "  notimestamp   - Skip timestamp extraction (faster)" << endl;
		cout << "  hybrid        - Hybrid mode: signature + ML scan (default)" << endl;
		cout << "  sig           - Signature-only scan (fastest)" << endl;
		cout << "  ml            - ML-only scan (for txt/html/xml)" << endl;
		cout << "\nML-only types (no signature): txt, html, xml" << endl;
		cout << "Use 'carvetypes' to see supported signature types." << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& fileTypeArg = GET_ARG_STRING(1);
		string& outputDir = GET_ARG_STRING(2);

		// 解析选项
		int threadCount = 0;
		bool extractTimestamps = true;
		string scanMode = "hybrid";  // 默认混合模式

		// 遍历所有可选参数
		for (size_t i = 3; i < GET_ARG_COUNT(); i++) {
			string arg = GET_ARG_STRING(i);

			if (arg == "notimestamp" || arg == "nots") {
				extractTimestamps = false;
			} else if (arg == "hybrid" || arg == "sig" || arg == "signature" || arg == "ml") {
				scanMode = arg;
				if (arg == "signature") scanMode = "sig";
			} else {
				try {
					int tc = stoi(arg);
					if (tc >= 0 && tc <= 64) {
						threadCount = tc;
					}
				} catch (...) {}
			}
		}

		char driveLetter;
		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		// 创建输出目录
		if (!CommandUtils::CreateOutputDirectory(outputDir)) {
			cout << "Warning: Could not create output directory" << endl;
		}

		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "Failed to open volume " << driveLetter << ":" << endl;
			return;
		}

		FileCarver carver(&reader);

		// 设置线程数
		if (threadCount > 0) {
			ScanThreadPoolConfig config = carver.GetThreadPoolConfig();
			config.workerCount = threadCount;
			config.autoDetectThreads = false;
			carver.SetThreadPoolConfig(config);
		}

		vector<CarvedFileInfo> results;

		// 解析文件类型 - 使用 CommandUtils
		vector<string> types = CommandUtils::ParseFileTypes(fileTypeArg, carver.GetSupportedTypes());

		// 添加 ML 支持的类型（txt, html, xml）如果是混合或 ML 模式
		if (scanMode == "hybrid" || scanMode == "ml") {
			vector<string> mlTypes = {"txt", "html", "xml"};
			for (const auto& mt : mlTypes) {
				// 检查是否在 fileTypeArg 中或 all
				if (fileTypeArg == "all" ||
					fileTypeArg.find(mt) != string::npos) {
					if (find(types.begin(), types.end(), mt) == types.end()) {
						types.push_back(mt);
					}
				}
			}
		}

		cout << "\n=== Scan Mode: " << scanMode << " ===" << endl;
		cout << "Target types: ";
		for (const auto& t : types) cout << t << " ";
		cout << endl;

		// 根据模式选择扫描方法
		if (scanMode == "hybrid") {
			// 混合模式：签名 + ML
			HybridScanConfig hybridConfig;
			results = carver.ScanHybridMode(types, hybridConfig, CARVE_SMART, 1000);
		} else if (scanMode == "ml") {
			// 纯 ML 模式
			results = carver.ScanWithMLOnly(CARVE_SMART, 1000, 0.7f);
		} else {
			// 纯签名模式 (sig)
			results = carver.ScanForFileTypesThreadPool(types, CARVE_SMART, 1000);
		}

		if (results.empty()) {
			cout << "\nNo files found." << endl;
			return;
		}

		// ========== 时间戳提取与排序 ==========
		if (extractTimestamps) {
			cout << "\n=== Extracting Timestamps ===" << endl;

			// 构建 MFT 索引
			cout << "Building MFT LCN index..." << endl;
			bool mftIndexOk = carver.BuildMFTIndex(false);
			if (!mftIndexOk) {
				cout << "Warning: MFT index failed, using embedded metadata only." << endl;
			}

			// 提取时间戳
			carver.ExtractTimestampsForResults(results, true);

			// 按置信度和日期排序
			cout << "\nSorting by confidence and date..." << endl;
			sort(results.begin(), results.end(), CompareCarvedFiles);

			cout << "Sorting complete." << endl;
		}

		// 保存结果
		lastCarveDrive = driveLetter;

		// 检查是否应该使用缓存/内存映射（与CarveCommand相同逻辑）
		if (results.size() >= 1000) {
			cout << "\n=== Saving " << results.size() << " results to cache ===" << endl;

			if (!resultsCache) {
				resultsCache = new CarvedResultsCache();
			}

			if (resultsCache->SaveResults(results, driveLetter)) {
				cout << "Results saved to cache file: " << resultsCache->GetCachePath() << endl;
				cout << "Cache size: " << (resultsCache->GetCacheSize(driveLetter) / (1024 * 1024)) << " MB" << endl;

				lastCarveResults.clear();

				if (MemoryMappedResults::ShouldUseMemoryMapping(results.size())) {
					cout << "\nUsing memory-mapped files for large result set..." << endl;

					if (!mappedResults) {
						mappedResults = new MemoryMappedResults();
					}

					if (mappedResults->OpenFromCache(resultsCache->GetCachePath())) {
						useMemoryMapping = true;
						cout << "Memory mapping enabled (" << results.size() << " records)" << endl;
					}
				}

				currentPageIndex = 0;
			} else {
				cout << "Failed to save cache, keeping results in memory" << endl;
				lastCarveResults = results;
			}
		} else {
			lastCarveResults = results;
			useMemoryMapping = false;
		}

		// ========== 显示结果 ==========
		cout << "\n=== Found " << results.size() << " file(s) ===" << endl;
		if (extractTimestamps) {
			cout << "Sorted by: Confidence (high) -> Date (earlier first)" << endl;
		}
		cout << "\nTop Files:" << endl;
		cout << string(85, '-') << endl;

		size_t withTimestamp = 0;
		for (size_t i = 0; i < results.size() && i < 30; i++) {
			const auto& info = results[i];

			cout << "[" << setw(3) << i << "] ";
			cout << setw(6) << info.extension << " | ";
			cout << setw(5) << (info.fileSize / 1024) << " KB | ";
			cout << setw(3) << (int)(info.confidence * 100) << "% | ";

			// 显示时间戳
			if (info.tsSource != TS_NONE_1 &&
				(info.modificationTime.dwHighDateTime != 0 || info.modificationTime.dwLowDateTime != 0)) {
				withTimestamp++;
				SYSTEMTIME st;
				FileTimeToSystemTime(&info.modificationTime, &st);
				printf("%04d-%02d-%02d %02d:%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute);

				// 来源标记
				if (info.tsSource == TS_EMBEDDED) cout << " [E]";
				else if (info.tsSource == TS_MFT_MATCH) cout << " [M]";
				else if (info.tsSource == TS_BOTH) cout << " [B]";
			} else {
				cout << "       N/A        ";
			}

			cout << endl;
		}

		cout << string(85, '-') << endl;
		if (results.size() > 30) {
			cout << "... and " << (results.size() - 30) << " more files" << endl;
		}
		cout << "Legend: [E]=Embedded, [M]=MFT, [B]=Both" << endl;
		cout << "Files with timestamp: " << withTimestamp << "/" << min(results.size(), (size_t)30) << endl;

		// ========== 检查删除状态 ==========
		cout << "\n=== Checking Deletion Status ===" << endl;
		carver.CheckDeletionStatusForResults(results, true);

		size_t deletedFileCount = carver.CountDeletedFiles(results);
		size_t activeFileCount = carver.CountActiveFiles(results);

		cout << "\nDeleted files (recoverable): " << deletedFileCount << endl;
		cout << "Active files (already exist): " << activeFileCount << " (will be skipped)" << endl;

		// ========== 智能自动恢复（只恢复已删除的文件） ==========
		cout << "\n=== Auto-Recovering Deleted Files (Confidence >= 80%, sorted by date) ===" << endl;

		size_t recoveredCount = 0;
		size_t skippedActiveCount = 0;
		for (size_t i = 0; i < results.size(); i++) {
			const auto& info = results[i];

			// 跳过活动文件（未删除）
			if (info.deletionChecked && info.isActiveFile) {
				skippedActiveCount++;
				continue;
			}

			if (info.confidence >= 0.8) {
				string outputPath = outputDir;
				if (outputPath.back() != '\\' && outputPath.back() != '/') {
					outputPath += '\\';
				}

				// 文件名包含序号和时间戳
				string filename = "carved_" + to_string(i);
				if (info.tsSource != TS_NONE_1 &&
					(info.modificationTime.dwHighDateTime != 0 || info.modificationTime.dwLowDateTime != 0)) {
					SYSTEMTIME st;
					FileTimeToSystemTime(&info.modificationTime, &st);
					char dateStr[20];
					sprintf_s(dateStr, "_%04d%02d%02d", st.wYear, st.wMonth, st.wDay);
					filename += dateStr;
				}
				filename += "." + info.extension;
				outputPath += filename;

				if (carver.RecoverCarvedFile(info, outputPath)) {
					recoveredCount++;
				}

				if (recoveredCount >= 20) {
					cout << "\nReached auto-recovery limit (20 files)." << endl;
					break;
				}
			}
		}

		// ========== 统计摘要 ==========
		size_t embeddedCount = 0, mftCount = 0;
		for (const auto& info : results) {
			if (info.tsSource == TS_EMBEDDED || info.tsSource == TS_BOTH) embeddedCount++;
			if (info.tsSource == TS_MFT_MATCH || info.tsSource == TS_BOTH) mftCount++;
		}

		cout << "\n=== Summary ===" << endl;
		cout << "Total files found: " << results.size() << endl;
		cout << "Deleted files: " << deletedFileCount << endl;
		cout << "Active files skipped: " << skippedActiveCount << endl;
		cout << "With timestamps: " << (embeddedCount + mftCount - (embeddedCount > 0 && mftCount > 0 ? min(embeddedCount, mftCount) : 0));
		cout << " (Embedded: " << embeddedCount << ", MFT: " << mftCount << ")" << endl;
		cout << "Auto-recovered: " << recoveredCount << " files" << endl;
		cout << "Output directory: " << outputDir << endl;

		cout << "\nCommands:" << endl;
		cout << "  carvelist              - Show all files with details" << endl;
		cout << "  carvelist deleted      - Show only deleted files" << endl;
		cout << "  carverecover <i> <path> - Recover specific file" << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// CarveTimestampCommand - 为扫描结果提取时间戳
// ============================================================================
DEFINE_COMMAND_BASE(CarveTimestampCommand, "carvetimestamp |name", TRUE)
REGISTER_COMMAND(CarveTimestampCommand);

void CarveTimestampCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (lastCarveResults.empty()) {
		cout << "No carving results available. Run 'carve' or 'carvepool' first." << endl;
		return;
	}

	// 检查是否要先构建 MFT 索引
	bool buildMftIndex = true;
	if (HAS_ARG(0)) {
		string& arg = GET_ARG_STRING(0);
		if (arg == "nomft" || arg == "embedded") {
			buildMftIndex = false;
			cout << "Skipping MFT index, using embedded metadata only." << endl;
		}
	}

	try {
		MFTReader reader;
		if (!reader.OpenVolume(lastCarveDrive)) {
			cout << "Failed to open volume " << lastCarveDrive << ":" << endl;
			return;
		}

		FileCarver carver(&reader);

		// 构建 MFT LCN 索引（可选）
		if (buildMftIndex) {
			cout << "Building MFT index for timestamp matching..." << endl;
			if (!carver.BuildMFTIndex(false)) {
				cout << "Warning: Failed to build MFT index. Using embedded metadata only." << endl;
			}
		}

		// 提取时间戳
		carver.ExtractTimestampsForResults(lastCarveResults, true);

		// 显示带时间戳的结果
		cout << "\n=== Files with Timestamps ===" << endl;

		size_t withTimestamp = 0;
		for (size_t i = 0; i < lastCarveResults.size() && i < 30; i++) {
			const auto& info = lastCarveResults[i];

			if (info.tsSource != TS_NONE_1) {
				withTimestamp++;
				cout << "\n[" << i << "] " << info.extension << " (" << GetTimestampSourceStr(info.tsSource) << ")" << endl;
				cout << "    Size: " << info.fileSize << " bytes" << endl;

				if (info.creationTime.dwHighDateTime != 0 || info.creationTime.dwLowDateTime != 0) {
					cout << "    Created:  " << FileTimeToString(info.creationTime) << endl;
				}
				if (info.modificationTime.dwHighDateTime != 0 || info.modificationTime.dwLowDateTime != 0) {
					cout << "    Modified: " << FileTimeToString(info.modificationTime) << endl;
				}

				if (!info.embeddedInfo.empty()) {
					cout << "    Info: " << info.embeddedInfo << endl;
				}
				if (info.matchedMftRecord > 0) {
					cout << "    MFT Record: #" << info.matchedMftRecord << endl;
				}
			}
		}

		// 统计
		size_t embeddedCount = 0, mftCount = 0;
		for (const auto& info : lastCarveResults) {
			if (info.tsSource == TS_EMBEDDED || info.tsSource == TS_BOTH) embeddedCount++;
			if (info.tsSource == TS_MFT_MATCH || info.tsSource == TS_BOTH) mftCount++;
		}

		cout << "\n=== Summary ===" << endl;
		cout << "Total files: " << lastCarveResults.size() << endl;
		cout << "With timestamp: " << withTimestamp << endl;
		cout << "  - Embedded metadata: " << embeddedCount << endl;
		cout << "  - MFT match: " << mftCount << endl;

		cout << "\nUse 'carvelist' to see all files with timestamps." << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// CarveListCommand - 列出 carved 文件详细信息
// ============================================================================
DEFINE_COMMAND_BASE(CarveListCommand, "carvelist |name |name", TRUE)
REGISTER_COMMAND(CarveListCommand);

void CarveListCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	// 检查是否有可用结果
	size_t totalCount = GetTotalResultCount();
	if (totalCount == 0) {
		cout << "No carving results available. Run 'carve' or 'carvepool' first." << endl;
		return;
	}

	// 解析参数
	size_t pageOrIndex = currentPageIndex;
	size_t count = pageSize;
	bool isPageMode = true;
	bool filterDeleted = false;    // 只显示已删除文件
	bool filterActive = false;     // 只显示活动文件

	// 检查第一个参数是否为筛选关键字
	if (HAS_ARG(0)) {
		string& arg0 = GET_ARG_STRING(0);
		if (arg0 == "deleted" || arg0 == "del" || arg0 == "d") {
			filterDeleted = true;
			// 如果有第二个参数，作为页码
			if (HAS_ARG(1)) {
				try {
					pageOrIndex = stoull(GET_ARG_STRING(1));
				} catch (...) {}
			}
		} else if (arg0 == "active" || arg0 == "act" || arg0 == "a") {
			filterActive = true;
			// 如果有第二个参数，作为页码
			if (HAS_ARG(1)) {
				try {
					pageOrIndex = stoull(GET_ARG_STRING(1));
				} catch (...) {}
			}
		} else {
			// 不是筛选关键字，尝试解析为页码
			try {
				pageOrIndex = stoull(arg0);
			} catch (...) {}

			// 检查第二个参数
			if (HAS_ARG(1)) {
				string& arg1 = GET_ARG_STRING(1);
				try {
					count = stoull(arg1);
					isPageMode = false;  // 如果指定了count，视为索引模式
				} catch (...) {}
			}
		}
	}

	// 计算实际的起始索引
	size_t startIndex;
	if (isPageMode) {
		currentPageIndex = pageOrIndex;
		startIndex = currentPageIndex * pageSize;
		count = pageSize;
	} else {
		startIndex = pageOrIndex;
	}

	// 加载结果
	vector<CarvedFileInfo> displayResults;
	if (useMemoryMapping || resultsCache) {
		// 从缓存或内存映射加载
		if (!LoadResultPage(isPageMode ? currentPageIndex : (startIndex / pageSize), displayResults)) {
			cout << "Failed to load results from cache/memory-mapping" << endl;
			return;
		}

		// 如果是索引模式，调整显示范围
		if (!isPageMode) {
			size_t offsetInPage = startIndex % pageSize;
			if (offsetInPage > 0 && offsetInPage < displayResults.size()) {
				displayResults.erase(displayResults.begin(), displayResults.begin() + offsetInPage);
			}
			if (displayResults.size() > count) {
				displayResults.resize(count);
			}
		}
	} else {
		// 从内存中获取
		for (size_t i = startIndex; i < min(startIndex + count, lastCarveResults.size()); i++) {
			displayResults.push_back(lastCarveResults[i]);
		}
	}

	// 应用筛选
	if (filterDeleted || filterActive) {
		vector<CarvedFileInfo> filtered;
		for (const auto& info : displayResults) {
			if (filterDeleted) {
				// 包含已删除的文件，或者未检查状态的文件
				if (!info.deletionChecked || info.isDeleted) {
					filtered.push_back(info);
				}
			} else if (filterActive) {
				// 只包含确认的活动文件
				if (info.deletionChecked && info.isActiveFile) {
					filtered.push_back(info);
				}
			}
		}
		displayResults = filtered;
	}

	// 显示结果
	cout << "\n=== Carved Files List ===" << endl;
	if (filterDeleted) {
		cout << "[Filter: Deleted files only]" << endl;
	} else if (filterActive) {
		cout << "[Filter: Active files only]" << endl;
	}

	if (isPageMode) {
		size_t totalPages = (totalCount + pageSize - 1) / pageSize;
		cout << "Page " << (currentPageIndex + 1) << " of " << totalPages;
		cout << " (" << displayResults.size() << " files shown";
		if (filterDeleted || filterActive) {
			cout << " after filter";
		}
		cout << ", total: " << totalCount << ")" << endl;
	} else {
		cout << "Showing " << displayResults.size() << " files starting from #" << startIndex;
		cout << " (total: " << totalCount << ")" << endl;
	}
	cout << string(80, '-') << endl;

	for (size_t i = 0; i < displayResults.size(); i++) {
		const auto& info = displayResults[i];
		size_t globalIndex = startIndex + i;

		cout << "\n[" << globalIndex << "] " << info.description << " (." << info.extension << ")";

		// 显示删除状态标记
		if (info.deletionChecked) {
			if (info.isDeleted) {
				cout << " [DELETED]";
			} else if (info.isActiveFile) {
				cout << " [ACTIVE]";
			}
		}
		cout << endl;

		cout << "    LCN: " << info.startLCN << " | Offset: " << info.startOffset << endl;
		cout << "    Size: " << info.fileSize << " bytes (" << (info.fileSize / 1024) << " KB)" << endl;
		cout << "    Confidence: " << (int)(info.confidence * 100) << "%" << endl;

		// 时间戳信息
		if (info.tsSource != TS_NONE_1) {
			cout << "    Timestamp Source: " << GetTimestampSourceStr(info.tsSource) << endl;

			if (info.creationTime.dwHighDateTime != 0 || info.creationTime.dwLowDateTime != 0) {
				cout << "    Created:  " << FileTimeToString(info.creationTime) << endl;
			}
			if (info.modificationTime.dwHighDateTime != 0 || info.modificationTime.dwLowDateTime != 0) {
				cout << "    Modified: " << FileTimeToString(info.modificationTime) << endl;
			}
			if (info.accessTime.dwHighDateTime != 0 || info.accessTime.dwLowDateTime != 0) {
				cout << "    Accessed: " << FileTimeToString(info.accessTime) << endl;
			}

			if (!info.embeddedInfo.empty()) {
				cout << "    Metadata: " << info.embeddedInfo << endl;
			}
			if (info.matchedMftRecord > 0) {
				cout << "    MFT Record: #" << info.matchedMftRecord << endl;
			}
		}

		// 完整性信息
		if (info.integrityValidated) {
			string status;
			if (info.integrityScore >= 0.8) status = "[OK]";
			else if (info.integrityScore >= 0.5) status = "[WARN]";
			else status = "[FAIL]";
			cout << "    Integrity: " << status << " " << (int)(info.integrityScore * 100) << "% - " << info.integrityDiagnosis << endl;
		}
	}

	cout << string(80, '-') << endl;
	cout << "Total: " << totalCount << " files" << endl;

	// 显示删除状态统计
	if (!lastCarveResults.empty() && lastCarveResults[0].deletionChecked) {
		size_t deletedTotal = 0, activeTotal = 0;
		for (const auto& info : lastCarveResults) {
			if (info.deletionChecked) {
				if (info.isDeleted) deletedTotal++;
				else if (info.isActiveFile) activeTotal++;
			}
		}
		cout << "Status: " << deletedTotal << " deleted, " << activeTotal << " active" << endl;
	}

	// 显示分页提示
	if (isPageMode && totalCount > pageSize) {
		size_t totalPages = (totalCount + pageSize - 1) / pageSize;
		cout << "\nNavigation:" << endl;
		if (currentPageIndex > 0) {
			cout << "  carvelist " << (currentPageIndex - 1) << " - Previous page" << endl;
		}
		if (currentPageIndex + 1 < totalPages) {
			cout << "  carvelist " << (currentPageIndex + 1) << " - Next page" << endl;
		}
		cout << "  carvelist <page_number> - Jump to specific page" << endl;
	}

	cout << "\nCommands:" << endl;
	cout << "  carvelist <page>           - Show page (default: " << pageSize << " items/page)" << endl;
	cout << "  carvelist deleted [page]   - Show only deleted files" << endl;
	cout << "  carvelist active [page]    - Show only active (not deleted) files" << endl;
	cout << "  carvelist <start> <count>  - Show custom range" << endl;
	cout << "  carvetimestamp             - Extract timestamps for all files" << endl;
	cout << "  carvevalidate              - Validate file integrity" << endl;
	cout << "  carverecover <idx> <path>  - Recover specific file" << endl;
}

// ============================================================================
// CarveValidateCommand - 验证 carved 文件完整性
// ============================================================================
DEFINE_COMMAND_BASE(CarveValidateCommand, "carvevalidate |name |name", TRUE)
REGISTER_COMMAND(CarveValidateCommand);

void CarveValidateCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (lastCarveResults.empty()) {
		cout << "No carving results available. Run 'carve' or 'carvepool' first." << endl;
		return;
	}

	// 解析参数
	double minScore = FileIntegrityValidator::MIN_INTEGRITY_SCORE;
	bool filterResults = false;

	if (HAS_ARG(0)) {
		string& arg0 = GET_ARG_STRING(0);
		if (arg0 == "filter") {
			filterResults = true;
			if (HAS_ARG(1)) {
				try {
					minScore = stod(GET_ARG_STRING(1));
					if (minScore < 0 || minScore > 1) minScore = 0.5;
				} catch (...) {}
			}
		} else {
			try {
				minScore = stod(arg0);
				if (minScore < 0 || minScore > 1) minScore = 0.5;
			} catch (...) {}
		}
	}

	try {
		MFTReader reader;
		if (!reader.OpenVolume(lastCarveDrive)) {
			cout << "Failed to open volume " << lastCarveDrive << ":" << endl;
			return;
		}

		FileCarver carver(&reader);

		cout << "\n=== File Integrity Validation ===" << endl;
		cout << "Files to validate: " << lastCarveResults.size() << endl;
		cout << "Minimum score threshold: " << fixed << setprecision(2) << minScore << endl;

		// 执行完整性验证
		carver.ValidateIntegrityForResults(lastCarveResults, true);

		// 显示详细结果
		cout << "\n=== Validation Results ===" << endl;
		cout << string(90, '-') << endl;

		size_t displayCount = 0;
		size_t corruptedCount = 0;
		size_t intactCount = 0;

		for (size_t i = 0; i < lastCarveResults.size() && displayCount < 30; i++) {
			const auto& info = lastCarveResults[i];

			if (info.integrityValidated) {
				string status;
				if (info.integrityScore >= 0.8) {
					status = "[OK]";
					intactCount++;
				} else if (info.integrityScore >= 0.5) {
					status = "[WARN]";
					intactCount++;
				} else {
					status = "[FAIL]";
					corruptedCount++;
				}

				cout << "[" << setw(3) << i << "] " << status << " ";
				cout << setw(6) << info.extension << " | ";
				cout << setw(3) << (int)(info.integrityScore * 100) << "% | ";
				cout << info.integrityDiagnosis << endl;

				displayCount++;
			}
		}

		cout << string(90, '-') << endl;
		cout << "Summary: " << intactCount << " intact, " << corruptedCount << " corrupted" << endl;

		// 如果需要过滤结果
		if (filterResults) {
			vector<CarvedFileInfo> filtered = carver.FilterCorruptedFiles(lastCarveResults, minScore);
			cout << "\nFiltered results: " << filtered.size() << " files (score >= "
			     << fixed << setprecision(2) << minScore << ")" << endl;
			lastCarveResults = filtered;
		}

		cout << "\nCommands:" << endl;
		cout << "  carvevalidate filter <min_score> - Filter out corrupted files" << endl;
		cout << "  carverecover <idx> <path>        - Recover specific file" << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// CarveIntegrityCommand - 检查单个文件完整性（详细模式）
// ============================================================================
DEFINE_COMMAND_BASE(CarveIntegrityCommand, "carveintegrity |name", TRUE)
REGISTER_COMMAND(CarveIntegrityCommand);

void CarveIntegrityCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 1) {
		cout << "Usage: carveintegrity <index>" << endl;
		cout << "Example: carveintegrity 0" << endl;
		return;
	}

	if (lastCarveResults.empty()) {
		cout << "No carving results available. Run 'carve' or 'carvepool' first." << endl;
		return;
	}

	try {
		size_t index = stoull(GET_ARG_STRING(0));

		if (index >= lastCarveResults.size()) {
			cout << "Invalid index. Valid range: 0-" << (lastCarveResults.size() - 1) << endl;
			return;
		}

		MFTReader reader;
		if (!reader.OpenVolume(lastCarveDrive)) {
			cout << "Failed to open volume " << lastCarveDrive << ":" << endl;
			return;
		}

		FileCarver carver(&reader);
		const CarvedFileInfo& info = lastCarveResults[index];

		cout << "\n=== Detailed Integrity Analysis ===" << endl;
		cout << "File index: " << index << endl;
		cout << "Type: " << info.description << " (." << info.extension << ")" << endl;
		cout << "Size: " << info.fileSize << " bytes" << endl;
		cout << "LCN: " << info.startLCN << endl;

		// 执行详细验证
		FileIntegrityScore score = carver.ValidateFileIntegrity(info);

		cout << "\n--- Entropy Analysis ---" << endl;
		cout << "  Raw entropy: " << fixed << setprecision(3) << score.entropy << " bits/byte" << endl;
		cout << "  Entropy score: " << setprecision(1) << (score.entropyScore * 100) << "%" << endl;
		cout << "  Anomaly detected: " << (score.hasEntropyAnomaly ? "YES" : "No") << endl;
		if (score.hasEntropyAnomaly) {
			cout << "  Anomaly offset: block #" << score.anomalyOffset << endl;
		}

		cout << "\n--- Structure Validation ---" << endl;
		cout << "  Structure score: " << setprecision(1) << (score.structureScore * 100) << "%" << endl;
		cout << "  Valid header: " << (score.hasValidHeader ? "Yes" : "NO") << endl;
		cout << "  Valid footer: " << (score.hasValidFooter ? "Yes" : "NO") << endl;

		cout << "\n--- Statistical Analysis ---" << endl;
		cout << "  Zero ratio: " << setprecision(2) << (score.zeroRatio * 100) << "%" << endl;
		cout << "  Chi-square: " << setprecision(1) << score.chiSquare << endl;
		cout << "  Statistical score: " << setprecision(1) << (score.statisticalScore * 100) << "%" << endl;

		cout << "\n--- Footer Validation ---" << endl;
		cout << "  Footer score: " << setprecision(1) << (score.footerScore * 100) << "%" << endl;

		cout << "\n=== Overall Assessment ===" << endl;
		cout << "Overall score: " << setprecision(1) << (score.overallScore * 100) << "%" << endl;
		cout << "Diagnosis: " << score.diagnosis << endl;
		cout << "Likely corrupted: " << (score.isLikelyCorrupted ? "YES" : "No") << endl;

		// 更新结果
		lastCarveResults[index].integrityScore = score.overallScore;
		lastCarveResults[index].integrityValidated = true;
		lastCarveResults[index].integrityDiagnosis = score.diagnosis;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// CarveRecoverPageCommand - 分页交互式恢复命令
// 支持：逐页浏览、选择性恢复、自动清理、低置信度强制恢复
// ============================================================================
DEFINE_COMMAND_BASE(CarveRecoverPageCommand, "crp |file |name |name |name |name", TRUE)
REGISTER_COMMAND(CarveRecoverPageCommand);

// 辅助函数：获取目录中的文件数量
static size_t GetFileCountInDirectory(const string& dirPath) {
	size_t count = 0;
	WIN32_FIND_DATAA findData;
	string searchPath = dirPath;
	if (searchPath.back() != '\\' && searchPath.back() != '/') {
		searchPath += '\\';
	}
	searchPath += "*";

	HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				count++;
			}
		} while (FindNextFileA(hFind, &findData));
		FindClose(hFind);
	}
	return count;
}

// 辅助函数：清空目录中的文件
static size_t ClearDirectory(const string& dirPath) {
	size_t deletedCount = 0;
	WIN32_FIND_DATAA findData;
	string searchPath = dirPath;
	if (searchPath.back() != '\\' && searchPath.back() != '/') {
		searchPath += '\\';
	}
	string basePath = searchPath;
	searchPath += "*";

	HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				string filePath = basePath + findData.cFileName;
				if (DeleteFileA(filePath.c_str())) {
					deletedCount++;
				}
			}
		} while (FindNextFileA(hFind, &findData));
		FindClose(hFind);
	}
	return deletedCount;
}

// 辅助函数：解析索引列表（如 "0 2 4" 或 "0,2,4"）
static vector<size_t> ParseIndices(const string& input, size_t maxIndex) {
	vector<size_t> indices;
	string token;
	string cleanInput = input;

	// 替换逗号为空格
	for (char& c : cleanInput) {
		if (c == ',') c = ' ';
	}

	istringstream iss(cleanInput);
	while (iss >> token) {
		try {
			size_t idx = stoull(token);
			if (idx <= maxIndex) {
				indices.push_back(idx);
			}
		} catch (...) {
			// 忽略无效输入
		}
	}

	return indices;
}

void CarveRecoverPageCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 1) {
		cout << "\n=== Paginated Interactive Recovery ===" << endl;
		cout << "Usage: crp <output_dir> [options...]" << endl;
		cout << "\nOptions:" << endl;
		cout << "  minconf=<0-100>   - Minimum confidence to show (default: 50)" << endl;
		cout << "  pagesize=<N>      - Files per page (default: 10)" << endl;
		cout << "  autoclean=<N>     - Auto-clean folder when file count >= N (default: 50)" << endl;
		cout << "  deleted           - Show only deleted files (default)" << endl;
		cout << "  all               - Show all files (including active)" << endl;
		cout << "\nExamples:" << endl;
		cout << "  crp D:\\recovered\\" << endl;
		cout << "  crp D:\\recovered\\ minconf=30 pagesize=20" << endl;
		cout << "  crp D:\\recovered\\ autoclean=100 all" << endl;
		cout << "\nInteractive Commands:" << endl;
		cout << "  r          - Recover all files on current page" << endl;
		cout << "  r <idx...> - Recover specific files (e.g., r 0 2 4)" << endl;
		cout << "  f <idx...> - Force recover low-confidence files" << endl;
		cout << "  n          - Next page" << endl;
		cout << "  p          - Previous page" << endl;
		cout << "  g <page>   - Go to specific page" << endl;
		cout << "  c          - Clear output folder" << endl;
		cout << "  q          - Quit recovery mode" << endl;
		cout << "\nRun 'carve' or 'carvepool' first to scan for files." << endl;
		return;
	}

	// 检查是否有扫描结果
	size_t totalCount = GetTotalResultCount();
	if (totalCount == 0) {
		cout << "No carving results available. Run 'carve' or 'carvepool' first." << endl;
		return;
	}

	try {
		// 解析参数
		string outputDir = GET_ARG_STRING(0);
		double minConfidence = 0.50;
		size_t pageSize_ = 10;
		size_t autoCleanThreshold = 50;
		bool showDeletedOnly = true;

		for (size_t i = 1; i < GET_ARG_COUNT(); i++) {
			string arg = GET_ARG_STRING(i);

			if (arg.find("minconf=") == 0) {
				try {
					int conf = stoi(arg.substr(8));
					minConfidence = conf / 100.0;
					if (minConfidence < 0) minConfidence = 0;
					if (minConfidence > 1) minConfidence = 1;
				} catch (...) {}
			}
			else if (arg.find("pagesize=") == 0) {
				try {
					pageSize_ = stoull(arg.substr(9));
					if (pageSize_ < 1) pageSize_ = 1;
					if (pageSize_ > 100) pageSize_ = 100;
				} catch (...) {}
			}
			else if (arg.find("autoclean=") == 0) {
				try {
					autoCleanThreshold = stoull(arg.substr(10));
				} catch (...) {}
			}
			else if (arg == "all") {
				showDeletedOnly = false;
			}
			else if (arg == "deleted" || arg == "del") {
				showDeletedOnly = true;
			}
		}

		// 创建输出目录
		if (!CommandUtils::CreateOutputDirectory(outputDir)) {
			cout << "Warning: Could not create output directory" << endl;
		}

		// 确保路径以反斜杠结尾
		if (outputDir.back() != '\\' && outputDir.back() != '/') {
			outputDir += '\\';
		}

		// 收集符合条件的文件
		vector<pair<size_t, CarvedFileInfo>> filteredResults;
		for (size_t i = 0; i < totalCount; i++) {
			CarvedFileInfo info;
			char drive;
			if (GetSingleResult(i, info, drive)) {
				// 置信度过滤
				if (info.confidence < minConfidence) continue;

				// 删除状态过滤
				if (showDeletedOnly) {
					if (info.deletionChecked && info.isActiveFile) continue;
				}

				filteredResults.push_back({i, info});
			}
		}

		if (filteredResults.empty()) {
			cout << "No files match the specified criteria." << endl;
			cout << "Try lowering minconf or using 'all' option." << endl;
			return;
		}

		// 打开卷
		MFTReader reader;
		if (!reader.OpenVolume(lastCarveDrive)) {
			cout << "Failed to open volume " << lastCarveDrive << ":" << endl;
			return;
		}
		FileCarver carver(&reader);

		// 开始交互循环
		size_t currentPage = 0;
		size_t totalPages = (filteredResults.size() + pageSize_ - 1) / pageSize_;
		size_t totalRecovered = 0;

		cout << "\n=== Paginated Recovery Mode ===" << endl;
		cout << "Output: " << outputDir << endl;
		cout << "Files matching criteria: " << filteredResults.size() << endl;
		cout << "Min confidence: " << (int)(minConfidence * 100) << "%" << endl;
		cout << "Page size: " << pageSize_ << endl;
		cout << "Auto-clean threshold: " << autoCleanThreshold << " files" << endl;
		cout << "Filter: " << (showDeletedOnly ? "Deleted only" : "All files") << endl;
		cout << "\nCommands: r=recover, f=force, n=next, p=prev, g=goto, c=clear, q=quit" << endl;

		while (true) {
			// 显示当前页
			cout << "\n" << string(70, '=') << endl;
			cout << "Page " << (currentPage + 1) << " of " << totalPages << endl;
			cout << string(70, '-') << endl;

			size_t startIdx = currentPage * pageSize_;
			size_t endIdx = min(startIdx + pageSize_, filteredResults.size());

			for (size_t i = startIdx; i < endIdx; i++) {
				const auto& [globalIdx, info] = filteredResults[i];
				size_t pageIdx = i - startIdx;

				cout << "[" << pageIdx << "] #" << globalIdx << " | ";
				cout << setw(6) << info.extension << " | ";
				cout << setw(8) << (info.fileSize / 1024) << " KB | ";

				// 置信度颜色提示
				int conf = (int)(info.confidence * 100);
				if (conf >= 80) {
					cout << setw(3) << conf << "% [HIGH]";
				} else if (conf >= 50) {
					cout << setw(3) << conf << "% [MED]";
				} else {
					cout << setw(3) << conf << "% [LOW!]";
				}

				// 时间戳
				if (info.tsSource != TS_NONE_1 &&
					(info.modificationTime.dwHighDateTime != 0 || info.modificationTime.dwLowDateTime != 0)) {
					SYSTEMTIME st;
					FileTimeToSystemTime(&info.modificationTime, &st);
					printf(" | %04d-%02d-%02d", st.wYear, st.wMonth, st.wDay);
				}

				// 删除状态
				if (info.deletionChecked) {
					if (info.isDeleted) cout << " [DEL]";
					else if (info.isActiveFile) cout << " [ACT]";
				}

				cout << endl;
			}

			cout << string(70, '-') << endl;

			// 检查输出目录文件数量
			size_t outputFileCount = GetFileCountInDirectory(outputDir);
			cout << "Output folder: " << outputFileCount << " files";
			if (outputFileCount >= autoCleanThreshold) {
				cout << " [!] Auto-clean threshold reached!";
			}
			cout << endl;
			cout << "Recovered this session: " << totalRecovered << endl;

			// 获取用户输入
			cout << "\nCommand> ";
			string input;
			getline(cin, input);

			// 去除首尾空格
			size_t start = input.find_first_not_of(" \t");
			size_t end = input.find_last_not_of(" \t");
			if (start == string::npos) {
				input = "";
			} else {
				input = input.substr(start, end - start + 1);
			}

			if (input.empty()) continue;

			// 解析命令
			char cmd = tolower(input[0]);
			string args = input.length() > 1 ? input.substr(1) : "";

			// 去除args首部空格
			start = args.find_first_not_of(" \t");
			if (start != string::npos) {
				args = args.substr(start);
			} else {
				args = "";
			}

			switch (cmd) {
				case 'q':  // 退出
					cout << "\n=== Recovery Session Complete ===" << endl;
					cout << "Total recovered: " << totalRecovered << " files" << endl;
					cout << "Output directory: " << outputDir << endl;
					return;

				case 'n':  // 下一页
					if (currentPage + 1 < totalPages) {
						currentPage++;
					} else {
						cout << "Already on last page." << endl;
					}
					break;

				case 'p':  // 上一页
					if (currentPage > 0) {
						currentPage--;
					} else {
						cout << "Already on first page." << endl;
					}
					break;

				case 'g':  // 跳转到指定页
					{
						try {
							size_t targetPage = stoull(args);
							if (targetPage >= 1 && targetPage <= totalPages) {
								currentPage = targetPage - 1;
							} else {
								cout << "Invalid page number. Valid range: 1-" << totalPages << endl;
							}
						} catch (...) {
							cout << "Usage: g <page_number>" << endl;
						}
					}
					break;

				case 'c':  // 清空输出目录
					{
						cout << "Clear all files in " << outputDir << "? (y/n): ";
						string confirm;
						getline(cin, confirm);
						if (!confirm.empty() && (confirm[0] == 'y' || confirm[0] == 'Y')) {
							size_t deleted = ClearDirectory(outputDir);
							cout << "Deleted " << deleted << " files." << endl;
						}
					}
					break;

				case 'r':  // 恢复文件
				case 'f':  // 强制恢复（允许低置信度）
					{
						// 检查是否需要自动清理
						size_t currentFileCount = GetFileCountInDirectory(outputDir);
						if (currentFileCount >= autoCleanThreshold) {
							cout << "\n[!] Output folder has " << currentFileCount << " files." << endl;
							cout << "Clear folder before continuing? (y/n/skip): ";
							string confirm;
							getline(cin, confirm);
							if (!confirm.empty() && (confirm[0] == 'y' || confirm[0] == 'Y')) {
								size_t deleted = ClearDirectory(outputDir);
								cout << "Deleted " << deleted << " files." << endl;
							} else if (!confirm.empty() && (confirm[0] == 's' || confirm[0] == 'S')) {
								// skip - 继续但不清理
							} else {
								cout << "Skipping recovery. Use 'c' to clear manually." << endl;
								break;
							}
						}

						vector<size_t> indicesToRecover;
						bool forceMode = (cmd == 'f');

						if (args.empty()) {
							// 恢复当前页所有文件
							for (size_t i = 0; i < endIdx - startIdx; i++) {
								indicesToRecover.push_back(i);
							}
						} else {
							// 恢复指定索引
							indicesToRecover = ParseIndices(args, endIdx - startIdx - 1);
						}

						if (indicesToRecover.empty()) {
							cout << "No valid indices specified." << endl;
							break;
						}

						size_t pageRecovered = 0;
						size_t pageSkipped = 0;

						for (size_t pageIdx : indicesToRecover) {
							size_t filteredIdx = startIdx + pageIdx;
							if (filteredIdx >= filteredResults.size()) continue;

							const auto& [globalIdx, info] = filteredResults[filteredIdx];

							// 检查置信度
							if (info.confidence < 0.3 && !forceMode) {
								cout << "  [" << pageIdx << "] Skipped (confidence " << (int)(info.confidence * 100) << "% < 30%). Use 'f' to force." << endl;
								pageSkipped++;
								continue;
							}

							// 生成文件名
							string filename = "crp_" + to_string(totalRecovered + pageRecovered);
							if (info.tsSource != TS_NONE_1 &&
								(info.modificationTime.dwHighDateTime != 0 || info.modificationTime.dwLowDateTime != 0)) {
								SYSTEMTIME st;
								FileTimeToSystemTime(&info.modificationTime, &st);
								char dateStr[20];
								sprintf_s(dateStr, "_%04d%02d%02d", st.wYear, st.wMonth, st.wDay);
								filename += dateStr;
							}
							filename += "." + info.extension;

							string outputPath = outputDir + filename;

							if (carver.RecoverCarvedFile(info, outputPath)) {
								cout << "  [" << pageIdx << "] Recovered: " << filename;
								if (forceMode && info.confidence < 0.5) {
									cout << " [FORCED]";
								}
								cout << endl;
								pageRecovered++;
							} else {
								cout << "  [" << pageIdx << "] FAILED to recover" << endl;
							}
						}

						totalRecovered += pageRecovered;
						cout << "\nPage recovery: " << pageRecovered << " succeeded, " << pageSkipped << " skipped" << endl;
					}
					break;

				default:
					cout << "Unknown command. Use: r, f, n, p, g, c, q" << endl;
					break;
			}
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}