#include "cmd.h"
#include <vector>
#include <iostream>
#include <Windows.h>
#include <queue>
#include "cli.h"
#include "ImageTable.h"
#include "FileRestore.h"
#include "DeletedFileScanner.h"
#include "MFTReader.h"
#include "OverwriteDetector.h"
#include "UsnJournalReader.h"
#include "MFTParser.h"
#include "PathResolver.h"
#include <algorithm>
using namespace std;
	 vector<LPVOID> HelpCommand::ArgsList = vector<LPVOID>();
	 vector<LPVOID> QueueDLLsCommand::ArgsList = vector<LPVOID>();
	 vector<LPVOID> GetProcessFuncAddressCommand::ArgsList = vector<LPVOID>();
	 // IATHookDLLCommand removed - static members removed
	 vector<LPVOID> ExitCommand::ArgsList = vector<LPVOID>();
	 vector<LPVOID> PrintAllFunction::Arglist = vector<LPVOID>();
	 // IATHookByNameCommand removed - static members removed
	 // IATHookByCreateProc removed - static members removed
	 // ElevateAdminPrivilegeCommand removed - static members removed
	 // ElevateSystemPrivilegeCommand removed - static members removed
	 vector<LPVOID> ListDeletedFilesCommand::ArgsList = vector<LPVOID>();
	 vector<LPVOID> RestoreByRecordCommand::ArgsList = vector<LPVOID>();
	 vector<LPVOID> DiagnoseMFTCommand::ArgsList = vector<LPVOID>();
	 vector<LPVOID> DetectOverwriteCommand::ArgsList = vector<LPVOID>();
	 vector<LPVOID> SearchDeletedFilesCommand::ArgsList = vector<LPVOID>();
	 vector<LPVOID> ScanUsnCommand::ArgsList = vector<LPVOID>();
	 vector<LPVOID> DiagnoseFileCommand::ArgsList = vector<LPVOID>();
	 // IATHookByCreateProc::pid removed
 PrintAllCommand::PrintAllCommand() {
		FlagHasArgs = FALSE;
     }
    void PrintAllCommand::AcceptArgs(vector<LPVOID> argslist){
		// This command does not accept any arguments
    }
	BOOL PrintAllCommand::CheckName(string input)
	{
		if (input.compare(name)==0) {
			return true;
		}
		return false;
	}
    void PrintAllCommand::Execute(string command) {
		if (!CheckName(command)) {
			return;
		}
		vector<queue<string>> allCommands = CLI::GetCommands();
		for (auto& cmdQueue : allCommands) {
			string output = "";
			queue<string> tempQueue = cmdQueue;
			int count = 0;
			int size = tempQueue.size();// Create a copy to preserve the original
			while (!tempQueue.empty()) {
				if (count<size) {
					output += tempQueue.front() + " ";
					tempQueue.pop();
				}
				else {
					output += tempQueue.front();
					tempQueue.pop();
				}
				count++;
			}
			cout << output << endl;
		}
    }
    BOOL PrintAllCommand::HasArgs()  {
		return FlagHasArgs;
    }
    HelpCommand::HelpCommand() {
		FlagHasArgs = TRUE;
    }
	void HelpCommand::AcceptArgs(vector<LPVOID> argslist)  {
		HelpCommand::ArgsList = argslist;
	}
	void HelpCommand::Execute(string command)  {
		if (!CheckName(command)) {
			return;
		}
		vector<string> currectcommands = vector<string>();
		cout << "Available args:" << endl;
		cout << "|file is the place where you put the file you want to analyze" << endl;
		cout << "|pid is the place where you put process id" << endl;
		cout << "|name is the place where you put function name or process name" << endl;
		if (ArgsList.size()==1) {
			for (auto command : CLI::GetCommands()) {
				if (command.front().compare(*(string*)ArgsList[0]) == 0) {
					// IATHook help text removed - command no longer available
					string currectcommand = "";
					while (!command.empty()) {
						currectcommand += command.front() + " ";
						command.pop();
					}
					currectcommands.push_back(currectcommand);
				}
			}
			for (auto command:currectcommands) {
				cout << "    "+command << endl;
			}
		}
		else {
			return;
		}
	}
	BOOL HelpCommand::HasArgs(){
		return FlagHasArgs;
	}
	BOOL HelpCommand::CheckName(string input)
	{
		if (input.compare(name) == 0) {
			return true;
		}
		return false;
	}
	QueueDLLsCommand::QueueDLLsCommand()
	{
		
		FlagHasArgs = TRUE;
	}
	QueueDLLsCommand::~QueueDLLsCommand()
	{
	}
	void QueueDLLsCommand::AcceptArgs(vector<LPVOID> theargslist)
	{
		QueueDLLsCommand::ArgsList = theargslist;
	}
	void QueueDLLsCommand::Execute(string command)
	{
		if (!CheckName(command)) {
			return;
		}
		ImageTableAnalyzer* analyzer = new ImageTableAnalyzer();
		if (ArgsList.size()!=1) {
			cout << "Invaild Args!" << endl;
		}
		else {
			string& pefile = *(string*)ArgsList[0];
			vector<string> dlllist = analyzer->AnalyzeTableForDLL(pefile);
			if (dlllist.size() != 0) {
				for (int i = 0; i < dlllist.size(); i++) {
					cout << "    "+dlllist[i] << endl;
				}
			}
			else {
				cout << "can't find the IAT" << endl;
			}
		}
		delete analyzer;
	}

	BOOL QueueDLLsCommand::HasArgs()
	{
		return TRUE;
	}

	BOOL QueueDLLsCommand::CheckName(string input)
	{
		if (input.compare(name) == 0) {
			return true;
		}
		return false;
	}

	GetProcessFuncAddressCommand::GetProcessFuncAddressCommand()
	{
		FlagHasArgs = TRUE;
	}
	GetProcessFuncAddressCommand::~GetProcessFuncAddressCommand()
	{
	}
	void GetProcessFuncAddressCommand::AcceptArgs(vector<LPVOID> argslist)
	{
		GetProcessFuncAddressCommand::ArgsList = argslist;
	}
	void GetProcessFuncAddressCommand::Execute(string command)
	{
		if (!CheckName(command)) {
			return;
		}
		ImageTableAnalyzer* analyzer = new ImageTableAnalyzer();
		if (ArgsList.size() != 2) {
			cout << "Invaild Args!" << endl;
		}
		else {
			string& funcname = *(string*)ArgsList[1];
			string& pefile = *(string*)ArgsList[0];
			ULONGLONG funcaddress = analyzer->GetFuncaddressByName(funcname, pefile);
			if (funcaddress != 0) {
				cout << "Function Address: 0x" << hex << funcaddress << endl;
			}
			else {
				cout << "can't find the function address" << endl;
			}
		}
		delete analyzer;
	}
	BOOL GetProcessFuncAddressCommand::HasArgs()
	{
		return FlagHasArgs;
	}
	BOOL GetProcessFuncAddressCommand::CheckName(string input)
	{
		if (input.compare(name) == 0) {
			return true;
		}
		return false;
	}
	// ==================== IATHookDLLCommand REMOVED ====================
	// IAT Hook functionality has been removed from public version
	// Original implementation: lines 214-262


	ExitCommand::ExitCommand()
	{
		FlagHasArgs = FALSE;
	}
	void ExitCommand::AcceptArgs(vector<LPVOID> argslist)
	{
	}
	void ExitCommand::Execute(string command)
	{
		if (!CheckName(command)) {
			return;
		}
		cout << "Exiting the application." << endl;
		exit(0);
	}
	BOOL ExitCommand::HasArgs()
	{
		return FlagHasArgs;
	}

	BOOL ExitCommand::CheckName(string input)
	{
		if (input.compare(name) == 0) {
			return true;
		}
		return false;
	}
	PrintAllFunction::PrintAllFunction()
	{
		FlagHasArgs = true;
	}

	PrintAllFunction::~PrintAllFunction()
	{
		delete analyzer;
	}
	void PrintAllFunction::AcceptArgs(vector<LPVOID> argslist)
	{
		PrintAllFunction::Arglist = argslist;
	}
	void PrintAllFunction::Execute(string command)
	{
		if (Arglist.size()==1) {
			string file = *(string*)Arglist[0];
			map<string,vector<string>> funclist = analyzer->AnalyzeTableForFunctions(file);
			vector<string> dllNames = analyzer->AnalyzeTableForDLL(file);
			for (auto dllName :dllNames) {
				vector<string> value = funclist[dllName];
				cout << dllName + ":" << endl;
				for (auto funcname:value) {
					ULONGLONG funcaddr = analyzer->GetFuncaddressByName(funcname,file);
					cout << "    " + funcname +"  " +"FunctionAddress:" +"0x" << hex << funcaddr << endl;
				}
			}
		}
	}
	BOOL PrintAllFunction::HasArgs()
	{
		return FlagHasArgs;
	}
	BOOL PrintAllFunction::CheckName(string input)
	{
		if (input.compare(name) == 0) {
			return true;
		}
		return false;
	}
	// ==================== IATHookByNameCommand REMOVED ====================
	// IAT Hook functionality has been removed from public version

	// ==================== IATHookByCreateProc REMOVED ====================
	// IAT Hook functionality has been removed from public version

	// ==================== ElevateAdminPrivilegeCommand REMOVED ====================
	// Privilege elevation functionality has been removed from public version

	// ==================== ElevateSystemPrivilegeCommand REMOVED ====================
	// Privilege elevation functionality has been removed from public version

	DiagnoseMFTCommand::DiagnoseMFTCommand()
	{
		FlagHasArgs = TRUE;
	}

	DiagnoseMFTCommand::~DiagnoseMFTCommand()
	{
	}

	void DiagnoseMFTCommand::AcceptArgs(vector<LPVOID> argslist)
	{
		DiagnoseMFTCommand::ArgsList = argslist;
	}

	void DiagnoseMFTCommand::Execute(string command)
	{
		if (!CheckName(command)) {
			return;
		}

		if (ArgsList.size() != 1) {
			cout << "Usage: diagnosemft <drive_letter>" << endl;
			cout << "Example: diagnosemft C" << endl;
			return;
		}

		string& driveStr = *(string*)ArgsList[0];
		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		// 打开卷
		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "Failed to open volume " << driveLetter << ":/" << endl;
			return;
		}

		// 执行诊断
		reader.DiagnoseMFTFragmentation();
	}

	BOOL DiagnoseMFTCommand::CheckName(string input)
	{
		if (input.compare(name) == 0) {
			return TRUE;
		}
		return FALSE;
	}

	BOOL DiagnoseMFTCommand::HasArgs()
	{
		return FlagHasArgs;
	}

	// ==================== DetectOverwriteCommand ====================

	DetectOverwriteCommand::DetectOverwriteCommand()
	{
		FlagHasArgs = TRUE;
	}

	DetectOverwriteCommand::~DetectOverwriteCommand()
	{
	}

	void DetectOverwriteCommand::AcceptArgs(vector<LPVOID> argslist)
	{
		DetectOverwriteCommand::ArgsList = argslist;
	}

	void DetectOverwriteCommand::Execute(string command)
	{
		if (!CheckName(command)) {
			return;
		}

		if (ArgsList.size() < 2) {
			cout << "Invalid Args! Usage: detectoverwrite <drive_letter> <MFT_record_number> [mode]" << endl;
			cout << "Example: detectoverwrite C 1234" << endl;
			cout << "Example: detectoverwrite C 1234 fast" << endl;
			cout << endl;
			cout << "Modes:" << endl;
			cout << "  fast      - Quick sampling detection (fastest)" << endl;
			cout << "  balanced  - Smart detection (default)" << endl;
			cout << "  thorough  - Complete detection (slowest, most accurate)" << endl;
			return;
		}

		string& driveStr = *(string*)ArgsList[0];
		string& recordStr = *(string*)ArgsList[1];

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];
		ULONGLONG recordNumber = 0;

		try {
			recordNumber = stoull(recordStr);
		}
		catch (...) {
			cout << "Invalid MFT record number." << endl;
			return;
		}

		// 检查是否指定了检测模式
		DetectionMode mode = MODE_BALANCED;  // 默认平衡模式
		if (ArgsList.size() >= 3) {
			string& modeStr = *(string*)ArgsList[2];
			if (modeStr == "fast") {
				mode = MODE_FAST;
			} else if (modeStr == "balanced") {
				mode = MODE_BALANCED;
			} else if (modeStr == "thorough") {
				mode = MODE_THOROUGH;
			} else {
				cout << "Unknown mode: " << modeStr << endl;
				cout << "Using default mode: balanced" << endl;
			}
		}

		cout << "=== Overwrite Detection ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "MFT Record: " << recordNumber << endl;

		string modeName;
		switch (mode) {
			case MODE_FAST: modeName = "Fast (Sampling)"; break;
			case MODE_BALANCED: modeName = "Balanced (Smart)"; break;
			case MODE_THOROUGH: modeName = "Thorough (Complete)"; break;
		}
		cout << "Detection Mode: " << modeName << endl;
		cout << endl;

		FileRestore* fileRestore = new FileRestore();
		OverwriteDetector* detector = fileRestore->GetOverwriteDetector();

		// 设置检测模式
		detector->SetDetectionMode(mode);

		// 执行检测
		OverwriteDetectionResult result = fileRestore->DetectFileOverwrite(driveLetter, recordNumber);

		// 显示详细结果
		cout << endl;
		cout << "=== Detection Summary ===" << endl;
		cout << "Storage Type: ";
		switch (result.detectedStorageType) {
			case STORAGE_HDD: cout << "HDD (Mechanical Hard Drive)"; break;
			case STORAGE_SSD: cout << "SATA SSD"; break;
			case STORAGE_NVME: cout << "NVMe SSD"; break;
			default: cout << "Unknown"; break;
		}
		cout << endl;

		if (result.usedMultiThreading) {
			cout << "Multi-Threading: Enabled (" << result.threadCount << " threads)" << endl;
		} else {
			cout << "Multi-Threading: Disabled" << endl;
		}

		if (result.usedSampling) {
			cout << "Sampling: Yes (" << result.sampledClusters << " out of " << result.totalClusters << " clusters)" << endl;
		}

		cout << "Detection Time: " << result.detectionTimeMs << " ms" << endl;
		cout << endl;

		cout << "=== Recovery Assessment ===" << endl;
		cout << "Total Clusters: " << result.totalClusters << endl;
		cout << "Available Clusters: " << result.availableClusters << endl;
		cout << "Overwritten Clusters: " << result.overwrittenClusters << endl;
		cout << "Overwrite Percentage: " << result.overwritePercentage << "%" << endl;
		cout << endl;

		if (result.isFullyAvailable) {
			cout << "Status: [EXCELLENT] Fully Recoverable" << endl;
			cout << "All data is available. Recovery should be 100% successful." << endl;
		} else if (result.isPartiallyAvailable) {
			cout << "Status: [WARNING] Partially Recoverable" << endl;
			cout << "Recovery Possibility: " << (100.0 - result.overwritePercentage) << "%" << endl;
			cout << "The recovered file may be corrupted or incomplete." << endl;
		} else {
			cout << "Status: [FAILED] Not Recoverable" << endl;
			cout << "All data has been overwritten. Recovery is not possible." << endl;
		}

		cout << endl;
		cout << "Use 'restorebyrecord " << driveLetter << " " << recordNumber << " <output_path>' to attempt recovery." << endl;

		delete fileRestore;
	}

	BOOL DetectOverwriteCommand::HasArgs()
	{
		return FlagHasArgs;
	}

	BOOL DetectOverwriteCommand::CheckName(string input)
	{
		if (input.compare(name) == 0) {
			return TRUE;
		}
		return FALSE;
	}

// ==================== SearchDeletedFilesCommand ====================

SearchDeletedFilesCommand::SearchDeletedFilesCommand()
{
	FlagHasArgs = TRUE;
}

SearchDeletedFilesCommand::~SearchDeletedFilesCommand()
{
}

void SearchDeletedFilesCommand::AcceptArgs(vector<LPVOID> argslist)
{
	SearchDeletedFilesCommand::ArgsList = argslist;
}

void SearchDeletedFilesCommand::Execute(string command)
{
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.size() < 2) {
		cout << "Invalid Args! Usage: searchdeleted <drive_letter> <filename_pattern> [extension] [filter_level]" << endl;
		cout << "Examples:" << endl;
		cout << "  searchdeleted C document         - Search for files containing 'document'" << endl;
		cout << "  searchdeleted C report .pdf      - Search for PDF files containing 'report'" << endl;
		cout << "  searchdeleted C * .jpg           - Search for all JPG files" << endl;
		cout << "  searchdeleted C * .xml skip      - Search for XML files with skip filter" << endl;
		cout << "\nFilter levels: none, skip (default), exclude" << endl;
		return;
	}

	try {
		string& driveStr = *(string*)ArgsList[0];
		string& pattern = *(string*)ArgsList[1];
		string extension = (ArgsList.size() >= 3) ? *(string*)ArgsList[2] : "";

		// 解析过滤级别参数（可选）
		FilterLevel filterLevel = FILTER_SKIP_PATH;  // 默认值
		if (ArgsList.size() >= 4) {
			string& filterStr = *(string*)ArgsList[3];
			if (filterStr == "none") {
				filterLevel = FILTER_NONE;
			} else if (filterStr == "skip") {
				filterLevel = FILTER_SKIP_PATH;
			} else if (filterStr == "exclude") {
				filterLevel = FILTER_EXCLUDE;
			} else {
				cout << "Unknown filter level: " << filterStr << ". Using default (skip)." << endl;
			}
		}

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		cout << "Searching drive " << driveLetter << ": for deleted files..." << endl;
		cout << "Pattern: " << pattern << endl;
		if (!extension.empty()) {
			cout << "Extension: " << extension << endl;
		}
		cout << "Filter level: ";
		switch (filterLevel) {
			case FILTER_NONE: cout << "None (all paths)"; break;
			case FILTER_SKIP_PATH: cout << "Skip path (default)"; break;
			case FILTER_EXCLUDE: cout << "Exclude low-value files"; break;
		}
		cout << endl;

		vector<DeletedFileInfo> allFiles;

		// 尝试从缓存加载
		bool usedCache = false;
		if (DeletedFileScanner::IsCacheValid(driveLetter, 60)) {
			cout << "Loading from cache (fast mode)..." << endl;
			if (DeletedFileScanner::LoadFromCache(allFiles, driveLetter)) {
				usedCache = true;
				cout << "Cache loaded: " << allFiles.size() << " files" << endl;
			}
		}

		// 如果缓存无效或加载失败，重新扫描
		if (!usedCache) {
			cout << "Cache not available. Scanning MFT (this may take a while)..." << endl;
			cout << "Tip: Results will be cached for faster future searches." << endl;

			FileRestore* fileRestore = new FileRestore();
			fileRestore->SetFilterLevel(filterLevel);
			allFiles = fileRestore->ScanDeletedFiles(driveLetter, 0);
			delete fileRestore;

			// 保存到缓存
			if (!allFiles.empty()) {
				cout << "Saving to cache..." << endl;
				DeletedFileScanner::SaveToCache(allFiles, driveLetter);
			}
		}

		if (allFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		// [DIAGNOSTIC] 显示样本文件名以验证扩展名
		cout << "\n[DIAGNOSTIC] Sample filenames from loaded data:" << endl;
		for (size_t i = 0; i < min((size_t)5, allFiles.size()); i++) {
			wcout << "  - fileName: \"" << allFiles[i].fileName << "\"" << endl;
		}
		cout << endl;

		// Apply filters
		vector<DeletedFileInfo> filtered = allFiles;
		cout << "[DIAGNOSTIC] Total files before filtering: " << filtered.size() << endl;

		// Filter by extension if specified
		if (!extension.empty() && extension != "*") {
			cout << "[DIAGNOSTIC] Filtering by extension: \"" << extension << "\"" << endl;

			wstring wext(extension.begin(), extension.end());
			wcout << "[DIAGNOSTIC] wstring extension: \"" << wext << "\"" << endl;

			filtered = DeletedFileScanner::FilterByExtension(filtered, wext);
			cout << "[DIAGNOSTIC] Files after extension filter: " << filtered.size() << endl;
		}

		// Filter by name pattern if not wildcard
		if (pattern != "*") {
			wstring wpattern(pattern.begin(), pattern.end());
			filtered = DeletedFileScanner::FilterByName(filtered, wpattern);
			cout << "[DIAGNOSTIC] Files after name filter: " << filtered.size() << endl;
		}

		if (filtered.empty()) {
			cout << "\nNo files matching your search criteria." << endl;
			return;
		}

		cout << "\n===== Search Results =====\n" << endl;
		cout << "Found: " << filtered.size() << " matching files." << endl;
		cout << "\nFormat: [MFT#] Size | Status | Path" << endl;
		cout << "----------------------------------------------" << endl;

		// Display all results (or limit to 100)
		size_t displayLimit = min(filtered.size(), (size_t)100);
		for (size_t i = 0; i < displayLimit; i++) {
			const auto& info = filtered[i];
			cout << "[" << info.recordNumber << "] ";
			cout << info.fileSize << " bytes | ";
			cout << (info.dataAvailable ? "Recoverable" : "Overwritten") << " | ";
			wcout << info.filePath << endl;
		}

		cout << "\n----------------------------------------------" << endl;
		if (filtered.size() > 100) {
			cout << "Note: Showing first 100 of " << filtered.size() << " matching files." << endl;
		}
		cout << "\nTo restore a file, use: restorebyrecord <drive> <MFT#> <output_path>" << endl;
		cout << "Example: restorebyrecord " << driveLetter << " " << filtered[0].recordNumber << " C:\recovered\file.txt" << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
	catch (...) {
		cout << "[ERROR] Unknown exception in SearchDeletedFilesCommand::Execute" << endl;
	}
}

BOOL SearchDeletedFilesCommand::HasArgs()
{
	return FlagHasArgs;
}

BOOL SearchDeletedFilesCommand::CheckName(string input)
{
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}

// ==================== ListDeletedFilesCommand ====================

ListDeletedFilesCommand::ListDeletedFilesCommand()
{
	FlagHasArgs = TRUE;
}

ListDeletedFilesCommand::~ListDeletedFilesCommand()
{
}

void ListDeletedFilesCommand::AcceptArgs(vector<LPVOID> argslist)
{
	ListDeletedFilesCommand::ArgsList = argslist;
}

void ListDeletedFilesCommand::Execute(string command)
{
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.size() < 1 || ArgsList.size() > 2) {
		cout << "Invalid Args! Usage: listdeleted <drive_letter> [filter_level]" << endl;
		cout << "Examples:" << endl;
		cout << "  listdeleted C           - List deleted files with default filter" << endl;
		cout << "  listdeleted C none      - List all deleted files" << endl;
		cout << "  listdeleted C skip      - Skip low-value paths (default)" << endl;
		cout << "  listdeleted C exclude   - Exclude low-value files completely" << endl;
		return;
	}

	try {
		string& driveStr = *(string*)ArgsList[0];

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		// Parse filter level (default: FILTER_SKIP_PATH)
		FilterLevel filterLevel = FILTER_SKIP_PATH;
		if (ArgsList.size() >= 2) {
			string& filterStr = *(string*)ArgsList[1];
			if (filterStr == "none") {
				filterLevel = FILTER_NONE;
			} else if (filterStr == "skip") {
				filterLevel = FILTER_SKIP_PATH;
			} else if (filterStr == "exclude") {
				filterLevel = FILTER_EXCLUDE;
			} else {
				cout << "Unknown filter level: " << filterStr << ". Using default (skip)." << endl;
			}
		}

		cout << "Scanning drive " << driveLetter << ": for deleted files..." << endl;
		cout << "Filter level: ";
		switch (filterLevel) {
			case FILTER_NONE: cout << "None (all paths)"; break;
			case FILTER_SKIP_PATH: cout << "Skip path (default)"; break;
			case FILTER_EXCLUDE: cout << "Exclude low-value files"; break;
		}
		cout << endl;

		FileRestore* fileRestore = new FileRestore();
		fileRestore->SetFilterLevel(filterLevel);
		vector<DeletedFileInfo> deletedFiles = fileRestore->ScanDeletedFiles(driveLetter, 0);
		delete fileRestore;

		if (deletedFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		// Save to cache for future searches
		cout << "Saving to cache for faster future searches..." << endl;
		DeletedFileScanner::SaveToCache(deletedFiles, driveLetter);

		cout << "\n===== Deleted Files on " << driveLetter << ": =====\n" << endl;
		cout << "Found: " << deletedFiles.size() << " deleted files." << endl;
		cout << "\nFormat: [MFT#] Size | Status | Path" << endl;
		cout << "----------------------------------------------" << endl;

		// Display all results (or limit to 100)
		size_t displayLimit = min(deletedFiles.size(), (size_t)100);
		for (size_t i = 0; i < displayLimit; i++) {
			const auto& info = deletedFiles[i];
			cout << "[" << info.recordNumber << "] ";
			cout << info.fileSize << " bytes | ";
			cout << (info.dataAvailable ? "Recoverable" : "Overwritten") << " | ";
			wcout << info.filePath << endl;
		}

		cout << "\n----------------------------------------------" << endl;
		if (deletedFiles.size() > 100) {
			cout << "Note: Showing first 100 of " << deletedFiles.size() << " files." << endl;
		}
		cout << "\nTo search for specific files, use: searchdeleted <drive> <pattern> [extension]" << endl;
		cout << "To restore a file, use: restorebyrecord <drive> <MFT#> <output_path>" << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
	catch (...) {
		cout << "[ERROR] Unknown exception in ListDeletedFilesCommand::Execute" << endl;
	}
}

BOOL ListDeletedFilesCommand::HasArgs()
{
	return FlagHasArgs;
}

BOOL ListDeletedFilesCommand::CheckName(string input)
{
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}

// ==================== RestoreByRecordCommand ====================

RestoreByRecordCommand::RestoreByRecordCommand()
{
	FlagHasArgs = TRUE;
}

RestoreByRecordCommand::~RestoreByRecordCommand()
{
}

void RestoreByRecordCommand::AcceptArgs(vector<LPVOID> argslist)
{
	RestoreByRecordCommand::ArgsList = argslist;
}

void RestoreByRecordCommand::Execute(string command)
{
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.size() != 3) {
		cout << "Invalid Args! Usage: restorebyrecord <drive_letter> <MFT_record_number> <output_path>" << endl;
		cout << "Example: restorebyrecord C 12345 C:\\recovered\\myfile.txt" << endl;
		return;
	}

	try {
		string& driveStr = *(string*)ArgsList[0];
		string& recordStr = *(string*)ArgsList[1];
		string& outputPath = *(string*)ArgsList[2];

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];
		ULONGLONG recordNumber = 0;

		try {
			recordNumber = stoull(recordStr);
		}
		catch (...) {
			cout << "Invalid MFT record number." << endl;
			return;
		}

		cout << "=== File Recovery ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "MFT Record: " << recordNumber << endl;
		cout << "Output Path: " << outputPath << endl;
		cout << endl;

		// First, detect overwrite status
		cout << "Step 1/2: Detecting file overwrite status..." << endl;
		FileRestore* fileRestore = new FileRestore();
		OverwriteDetectionResult result = fileRestore->DetectFileOverwrite(driveLetter, recordNumber);

		if (!result.isFullyAvailable && !result.isPartiallyAvailable) {
			cout << "\n[FAILED] File data has been completely overwritten." << endl;
			cout << "Recovery is not possible." << endl;
			delete fileRestore;
			return;
		}

		if (result.isPartiallyAvailable) {
			cout << "\n[WARNING] File is partially overwritten (" << result.overwritePercentage << "% lost)." << endl;
			cout << "Recovery will attempt to save available data, but the file may be corrupted." << endl;
		} else {
			cout << "\n[OK] File data is fully available." << endl;
		}

		// Attempt recovery
		cout << "\nStep 2/2: Restoring file data..." << endl;
		bool success = fileRestore->RestoreFileByRecordNumber(driveLetter, recordNumber, outputPath);

		if (success) {
			cout << "\n=== Recovery Successful ===" << endl;
			cout << "File has been saved to: " << outputPath << endl;

			if (result.isPartiallyAvailable) {
				cout << "\nNote: The recovered file may be incomplete or corrupted." << endl;
				cout << "Available data: " << (100.0 - result.overwritePercentage) << "%" << endl;
			} else {
				cout << "\nThe file should be fully intact." << endl;
			}
		} else {
			cout << "\n=== Recovery Failed ===" << endl;
			cout << "Unable to restore the file. Possible reasons:" << endl;
			cout << "  - Insufficient permissions" << endl;
			cout << "  - Invalid output path" << endl;
			cout << "  - MFT record not found or corrupted" << endl;
		}

		delete fileRestore;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
	catch (...) {
		cout << "[ERROR] Unknown exception in RestoreByRecordCommand::Execute" << endl;
	}
}

BOOL RestoreByRecordCommand::HasArgs()
{
	return FlagHasArgs;
}

BOOL RestoreByRecordCommand::CheckName(string input)
{
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}

// ==================== ScanUsnCommand ====================

ScanUsnCommand::ScanUsnCommand() {
	FlagHasArgs = TRUE;
}

ScanUsnCommand::~ScanUsnCommand() {
}

void ScanUsnCommand::AcceptArgs(vector<LPVOID> argslist) {
	ScanUsnCommand::ArgsList = argslist;
}

void ScanUsnCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.size() < 1 || ArgsList.size() > 2) {
		cout << "Invalid Args! Usage: scanusn <drive_letter> [max_hours]" << endl;
		cout << "Examples:" << endl;
		cout << "  scanusn C         - Scan C: for files deleted in the last hour" << endl;
		cout << "  scanusn C 24      - Scan C: for files deleted in the last 24 hours" << endl;
		cout << "  scanusn C 168     - Scan C: for files deleted in the last week" << endl;
		return;
	}

	try {
		string& driveStr = *(string*)ArgsList[0];
		int maxHours = 1;  // Default: 1 hour

		if (ArgsList.size() >= 2) {
			string& hoursStr = *(string*)ArgsList[1];
			try {
				maxHours = stoi(hoursStr);
				if (maxHours <= 0) {
					cout << "Invalid hours value. Using default (1 hour)." << endl;
					maxHours = 1;
				}
			} catch (...) {
				cout << "Invalid hours value. Using default (1 hour)." << endl;
			}
		}

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		cout << "\n========== USN Journal Scanner ==========\n" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Time range: Last " << maxHours << " hour(s)" << endl;
		cout << endl;

		// 创建 USN Journal 读取器
		UsnJournalReader usnReader;

		if (!usnReader.Open(driveLetter)) {
			cout << "ERROR: " << usnReader.GetLastError() << endl;
			cout << "\nNote: USN Journal requires:" << endl;
			cout << "  1. Administrator privileges" << endl;
			cout << "  2. USN Journal enabled on the volume" << endl;
			return;
		}

		// 获取并显示 USN Journal 统计信息
		UsnJournalStats stats;
		if (usnReader.GetJournalStats(stats)) {
			cout << "USN Journal Information:" << endl;
			cout << "  Journal ID: " << stats.UsnJournalID << endl;
			cout << "  Maximum Size: " << (stats.MaximumSize / 1024 / 1024) << " MB" << endl;
			cout << "  First USN: " << stats.FirstUsn << endl;
			cout << "  Next USN: " << stats.NextUsn << endl;
			cout << endl;
		}

		// 扫描删除的文件
		int maxTimeSeconds = maxHours * 3600;
		vector<UsnDeletedFileInfo> deletedFiles = usnReader.ScanRecentlyDeletedFiles(
			maxTimeSeconds, 10000);

		if (deletedFiles.empty()) {
			cout << "\nNo deleted files found in the specified time range." << endl;
			return;
		}

		cout << "\n===== Recently Deleted Files (from USN Journal) =====\n" << endl;
		cout << "Found: " << deletedFiles.size() << " deleted files" << endl;
		cout << "\nFormat: [MFT#] Filename | Parent MFT# | Time" << endl;
		cout << "----------------------------------------------" << endl;

		// 显示结果
		size_t displayLimit = min(deletedFiles.size(), (size_t)100);
		for (size_t i = 0; i < displayLimit; i++) {
			const auto& info = deletedFiles[i];

			// 转换时间戳
			SYSTEMTIME st;
			FILETIME ft;
			ft.dwLowDateTime = info.TimeStamp.LowPart;
			ft.dwHighDateTime = info.TimeStamp.HighPart;
			FileTimeToSystemTime(&ft, &st);

			cout << "[" << info.FileReferenceNumber << "] ";
			wcout << info.FileName << " | ";
			cout << "Parent: " << info.ParentFileReferenceNumber << " | ";
			printf("%04d-%02d-%02d %02d:%02d:%02d\n",
				   st.wYear, st.wMonth, st.wDay,
				   st.wHour, st.wMinute, st.wSecond);
		}

		cout << "\n----------------------------------------------" << endl;
		if (deletedFiles.size() > 100) {
			cout << "Note: Showing first 100 of " << deletedFiles.size() << " files." << endl;
		}

		cout << "\nTip: Use 'diagnosefile <drive> <filename>' to check if a file exists in MFT" << endl;

	} catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	} catch (...) {
		cout << "[ERROR] Unknown exception in ScanUsnCommand::Execute" << endl;
	}
}

BOOL ScanUsnCommand::HasArgs() {
	return FlagHasArgs;
}

BOOL ScanUsnCommand::CheckName(string input) {
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}

// ==================== DiagnoseFileCommand ====================

DiagnoseFileCommand::DiagnoseFileCommand() {
	FlagHasArgs = TRUE;
}

DiagnoseFileCommand::~DiagnoseFileCommand() {
}

void DiagnoseFileCommand::AcceptArgs(vector<LPVOID> argslist) {
	DiagnoseFileCommand::ArgsList = argslist;
}

void DiagnoseFileCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.size() != 2) {
		cout << "Invalid Args! Usage: diagnosefile <drive_letter> <filename>" << endl;
		cout << "Examples:" << endl;
		cout << "  diagnosefile C test.txt          - Search for exact filename" << endl;
		cout << "  diagnosefile C test              - Search for files containing 'test'" << endl;
		return;
	}

	try {
		string& driveStr = *(string*)ArgsList[0];
		string& fileNameStr = *(string*)ArgsList[1];

		if (driveStr.empty() || fileNameStr.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter = driveStr[0];
		wstring searchName(fileNameStr.begin(), fileNameStr.end());

		cout << "\n========== File Diagnostic Tool ==========\n" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		wcout << L"Searching for: " << searchName << endl;
		cout << endl;

		// 创建 MFT 读取器和解析器
		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "ERROR: Failed to open volume " << driveLetter << ":" << endl;
			cout << "Administrator privileges are required." << endl;
			return;
		}

		MFTParser parser(&reader);
		PathResolver pathResolver(&reader, &parser);

		ULONGLONG totalRecords = reader.GetTotalMFTRecords();
		cout << "Total MFT records: " << totalRecords << endl;

		vector<BYTE> record;
		ULONGLONG foundCount = 0;
		ULONGLONG scannedCount = 0;
		ULONGLONG activeFiles = 0;
		ULONGLONG deletedFiles = 0;
		bool usedCache = false;

		// 转换为小写进行不区分大小写的搜索
		wstring searchNameLower = searchName;
		transform(searchNameLower.begin(), searchNameLower.end(),
				  searchNameLower.begin(), ::towlower);

		// ========== 优化：优先尝试从缓存搜索 ==========
		cout << "Attempting to load path cache..." << endl;
		if (pathResolver.LoadCache(driveLetter)) {
			cout << "Path cache loaded successfully (" << pathResolver.GetCacheSize() << " entries)" << endl;
			cout << "Searching in cache (much faster)..." << endl;
			cout << endl;

			usedCache = true;
			auto& cache = pathResolver.GetCacheRef();

			// 在缓存中搜索匹配的路径
			for (const auto& entry : cache) {
				ULONGLONG recordNum = entry.first;
				const wstring& fullPath = entry.second;

				// 提取文件名（路径的最后一部分）
				size_t lastSlash = fullPath.find_last_of(L"\\/");
				wstring fileName = (lastSlash != wstring::npos) ?
								  fullPath.substr(lastSlash + 1) : fullPath;

				// 转换为小写进行比较
				wstring fileNameLower = fileName;
				transform(fileNameLower.begin(), fileNameLower.end(),
						  fileNameLower.begin(), ::towlower);

				// 检查是否匹配
				if (fileNameLower.find(searchNameLower) != wstring::npos) {
					// 读取 MFT 记录以获取详细信息
					if (!reader.ReadMFT(recordNum, record)) {
						continue;
					}

					FILE_RECORD_HEADER* header = (FILE_RECORD_HEADER*)record.data();
					bool isDeleted = ((header->Flags & 0x01) == 0);
					bool isDirectory = ((header->Flags & 0x02) != 0);

					if (isDeleted) {
						deletedFiles++;
					} else {
						activeFiles++;
					}

					foundCount++;

					// 显示找到的文件
					cout << "\n[" << foundCount << "] MFT Record #" << recordNum << endl;
					wcout << L"  Name: " << fileName << endl;
					cout << "  Status: " << (isDeleted ? "DELETED" : "ACTIVE") << endl;
					cout << "  Type: " << (isDirectory ? "Directory" : "File") << endl;
					wcout << L"  Full Path: " << fullPath << endl;

					if (foundCount >= 50) {
						cout << "\n(Limiting results to first 50 matches)" << endl;
						break;
					}
				}

				scannedCount++;
			}

			cout << "\nCache search completed." << endl;
		} else {
			// ========== 回退：缓存不可用，扫描所有 MFT 记录 ==========
			cout << "Path cache not available, performing full MFT scan..." << endl;
			cout << "Note: This may take several minutes. Use 'listdeleted' first to build cache." << endl;
			cout << "Scanning..." << endl;
			cout << endl;

			for (ULONGLONG i = 16; i < totalRecords; i++) {
				if (!reader.ReadMFT(i, record)) {
					continue;
				}

				scannedCount++;

				// 解析文件名
				ULONGLONG parentDir;
				wstring fileName = parser.GetFileNameFromRecord(record, parentDir);

				if (fileName.empty()) {
					continue;
				}

				// 转换为小写进行比较
				wstring fileNameLower = fileName;
				transform(fileNameLower.begin(), fileNameLower.end(),
						  fileNameLower.begin(), ::towlower);

				// 检查是否匹配
				if (fileNameLower.find(searchNameLower) != wstring::npos) {
					FILE_RECORD_HEADER* header = (FILE_RECORD_HEADER*)record.data();
					bool isDeleted = ((header->Flags & 0x01) == 0);
					bool isDirectory = ((header->Flags & 0x02) != 0);

					if (isDeleted) {
						deletedFiles++;
					} else {
						activeFiles++;
					}

					foundCount++;

					// 显示找到的文件
					cout << "\n[" << foundCount << "] MFT Record #" << i << endl;
					wcout << L"  Name: " << fileName << endl;
					cout << "  Status: " << (isDeleted ? "DELETED" : "ACTIVE") << endl;
					cout << "  Type: " << (isDirectory ? "Directory" : "File") << endl;
					cout << "  Parent MFT#: " << parentDir << endl;

					// 尝试重建路径
					try {
						wstring fullPath = pathResolver.ReconstructPath(i);
						if (!fullPath.empty()) {
							wcout << L"  Full Path: " << fullPath << endl;
						}
					} catch (...) {
						cout << "  Full Path: (unable to reconstruct)" << endl;
					}

					if (foundCount >= 50) {
						cout << "\n(Limiting results to first 50 matches)" << endl;
						break;
					}
				}

				// 显示进度
				if (scannedCount % 100000 == 0) {
					cout << "\r  Progress: " << scannedCount << " / " << totalRecords
						 << " (" << (scannedCount * 100 / totalRecords) << "%)" << flush;
				}
			}

			cout << "\r                                                                " << flush;
			cout << "\r";
		}

		// 显示统计信息
		cout << "\n========== Scan Results ==========\n" << endl;
		cout << "Search method: " << (usedCache ? "Cache (Fast)" : "Full MFT Scan (Slow)") << endl;
		cout << "Total records searched: " << scannedCount << endl;
		cout << "Total matches found: " << foundCount << endl;
		cout << "  - Active files: " << activeFiles << endl;
		cout << "  - Deleted files: " << deletedFiles << endl;

		if (foundCount == 0) {
			cout << "\nNo files matching '" << fileNameStr << "' were found." << endl;
			cout << "\nPossible reasons:" << endl;
			cout << "  1. File was never created on this volume" << endl;
			cout << "  2. MFT record was reused (old data overwritten)" << endl;
			if (!usedCache) {
				cout << "  3. Run 'listdeleted " << driveLetter << " <pattern>' first to build cache for faster searches" << endl;
			}
			cout << "  " << (usedCache ? "3" : "4") << ". Try using USN Journal: scanusn " << driveLetter << endl;
		} else {
			cout << "\nNote: If your target file is not in the list above:" << endl;
			cout << "  - It may have been created with a different name" << endl;
			if (!usedCache) {
				cout << "  - Run 'listdeleted " << driveLetter << " <pattern>' first to build cache for faster searches" << endl;
			}
			cout << "  - Try USN Journal for recently deleted files: scanusn " << driveLetter << endl;
		}

	} catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	} catch (...) {
		cout << "[ERROR] Unknown exception in DiagnoseFileCommand::Execute" << endl;
	}
}

BOOL DiagnoseFileCommand::HasArgs() {
	return FlagHasArgs;
}

BOOL DiagnoseFileCommand::CheckName(string input) {
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}
