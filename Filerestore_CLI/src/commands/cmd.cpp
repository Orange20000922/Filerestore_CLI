#include "cmd.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <Windows.h>
#include <queue>
#include "cli.h"
#include "ImageTable.h"
#include "FileRestore.h"
#include "DeletedFileScanner.h"
#include "MFTReader.h"
#include "MFTBatchReader.h"
#include "OverwriteDetector.h"
#include "UsnJournalReader.h"
#include "MFTParser.h"
#include "PathResolver.h"
#include "ProgressBar.h"
#include "LocalizationManager.h"
#include "FileCarver.h"
#include "SignatureScanThreadPool.h"
#include "FileIntegrityValidator.h"
#include "CarvedResultsCache.h"
#include "MemoryMappedResults.h"
#include <algorithm>
#include <cmath>
#include <map>
using namespace std;

// USN_REASON 标志定义（用于显示删除类型）
#ifndef USN_REASON_FILE_DELETE
#define USN_REASON_FILE_DELETE 0x00000200
#endif
#ifndef USN_REASON_RENAME_OLD_NAME
#define USN_REASON_RENAME_OLD_NAME 0x00001000
#endif

// ============================================================================
// PrintAllFunction - 特殊命令，包含额外成员变量，需要手动定义
// ============================================================================
string PrintAllFunction::name = "printallfunc |file";
vector<LPVOID> PrintAllFunction::Arglist = vector<LPVOID>();
REGISTER_COMMAND(PrintAllFunction);

PrintAllFunction::PrintAllFunction() {
	FlagHasArgs = true;
}

PrintAllFunction::~PrintAllFunction() {
	delete analyzer;
}

void PrintAllFunction::AcceptArgs(vector<LPVOID> argslist) {
	PrintAllFunction::Arglist = argslist;
}

BOOL PrintAllFunction::HasArgs() {
	return FlagHasArgs;
}

BOOL PrintAllFunction::CheckName(string input) {
	return input.compare(name) == 0;
}

void PrintAllFunction::Execute(string command) {
	if (Arglist.size() == 1) {
		string file = *(string*)Arglist[0];
		map<string, vector<string>> funclist = analyzer->AnalyzeTableForFunctions(file);
		vector<string> dllNames = analyzer->AnalyzeTableForDLL(file);
		for (auto dllName : dllNames) {
			vector<string> value = funclist[dllName];
			cout << dllName + ":" << endl;
			for (auto funcname : value) {
				ULONGLONG funcaddr = analyzer->GetFuncaddressByName(funcname, file);
				cout << "    " + funcname + "  " + "FunctionAddress:" + "0x" << hex << funcaddr << endl;
			}
		}
	}
}

// ============================================================================
// PrintAllCommand - 打印所有命令 (无参数)
// ============================================================================
DEFINE_COMMAND_BASE_NOARGS(PrintAllCommand, "printallcommand -list")
REGISTER_COMMAND(PrintAllCommand);

void PrintAllCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}
	const vector<vector<string>>& allCommands = CLI::GetCommands();
	for (const auto& cmdVec : allCommands) {
		string output = "";
		for (size_t i = 0; i < cmdVec.size(); i++) {
			output += cmdVec[i];
			if (i < cmdVec.size() - 1) {
				output += " ";
			}
		}
		cout << output << endl;
	}
}

// ============================================================================
// HelpCommand - 帮助命令
// ============================================================================
DEFINE_COMMAND_BASE(HelpCommand, "help |name", TRUE)
REGISTER_COMMAND(HelpCommand);

void HelpCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	// 如果没有参数，显示所有命令的概览
	if (ArgsList.empty()) {
		cout << "\n";
		cout << "=============================================================================\n";
		cout << "                    Filerestore_CLI - 文件恢复工具                          \n";
		cout << "=============================================================================\n";
		cout << "\n用法: <command> [arguments...]" << endl;
		cout << "      help <command>  - 查看特定命令的详细帮助\n" << endl;

		cout << "=== 文件恢复命令 ===" << endl;
		cout << "  restorebyrecord   - 通过MFT记录号恢复单个文件" << endl;
		cout << "  forcerestore      - 强制恢复文件（跳过覆盖检测）" << endl;
		cout << "  batchrestore      - 批量恢复多个已删除文件" << endl;
		cout << endl;

		cout << "=== 文件搜索命令 ===" << endl;
		cout << "  listdeleted       - 列出指定驱动器的已删除文件" << endl;
		cout << "  searchdeleted     - 按文件名/扩展名搜索已删除文件" << endl;
		cout << "  searchusn         - 使用USN日志搜索最近删除的文件" << endl;
		cout << "  filtersize        - 按文件大小范围筛选已删除文件" << endl;
		cout << "  finduserfiles     - 查找用户文件(文档/图片/视频等)" << endl;
		cout << "  findrecord        - 通过文件路径查找MFT记录号" << endl;
		cout << "  diagnosefile      - 诊断并搜索特定文件名" << endl;
		cout << endl;

		cout << "=== 系统诊断命令 ===" << endl;
		cout << "  detectoverwrite   - 检测文件数据是否被覆盖" << endl;
		cout << "  diagnosemft       - 诊断MFT碎片化状态" << endl;
		cout << "  scanusn           - 扫描USN日志中的文件操作" << endl;
		cout << endl;

		cout << "=== 分析工具命令 ===" << endl;
		cout << "  printallfunc      - 打印PE文件的所有导入函数" << endl;
		cout << "  queuedllsname     - 列出PE文件依赖的所有DLL" << endl;
		cout << "  getfuncaddr       - 获取函数在进程中的地址" << endl;
		cout << endl;

		cout << "=== 签名搜索命令 (File Carving) ===" << endl;
		cout << "  carve             - 扫描磁盘搜索特定类型文件" << endl;
		cout << "  carvepool         - 线程池并行扫描（多核优化）" << endl;
		cout << "  carvetypes        - 列出支持的文件类型" << endl;
		cout << "  carvelist         - 列出扫描结果（支持筛选）" << endl;
		cout << "  carverecover      - 恢复指定的carved文件" << endl;
		cout << "  carvevalidate     - 批量验证文件完整性" << endl;
		cout << "  carveintegrity    - 详细分析单个文件完整性" << endl;
		cout << endl;

		cout << "=== 系统命令 ===" << endl;
		cout << "  printallcommand   - 列出所有可用命令" << endl;
		cout << "  help              - 显示帮助信息" << endl;
		cout << "  setlang           - 切换界面语言" << endl;
		cout << "  exit              - 退出程序" << endl;
		cout << endl;

		cout << "提示: 大多数命令需要管理员权限才能访问磁盘MFT。" << endl;
		cout << "      使用 'help <命令名>' 查看详细用法示例。" << endl;
		cout << "\n=============================================================================\n";
		return;
	}

	// 如果有参数，显示特定命令的详细帮助
	string& cmdName = GET_ARG_STRING(0);

	// 文件恢复命令帮助
	if (cmdName == "restorebyrecord") {
		cout << "\n=== restorebyrecord - 恢复已删除文件 ===\n" << endl;
		cout << "用法: restorebyrecord <drive> <record_number> <output_path>\n" << endl;
		cout << "参数:" << endl;
		cout << "  <drive>          - 驱动器字母 (如: C, D)" << endl;
		cout << "  <record_number>  - MFT记录号" << endl;
		cout << "  <output_path>    - 输出文件完整路径\n" << endl;
		cout << "示例:" << endl;
		cout << "  restorebyrecord C 12345 D:\\recovered\\document.docx" << endl;
		cout << "  restorebyrecord D 99999 C:\\backup\\photo.jpg\n" << endl;
		cout << "说明:" << endl;
		cout << "  1. 自动检测文件是否被覆盖" << endl;
		cout << "  2. 如果文件部分覆盖，会尝试恢复可用数据" << endl;
		cout << "  3. 完全覆盖的文件无法恢复\n" << endl;
	}
	else if (cmdName == "batchrestore") {
		cout << "\n=== batchrestore - 批量恢复文件 ===\n" << endl;
		cout << "用法: batchrestore <drive> <record_numbers> <output_directory>\n" << endl;
		cout << "参数:" << endl;
		cout << "  <drive>            - 驱动器字母" << endl;
		cout << "  <record_numbers>   - 逗号分隔的MFT记录号列表" << endl;
		cout << "  <output_directory> - 输出目录路径\n" << endl;
		cout << "示例:" << endl;
		cout << "  batchrestore C 12345,12346,12347 D:\\recovered\\" << endl;
		cout << "  batchrestore D 1000,2000,3000,4000 C:\\backup\\\n" << endl;
	}
	else if (cmdName == "forcerestore") {
		cout << "\n=== forcerestore - 强制恢复文件 ===\n" << endl;
		cout << "用法: forcerestore <drive> <record_number> <output_path>\n" << endl;
		cout << "参数:" << endl;
		cout << "  <drive>          - 驱动器字母 (如: C, D)" << endl;
		cout << "  <record_number>  - MFT记录号" << endl;
		cout << "  <output_path>    - 输出文件完整路径\n" << endl;
		cout << "示例:" << endl;
		cout << "  forcerestore C 12345 D:\\recovered\\document.docx" << endl;
		cout << "  forcerestore D 99999 C:\\backup\\archive.zip\n" << endl;
		cout << "说明:" << endl;
		cout << "  此命令跳过覆盖检测，直接尝试恢复文件。" << endl;
		cout << "  适用于以下情况:" << endl;
		cout << "  1. SSD TRIM导致的读取全零（数据可能仍存在）" << endl;
		cout << "  2. 高熵文件（压缩包、加密文件）被误判为覆盖" << endl;
		cout << "  3. detectoverwrite检测结果可能为误报" << endl;
		cout << "  4. 愿意尝试恢复即使检测显示已覆盖\n" << endl;
		cout << "注意:" << endl;
		cout << "  - 恢复后请验证文件完整性" << endl;
		cout << "  - 可能恢复出损坏或部分数据" << endl;
		cout << "  - 如果簇被重用，可能恢复出随机数据\n" << endl;
	}
	else if (cmdName == "listdeleted") {
		cout << "\n=== listdeleted - 列出已删除文件 ===\n" << endl;
		cout << "用法: listdeleted <drive> <max_files>\n" << endl;
		cout << "参数:" << endl;
		cout << "  <drive>      - 驱动器字母" << endl;
		cout << "  <max_files>  - 最大显示数量 (0=不限制)\n" << endl;
		cout << "示例:" << endl;
		cout << "  listdeleted C 100   - 显示C盘前100个已删除文件" << endl;
		cout << "  listdeleted D 0     - 显示D盘所有已删除文件\n" << endl;
	}
	else if (cmdName == "searchdeleted") {
		cout << "\n=== searchdeleted - 搜索已删除文件 ===\n" << endl;
		cout << "用法: searchdeleted <drive> <pattern> [extension] [filter_level]\n" << endl;
		cout << "示例:" << endl;
		cout << "  searchdeleted C document" << endl;
		cout << "  searchdeleted C report .pdf" << endl;
		cout << "  searchdeleted D * .jpg skip\n" << endl;
	}
	else if (cmdName == "searchusn") {
		cout << "\n=== searchusn - USN日志搜索 ===\n" << endl;
		cout << "用法: searchusn <drive> <filename> [exact]\n" << endl;
		cout << "示例:" << endl;
		cout << "  searchusn C document.docx" << endl;
		cout << "  searchusn D report.pdf exact\n" << endl;
	}
	else if (cmdName == "filtersize") {
		cout << "\n=== filtersize - 按大小筛选文件 ===\n" << endl;
		cout << "用法: filtersize <drive> <min_size> <max_size> [limit]\n" << endl;
		cout << "示例:" << endl;
		cout << "  filtersize C 1M 100M       - 1MB到100MB之间的文件" << endl;
		cout << "  filtersize D 0B 1K         - 小于1KB的文件\n" << endl;
	}
	else if (cmdName == "finduserfiles") {
		cout << "\n=== finduserfiles - 查找用户文件 ===\n" << endl;
		cout << "用法: finduserfiles <drive> [limit]\n" << endl;
		cout << "示例:" << endl;
		cout << "  finduserfiles C" << endl;
		cout << "  finduserfiles D 200\n" << endl;
	}
	else if (cmdName == "findrecord") {
		cout << "\n=== findrecord - 查找MFT记录号 ===\n" << endl;
		cout << "用法: findrecord <drive> <file_path>\n" << endl;
		cout << "示例:" << endl;
		cout << "  findrecord C C:\\Windows\\System32\\notepad.exe\n" << endl;
	}
	else if (cmdName == "diagnosefile") {
		cout << "\n=== diagnosefile - 文件诊断 ===\n" << endl;
		cout << "用法: diagnosefile <drive> <filename>\n" << endl;
		cout << "示例:" << endl;
		cout << "  diagnosefile C document.docx\n" << endl;
	}
	else if (cmdName == "detectoverwrite") {
		cout << "\n=== detectoverwrite - 覆盖检测 ===\n" << endl;
		cout << "用法: detectoverwrite <drive> <record_number> [mode]\n" << endl;
		cout << "检测模式: fast, balanced, thorough\n" << endl;
	}
	else if (cmdName == "diagnosemft") {
		cout << "\n=== diagnosemft - MFT诊断 ===\n" << endl;
		cout << "用法: diagnosemft <drive>\n" << endl;
		cout << "示例:" << endl;
		cout << "  diagnosemft C\n" << endl;
	}
	else if (cmdName == "scanusn") {
		cout << "\n=== scanusn - USN日志扫描 ===\n" << endl;
		cout << "用法: scanusn <drive> <max_hours>\n" << endl;
		cout << "示例:" << endl;
		cout << "  scanusn C 24   - 扫描最近24小时\n" << endl;
	}
	else if (cmdName == "printallfunc") {
		cout << "\n=== printallfunc - 打印PE函数 ===\n" << endl;
		cout << "用法: printallfunc <file_path>\n" << endl;
	}
	else if (cmdName == "queuedllsname") {
		cout << "\n=== queuedllsname - 列出依赖DLL ===\n" << endl;
		cout << "用法: queuedllsname <file_path>\n" << endl;
	}
	else if (cmdName == "getfuncaddr") {
		cout << "\n=== getfuncaddr - 获取函数地址 ===\n" << endl;
		cout << "用法: getfuncaddr <file_path> <function_name>\n" << endl;
	}
	else if (cmdName == "exit") {
		cout << "\n=== exit - 退出程序 ===\n" << endl;
		cout << "用法: exit\n" << endl;
	}
	else if (cmdName == "setlang") {
		cout << "\n=== setlang - 语言设置 ===\n" << endl;
		cout << "用法: setlang [language_code]\n" << endl;
		cout << "支持: en (English), zh (中文)\n" << endl;
	}
	else if (cmdName == "help") {
		cout << "\n=== help - 帮助系统 ===\n" << endl;
		cout << "用法: help [command]\n" << endl;
	}
	else if (cmdName == "carve") {
		cout << "\n=== carve - 签名搜索（文件雕刻）===\n" << endl;
		cout << "用法: carve <drive> <type|types|all> <output_dir> [async|sync]\n" << endl;
		cout << "参数:" << endl;
		cout << "  <drive>      - 驱动器字母 (如: C, D)" << endl;
		cout << "  <type>       - 文件类型或逗号分隔的多类型" << endl;
		cout << "  <output_dir> - 恢复文件的输出目录" << endl;
		cout << "  [async|sync] - 扫描模式 (默认: async)\n" << endl;
		cout << "示例:" << endl;
		cout << "  carve C zip D:\\recovered\\" << endl;
		cout << "  carve C jpg,png,gif D:\\recovered\\" << endl;
		cout << "  carve D all D:\\recovered\\ async" << endl;
		cout << "  carve D all D:\\recovered\\ sync\n" << endl;
		cout << "扫描模式:" << endl;
		cout << "  async - 双缓冲异步I/O（默认，更快）" << endl;
		cout << "        - I/O读取和CPU扫描并行执行" << endl;
		cout << "        - 典型提升: 30-50%" << endl;
		cout << "  sync  - 同步I/O（简单模式）" << endl;
		cout << "        - 读取完成后再扫描\n" << endl;
		cout << "性能特性:" << endl;
		cout << "  - 64MB x 2 双缓冲区（异步模式）" << endl;
		cout << "  - 单次扫描多签名匹配" << endl;
		cout << "  - 首字节索引，O(1) 签名查找" << endl;
		cout << "  - 智能跳过空白区域" << endl;
		cout << "  - 实时显示扫描速度和并行效率\n" << endl;
	}
	else if (cmdName == "carvetypes") {
		cout << "\n=== carvetypes - 列出支持的文件类型 ===\n" << endl;
		cout << "用法: carvetypes\n" << endl;
		cout << "显示carve命令支持的所有文件类型及其签名信息。\n" << endl;
	}
	else if (cmdName == "carverecover") {
		cout << "\n=== carverecover - 恢复carved文件 ===\n" << endl;
		cout << "用法: carverecover <index> <output_path>\n" << endl;
		cout << "参数:" << endl;
		cout << "  <index>       - 扫描结果中的文件索引" << endl;
		cout << "  <output_path> - 输出文件路径\n" << endl;
		cout << "示例:" << endl;
		cout << "  carverecover 0 D:\\recovered\\file.zip\n" << endl;
		cout << "说明:" << endl;
		cout << "  必须先运行carve命令扫描，然后使用此命令恢复。\n" << endl;
	}
	else if (cmdName == "carvepool") {
		cout << "\n=== carvepool - 线程池并行签名搜索 ===\n" << endl;
		cout << "用法: carvepool <drive> <type|types|all> <output_dir> [threads]\n" << endl;
		cout << "参数:" << endl;
		cout << "  <drive>      - 驱动器字母 (如: C, D)" << endl;
		cout << "  <type>       - 文件类型或逗号分隔的多类型" << endl;
		cout << "  <output_dir> - 恢复文件的输出目录" << endl;
		cout << "  [threads]    - 工作线程数（默认自动检测）\n" << endl;
		cout << "示例:" << endl;
		cout << "  carvepool C zip D:\\recovered\\" << endl;
		cout << "  carvepool C jpg,png,gif D:\\recovered\\" << endl;
		cout << "  carvepool D all D:\\recovered\\ 8\n" << endl;
		cout << "特性:" << endl;
		cout << "  - 多线程并行扫描，充分利用多核CPU" << endl;
		cout << "  - 自动检测CPU核心数并优化线程配置" << endl;
		cout << "  - 128MB读取缓冲区，优化NVMe SSD性能" << endl;
		cout << "  - 8MB任务块，平衡负载分配\n" << endl;
		cout << "性能对比:" << endl;
		cout << "  - 相比carve sync: 提升3-6倍" << endl;
		cout << "  - 相比carve async: 提升2-3倍" << endl;
		cout << "  - 16核CPU + NVMe: 可达2500+ MB/s\n" << endl;
	}
	else if (cmdName == "carvevalidate") {
		cout << "\n=== carvevalidate - 批量验证文件完整性 ===\n" << endl;
		cout << "用法: carvevalidate [filter] [min_score]\n" << endl;
		cout << "参数:" << endl;
		cout << "  [filter]     - 添加此参数以过滤低完整性文件" << endl;
		cout << "  [min_score]  - 最小完整性分数阈值 (0.0-1.0, 默认0.5)\n" << endl;
		cout << "示例:" << endl;
		cout << "  carvevalidate              - 验证所有文件" << endl;
		cout << "  carvevalidate filter       - 验证并过滤掉损坏文件" << endl;
		cout << "  carvevalidate filter 0.7   - 过滤分数低于70%的文件\n" << endl;
		cout << "说明:" << endl;
		cout << "  必须先运行carve或carvepool扫描，然后使用此命令验证。" << endl;
		cout << "  验证方法包括：熵值分析、结构验证、统计分析、文件尾检测\n" << endl;
		cout << "完整性分数含义:" << endl;
		cout << "  >= 80%: 高置信度，文件可能完好" << endl;
		cout << "  50-80%: 中置信度，可能有小问题" << endl;
		cout << "  < 50%:  低置信度，很可能已损坏\n" << endl;
	}
	else if (cmdName == "carveintegrity") {
		cout << "\n=== carveintegrity - 详细分析单个文件完整性 ===\n" << endl;
		cout << "用法: carveintegrity <index>\n" << endl;
		cout << "参数:" << endl;
		cout << "  <index>  - 扫描结果中的文件索引\n" << endl;
		cout << "示例:" << endl;
		cout << "  carveintegrity 0   - 分析第一个文件\n" << endl;
		cout << "说明:" << endl;
		cout << "  显示详细的完整性分析报告，包括：" << endl;
		cout << "  - 熵值分析：原始熵值、熵值评分、异常检测" << endl;
		cout << "  - 结构验证：头部/尾部验证、格式特定检查" << endl;
		cout << "  - 统计分析：零字节比例、卡方检验" << endl;
		cout << "  - 综合评估：总体分数和诊断信息\n" << endl;
	}
	else if (cmdName == "carvelist") {
		cout << "\n=== carvelist - 列出扫描结果 ===\n" << endl;
		cout << "用法: carvelist [filter] [page]\n" << endl;
		cout << "参数:" << endl;
		cout << "  [filter]  - 筛选条件：deleted/del/d (已删除) 或 active/act/a (活动文件)" << endl;
		cout << "  [page]    - 页码 (从0开始)\n" << endl;
		cout << "示例:" << endl;
		cout << "  carvelist              - 显示第一页所有文件" << endl;
		cout << "  carvelist 2            - 显示第3页" << endl;
		cout << "  carvelist deleted      - 只显示已删除的文件" << endl;
		cout << "  carvelist deleted 1    - 已删除文件的第2页" << endl;
		cout << "  carvelist active       - 只显示活动（未删除）的文件" << endl;
		cout << "  carvelist 0 100        - 显示索引0-99的文件\n" << endl;
		cout << "说明:" << endl;
		cout << "  文件状态标记：" << endl;
		cout << "  [DELETED] - 已删除的文件，可以安全恢复" << endl;
		cout << "  [ACTIVE]  - 活动文件，仍存在于文件系统中，无需恢复\n" << endl;
		cout << "  自动恢复只会处理已删除的文件，活动文件会被跳过。" << endl;
	}
	else {
		cout << "\n未知命令: " << cmdName << endl;
		cout << "使用 'help' 查看所有可用命令。\n" << endl;
	}
}

// ============================================================================
// ExitCommand - 退出命令 (无参数)
// ============================================================================
DEFINE_COMMAND_BASE_NOARGS(ExitCommand, "exit")
REGISTER_COMMAND(ExitCommand);

void ExitCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}
	cout << "Exiting the application." << endl;
	CLI::SetShouldExit(true);
}

// ============================================================================
// SetLanguageCommand - 设置语言命令
// ============================================================================
DEFINE_COMMAND_BASE(SetLanguageCommand, "setlang |name", TRUE)
REGISTER_COMMAND(SetLanguageCommand);

void SetLanguageCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.empty()) {
		cout << "\n=== Language Settings / 语言设置 ===\n" << endl;
		cout << "Current language / 当前语言: ";
		wcout << LocalizationManager::Instance().GetCurrentLanguage() << endl;
		cout << endl;

		cout << "Available languages / 可用语言:" << endl;
		vector<wstring> languages;
		LocalizationManager::Instance().GetSupportedLanguages(languages);
		for (const auto& lang : languages) {
			wcout << L"  - " << lang;
			if (lang == L"en") wcout << L" (English)";
			else if (lang == L"zh") wcout << L" (中文)";
			wcout << endl;
		}
		cout << endl;

		cout << "Usage: setlang <language_code>" << endl;
		return;
	}

	try {
		string& langStr = GET_ARG_STRING(0);
		wstring langCode(langStr.begin(), langStr.end());

		cout << "\nSwitching language to: " << langStr << "..." << endl;

		if (!LocalizationManager::Instance().SetLanguage(langCode)) {
			cout << "\n[ERROR] Failed to load language: " << langStr << endl;
			return;
		}

		cout << "\n=== Language Changed Successfully ===" << endl;
		wcout << L"Current language: " << LocalizationManager::Instance().GetCurrentLanguage() << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// QueueDLLsCommand - 列出 DLL 依赖
// ============================================================================
DEFINE_COMMAND_BASE(QueueDLLsCommand, "queuedllsname |file", TRUE)
REGISTER_COMMAND(QueueDLLsCommand);

void QueueDLLsCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}
	ImageTableAnalyzer* analyzer = new ImageTableAnalyzer();
	if (GET_ARG_COUNT() != 1) {
		cout << "Invalid Args!" << endl;
	}
	else {
		string& pefile = GET_ARG_STRING(0);
		vector<string> dlllist = analyzer->AnalyzeTableForDLL(pefile);
		if (dlllist.size() != 0) {
			for (int i = 0; i < dlllist.size(); i++) {
				cout << "    " + dlllist[i] << endl;
			}
		}
		else {
			cout << "can't find the IAT" << endl;
		}
	}
	delete analyzer;
}

// ============================================================================
// GetProcessFuncAddressCommand - 获取函数地址
// ============================================================================
DEFINE_COMMAND_BASE(GetProcessFuncAddressCommand, "getfuncaddr |file |name", TRUE)
REGISTER_COMMAND(GetProcessFuncAddressCommand);

void GetProcessFuncAddressCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}
	ImageTableAnalyzer* analyzer = new ImageTableAnalyzer();
	if (GET_ARG_COUNT() != 2) {
		cout << "Invalid Args!" << endl;
	}
	else {
		string& funcname = GET_ARG_STRING(1);
		string& pefile = GET_ARG_STRING(0);
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

// ============================================================================
// DiagnoseMFTCommand - 诊断 MFT 碎片化
// ============================================================================
DEFINE_COMMAND_BASE(DiagnoseMFTCommand, "diagnosemft |name", TRUE)
REGISTER_COMMAND(DiagnoseMFTCommand);

void DiagnoseMFTCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() != 1) {
		cout << "Usage: diagnosemft <drive_letter>" << endl;
		cout << "Example: diagnosemft C" << endl;
		return;
	}

	string& driveStr = GET_ARG_STRING(0);
	if (driveStr.empty()) {
		cout << "Invalid drive letter." << endl;
		return;
	}

	char driveLetter = driveStr[0];

	MFTReader reader;
	if (!reader.OpenVolume(driveLetter)) {
		cout << "Failed to open volume " << driveLetter << ":/" << endl;
		return;
	}

	reader.DiagnoseMFTFragmentation();
}

// ============================================================================
// DetectOverwriteCommand - 检测文件覆盖
// ============================================================================
DEFINE_COMMAND_BASE(DetectOverwriteCommand, "detectoverwrite |name |name |name", TRUE)
REGISTER_COMMAND(DetectOverwriteCommand);

void DetectOverwriteCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 2) {
		cout << "Invalid Args! Usage: detectoverwrite <drive_letter> <MFT_record_number> [mode]" << endl;
		cout << "Modes: fast, balanced, thorough" << endl;
		return;
	}

	string& driveStr = GET_ARG_STRING(0);
	string& recordStr = GET_ARG_STRING(1);

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

	DetectionMode mode = MODE_BALANCED;
	if (GET_ARG_COUNT() >= 3) {
		string& modeStr = GET_ARG_STRING(2);
		if (modeStr == "fast") mode = MODE_FAST;
		else if (modeStr == "thorough") mode = MODE_THOROUGH;
	}

	cout << "=== Overwrite Detection ===" << endl;
	cout << "Drive: " << driveLetter << ":" << endl;
	cout << "MFT Record: " << recordNumber << endl;

	FileRestore* fileRestore = new FileRestore();
	OverwriteDetector* detector = fileRestore->GetOverwriteDetector();
	detector->SetDetectionMode(mode);

	OverwriteDetectionResult result = fileRestore->DetectFileOverwrite(driveLetter, recordNumber);

	cout << "\n=== Detection Summary ===" << endl;
	cout << "Overwrite Percentage: " << result.overwritePercentage << "%" << endl;

	if (result.isFullyAvailable) {
		cout << "Status: [EXCELLENT] Fully Recoverable" << endl;
	}
	else if (result.isPartiallyAvailable) {
		cout << "Status: [WARNING] Partially Recoverable" << endl;
	}
	else {
		cout << "Status: [FAILED] Not Recoverable" << endl;
	}

	delete fileRestore;
}

// ============================================================================
// SearchDeletedFilesCommand - 搜索已删除文件
// ============================================================================
DEFINE_COMMAND_BASE(SearchDeletedFilesCommand, "searchdeleted |name |name |name |name", TRUE)
REGISTER_COMMAND(SearchDeletedFilesCommand);

void SearchDeletedFilesCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 2) {
		cout << "Invalid Args! Usage: searchdeleted <drive_letter> <filename_pattern> [extension] [filter_level]" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& pattern = GET_ARG_STRING(1);
		string extension = HAS_ARG(2) ? GET_ARG_STRING(2) : "";

		FilterLevel filterLevel = FILTER_SKIP_PATH;
		if (HAS_ARG(3)) {
			string& filterStr = GET_ARG_STRING(3);
			if (filterStr == "none") filterLevel = FILTER_NONE;
			else if (filterStr == "exclude") filterLevel = FILTER_EXCLUDE;
		}

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		cout << "Searching drive " << driveLetter << ": for deleted files..." << endl;
		cout << "Pattern: " << pattern << endl;

		vector<DeletedFileInfo> allFiles;

		if (DeletedFileScanner::IsCacheValid(driveLetter, 60)) {
			cout << "Loading from cache..." << endl;
			DeletedFileScanner::LoadFromCache(allFiles, driveLetter);
		}

		if (allFiles.empty()) {
			cout << "Scanning MFT..." << endl;
			FileRestore* fileRestore = new FileRestore();
			fileRestore->SetFilterLevel(filterLevel);
			allFiles = fileRestore->ScanDeletedFiles(driveLetter, 0);
			delete fileRestore;

			if (!allFiles.empty()) {
				DeletedFileScanner::SaveToCache(allFiles, driveLetter);
			}
		}

		if (allFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		vector<DeletedFileInfo> filtered = allFiles;

		if (!extension.empty() && extension != "*") {
			wstring wext(extension.begin(), extension.end());
			filtered = DeletedFileScanner::FilterByExtension(filtered, wext);
		}

		if (pattern != "*") {
			wstring wpattern(pattern.begin(), pattern.end());
			filtered = DeletedFileScanner::FilterByName(filtered, wpattern);
		}

		cout << "\n===== Search Results =====" << endl;
		cout << "Found: " << filtered.size() << " matching files." << endl;

		size_t displayLimit = min(filtered.size(), (size_t)100);
		for (size_t i = 0; i < displayLimit; i++) {
			const auto& info = filtered[i];
			cout << "[" << info.recordNumber << "] ";
			cout << info.fileSize << " bytes | ";
			wcout << info.filePath << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// ListDeletedFilesCommand - 列出已删除文件
// ============================================================================
DEFINE_COMMAND_BASE(ListDeletedFilesCommand, "listdeleted |name |name", TRUE)
REGISTER_COMMAND(ListDeletedFilesCommand);

void ListDeletedFilesCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 1 || GET_ARG_COUNT() > 2) {
		cout << "Invalid Args! Usage: listdeleted <drive_letter> [filter_level]" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		FilterLevel filterLevel = FILTER_SKIP_PATH;
		if (HAS_ARG(1)) {
			string& filterStr = GET_ARG_STRING(1);
			if (filterStr == "none") filterLevel = FILTER_NONE;
			else if (filterStr == "exclude") filterLevel = FILTER_EXCLUDE;
		}

		cout << "Scanning drive " << driveLetter << ": for deleted files..." << endl;

		FileRestore* fileRestore = new FileRestore();
		fileRestore->SetFilterLevel(filterLevel);
		vector<DeletedFileInfo> deletedFiles = fileRestore->ScanDeletedFiles(driveLetter, 0);
		delete fileRestore;

		if (deletedFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		DeletedFileScanner::SaveToCache(deletedFiles, driveLetter);

		cout << "\n===== Deleted Files on " << driveLetter << ": =====" << endl;
		cout << "Found: " << deletedFiles.size() << " deleted files." << endl;

		size_t displayLimit = min(deletedFiles.size(), (size_t)100);
		for (size_t i = 0; i < displayLimit; i++) {
			const auto& info = deletedFiles[i];
			cout << "[" << info.recordNumber << "] ";
			cout << info.fileSize << " bytes | ";
			wcout << info.filePath << endl;
		}

		if (deletedFiles.size() > 100) {
			cout << "\nNote: Showing first 100 of " << deletedFiles.size() << " files." << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// RestoreByRecordCommand - 按记录号恢复文件
// ============================================================================
DEFINE_COMMAND_BASE(RestoreByRecordCommand, "restorebyrecord |name |name |file", TRUE)
REGISTER_COMMAND(RestoreByRecordCommand);

void RestoreByRecordCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() != 3) {
		cout << "Invalid Args! Usage: restorebyrecord <drive_letter> <MFT_record_number> <output_path>" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& recordStr = GET_ARG_STRING(1);
		string& outputPath = GET_ARG_STRING(2);

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

		FileRestore* fileRestore = new FileRestore();
		OverwriteDetectionResult result = fileRestore->DetectFileOverwrite(driveLetter, recordNumber);

		if (!result.isFullyAvailable && !result.isPartiallyAvailable) {
			cout << "\n[FAILED] File data has been completely overwritten." << endl;
			delete fileRestore;
			return;
		}

		bool success = fileRestore->RestoreFileByRecordNumber(driveLetter, recordNumber, outputPath);

		if (success) {
			cout << "\n=== Recovery Successful ===" << endl;
			cout << "File has been saved to: " << outputPath << endl;
		}
		else {
			cout << "\n=== Recovery Failed ===" << endl;
		}

		delete fileRestore;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// ForceRestoreCommand - 强制恢复文件（跳过覆盖检测）
// ============================================================================
DEFINE_COMMAND_BASE(ForceRestoreCommand, "forcerestore |name |name |file", TRUE)
REGISTER_COMMAND(ForceRestoreCommand);

void ForceRestoreCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() != 3) {
		cout << "Invalid Args! Usage: forcerestore <drive_letter> <MFT_record_number> <output_path>" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& recordStr = GET_ARG_STRING(1);
		string& outputPath = GET_ARG_STRING(2);

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

		cout << "=== FORCE File Recovery (Overwrite Detection Bypassed) ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "MFT Record: " << recordNumber << endl;
		cout << "Output Path: " << outputPath << endl;
		cout << endl;

		cout << "WARNING: This command bypasses overwrite detection!" << endl;
		cout << "         - May recover corrupted or partial data" << endl;
		cout << "         - May recover random data if clusters were reused" << endl;
		cout << "         - Useful for SSD TRIM, high-entropy files, or false positives" << endl;
		cout << endl;

		FileRestore* fileRestore = new FileRestore();

		// 直接尝试恢复，不检测覆盖
		cout << "Attempting forced recovery..." << endl;
		bool success = fileRestore->RestoreFileByRecordNumber(driveLetter, recordNumber, outputPath);

		if (success) {
			cout << "\n=== Recovery Completed ===" << endl;
			cout << "File has been saved to: " << outputPath << endl;
			cout << "\nIMPORTANT: Please verify the recovered file!" << endl;
			cout << "           - Check file size" << endl;
			cout << "           - Open with appropriate application" << endl;
			cout << "           - Compare with known good file if possible" << endl;
		}
		else {
			cout << "\n=== Recovery Failed ===" << endl;
			cout << "Possible reasons:" << endl;
			cout << "  - MFT record is completely invalid" << endl;
			cout << "  - No DATA attribute found" << endl;
			cout << "  - Cannot read cluster data" << endl;
		}

		delete fileRestore;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// ScanUsnCommand - 扫描 USN 日志
// ============================================================================
DEFINE_COMMAND_BASE(ScanUsnCommand, "scanusn |name |name", TRUE)
REGISTER_COMMAND(ScanUsnCommand);

void ScanUsnCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 1 || GET_ARG_COUNT() > 2) {
		cout << "Invalid Args! Usage: scanusn <drive_letter> [max_hours]" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		int maxHours = 1;

		if (HAS_ARG(1)) {
			string& hoursStr = GET_ARG_STRING(1);
			try {
				maxHours = stoi(hoursStr);
				if (maxHours <= 0) maxHours = 1;
			}
			catch (...) {}
		}

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		cout << "\n========== USN Journal Scanner ==========" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Time range: Last " << maxHours << " hour(s)" << endl;

		UsnJournalReader usnReader;

		if (!usnReader.Open(driveLetter)) {
			cout << "ERROR: " << usnReader.GetLastError() << endl;
			return;
		}

		// 显示 USN Journal 统计信息（诊断用）
		UsnJournalStats stats;
		if (usnReader.GetJournalStats(stats)) {
			cout << "\n--- USN Journal Statistics ---" << endl;
			cout << "Journal ID: " << stats.UsnJournalID << endl;
			cout << "First USN: " << stats.FirstUsn << endl;
			cout << "Next USN: " << stats.NextUsn << endl;
			cout << "Max Size: " << (stats.MaximumSize / (1024 * 1024)) << " MB" << endl;

			// 计算 Journal 使用率
			ULONGLONG usedSize = stats.NextUsn - stats.FirstUsn;
			double usagePercent = (stats.MaximumSize > 0) ?
				((double)usedSize / stats.MaximumSize * 100.0) : 0;
			cout << "Usage: ~" << fixed << setprecision(1) << usagePercent << "%" << endl;

			if (usagePercent > 90.0) {
				cout << "WARNING: Journal is nearly full! Old records may have been overwritten." << endl;
			}
			cout << "------------------------------" << endl;
		}

		int maxTimeSeconds = maxHours * 3600;
		vector<UsnDeletedFileInfo> deletedFiles = usnReader.ScanRecentlyDeletedFiles(maxTimeSeconds, 10000);

		if (deletedFiles.empty()) {
			cout << "\nNo deleted files found in the specified time range." << endl;
			cout << "\nPossible reasons:" << endl;
			cout << "  1. No files were deleted in the last " << maxHours << " hour(s)" << endl;
			cout << "  2. USN Journal may have wrapped (old records overwritten)" << endl;
			cout << "  3. Files were moved to Recycle Bin (try searching by name)" << endl;
			cout << "  4. Try increasing the time range: scanusn " << driveLetter << " 24" << endl;
			return;
		}

		cout << "\n===== Recently Deleted Files =====" << endl;
		cout << "Found: " << deletedFiles.size() << " deleted file records" << endl;
		cout << "(Sorted by time, newest first)" << endl;
		cout << endl;

		size_t displayLimit = min(deletedFiles.size(), (size_t)100);
		for (size_t i = 0; i < displayLimit; i++) {
			const auto& info = deletedFiles[i];

			SYSTEMTIME st;
			FILETIME ft;
			ZeroMemory(&ft, sizeof(FILETIME));
			ft.dwLowDateTime = info.TimeStamp.LowPart;
			ft.dwHighDateTime = info.TimeStamp.HighPart;
			FileTimeToSystemTime(&ft, &st);

			// 转换为本地时间
			SYSTEMTIME localSt;
			SystemTimeToTzSpecificLocalTime(NULL, &st, &localSt);

			cout << "[" << info.GetMftRecordNumber() << "] ";
			wcout << info.FileName << " | ";
			printf("%04d-%02d-%02d %02d:%02d:%02d",
				localSt.wYear, localSt.wMonth, localSt.wDay,
				localSt.wHour, localSt.wMinute, localSt.wSecond);

			// 显示操作类型
			if (info.Reason & USN_REASON_FILE_DELETE) {
				cout << " [DELETE]";
			}
			if (info.Reason & USN_REASON_RENAME_OLD_NAME) {
				cout << " [RENAME/MOVE]";
			}
			cout << endl;
		}

		if (deletedFiles.size() > displayLimit) {
			cout << "\n... and " << (deletedFiles.size() - displayLimit) << " more records" << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// DiagnoseFileCommand - 文件诊断
// ============================================================================
DEFINE_COMMAND_BASE(DiagnoseFileCommand, "diagnosefile |name |name", TRUE)
REGISTER_COMMAND(DiagnoseFileCommand);

void DiagnoseFileCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() != 2) {
		cout << "Invalid Args! Usage: diagnosefile <drive_letter> <filename>" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& fileNameStr = GET_ARG_STRING(1);

		if (driveStr.empty() || fileNameStr.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter = driveStr[0];
		wstring searchName(fileNameStr.begin(), fileNameStr.end());

		cout << "\n========== File Diagnostic Tool ==========" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		wcout << L"Searching for: " << searchName << endl;

		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "ERROR: Failed to open volume " << driveLetter << ":" << endl;
			return;
		}

		MFTParser parser(&reader);
		PathResolver pathResolver(&reader, &parser);

		ULONGLONG totalRecords = reader.GetTotalMFTRecords();
		cout << "Total MFT records: " << totalRecords << endl;

		vector<BYTE> record;
		ULONGLONG foundCount = 0;
		ULONGLONG scannedCount = 0;

		wstring searchNameLower = searchName;
		transform(searchNameLower.begin(), searchNameLower.end(), searchNameLower.begin(), ::towlower);

		cout << "Attempting to load path cache..." << endl;
		if (pathResolver.LoadCache(driveLetter)) {
			cout << "Path cache loaded (" << pathResolver.GetCacheSize() << " entries)" << endl;

			auto& cache = pathResolver.GetCacheRef();

			for (const auto& entry : cache) {
				ULONGLONG recordNum = entry.first;
				const wstring& fullPath = entry.second;

				size_t lastSlash = fullPath.find_last_of(L"\\/");
				wstring fileName = (lastSlash != wstring::npos) ?
					fullPath.substr(lastSlash + 1) : fullPath;

				wstring fileNameLower = fileName;
				transform(fileNameLower.begin(), fileNameLower.end(), fileNameLower.begin(), ::towlower);

				if (fileNameLower.find(searchNameLower) != wstring::npos) {
					if (!reader.ReadMFT(recordNum, record)) continue;

					FILE_RECORD_HEADER* header = (FILE_RECORD_HEADER*)record.data();
					bool isDeleted = ((header->Flags & 0x01) == 0);
					bool isDirectory = ((header->Flags & 0x02) != 0);

					foundCount++;

					cout << "\n[" << foundCount << "] MFT Record #" << recordNum << endl;
					wcout << L"  Name: " << fileName << endl;
					cout << "  Status: " << (isDeleted ? "DELETED" : "ACTIVE") << endl;
					cout << "  Type: " << (isDirectory ? "Directory" : "File") << endl;
					wcout << L"  Full Path: " << fullPath << endl;

					if (foundCount >= 50) break;
				}
				scannedCount++;
			}
		}
		else {
			cout << "Cache not available, performing MFT scan..." << endl;
		}

		cout << "\n========== Scan Results ==========" << endl;
		cout << "Total matches found: " << foundCount << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// SearchUsnCommand - USN 搜索已删除文件
// ============================================================================
DEFINE_COMMAND_BASE(SearchUsnCommand, "searchusn |name |name |name", TRUE)
REGISTER_COMMAND(SearchUsnCommand);

void SearchUsnCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 2) {
		cout << "Usage: searchusn <drive_letter> <filename> [exact]" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& filenameStr = GET_ARG_STRING(1);

		bool exactMatch = false;
		if (HAS_ARG(2)) {
			string& matchMode = GET_ARG_STRING(2);
			exactMatch = (matchMode == "exact" || matchMode == "e");
		}

		if (driveStr.empty() || filenameStr.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter = driveStr[0];
		wstring searchName(filenameStr.begin(), filenameStr.end());

		cout << "=== USN Journal Search ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		wcout << L"Searching for: " << searchName << endl;

		UsnJournalReader reader;
		if (!reader.Open(driveLetter)) {
			cout << "Failed to open USN Journal" << endl;
			return;
		}

		auto results = reader.SearchDeletedByName(searchName, exactMatch);

		if (results.empty()) {
			cout << "\nNo deleted files found matching '" << filenameStr << "'" << endl;
			return;
		}

		cout << "\n=== Found " << results.size() << " deleted file(s) ===" << endl;

		for (size_t i = 0; i < results.size(); i++) {
			const auto& info = results[i];
			cout << "\n[" << (i + 1) << "] MFT Record #" << info.GetMftRecordNumber() << endl;
			wcout << L"  Name: " << info.FileName << endl;
			cout << "  Parent Record: " << info.GetParentMftRecordNumber() << endl;

			FILETIME ft;
			ft.dwLowDateTime = info.TimeStamp.LowPart;
			ft.dwHighDateTime = info.TimeStamp.HighPart;
			SYSTEMTIME st;
			FileTimeToSystemTime(&ft, &st);
			cout << "  Deleted: " << st.wYear << "-"
				<< setfill('0') << setw(2) << st.wMonth << "-"
				<< setw(2) << st.wDay << " "
				<< setw(2) << st.wHour << ":"
				<< setw(2) << st.wMinute << ":"
				<< setw(2) << st.wSecond << endl;
		}

		cout << "\n=== Recovery Instructions ===" << endl;
		cout << "To restore: restorebyrecord " << driveLetter << " <record_number> <output_path>" << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// FilterSizeCommand - 按文件大小过滤
// ============================================================================
DEFINE_COMMAND_BASE(FilterSizeCommand, "filtersize |name |name |name |name", TRUE)
REGISTER_COMMAND(FilterSizeCommand);

void FilterSizeCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 3) {
		cout << "Usage: filtersize <drive_letter> <min_size> <max_size> [limit]" << endl;
		cout << "Size format: number + unit (B/K/M/G)" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& minSizeStr = GET_ARG_STRING(1);
		string& maxSizeStr = GET_ARG_STRING(2);

		size_t displayLimit = 100;
		if (HAS_ARG(3)) {
			string& limitStr = GET_ARG_STRING(3);
			displayLimit = stoull(limitStr);
		}

		auto parseSize = [](const string& sizeStr) -> ULONGLONG {
			string numPart;
			char unit = 'B';
			for (char c : sizeStr) {
				if (isdigit(c) || c == '.') {
					numPart += c;
				}
				else {
					unit = toupper(c);
					break;
				}
			}

			double value = stod(numPart);
			switch (unit) {
			case 'K': return (ULONGLONG)(value * 1024);
			case 'M': return (ULONGLONG)(value * 1024 * 1024);
			case 'G': return (ULONGLONG)(value * 1024 * 1024 * 1024);
			default: return (ULONGLONG)value;
			}
			};

		char driveLetter = driveStr[0];
		ULONGLONG minSize = parseSize(minSizeStr);
		ULONGLONG maxSize = parseSize(maxSizeStr);

		cout << "=== Filter Deleted Files by Size ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Size range: " << minSize << " - " << maxSize << " bytes" << endl;

		vector<DeletedFileInfo> allFiles;
		if (DeletedFileScanner::IsCacheValid(driveLetter, 60)) {
			cout << "Loading from cache..." << endl;
			DeletedFileScanner::LoadFromCache(allFiles, driveLetter);
		}

		if (allFiles.empty()) {
			cout << "Scanning deleted files..." << endl;
			FileRestore fr;
			allFiles = fr.ScanDeletedFiles(driveLetter, 0);
			if (!allFiles.empty()) {
				DeletedFileScanner::SaveToCache(allFiles, driveLetter);
			}
		}

		if (allFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		auto filtered = DeletedFileScanner::FilterBySize(allFiles, minSize, maxSize);

		cout << "\n=== Found " << filtered.size() << " file(s) ===" << endl;

		size_t displayCount = min(filtered.size(), displayLimit);
		for (size_t i = 0; i < displayCount; i++) {
			const auto& file = filtered[i];
			cout << "\n[" << (i + 1) << "] MFT Record #" << file.recordNumber << endl;
			wcout << L"  Name: " << file.fileName << endl;
			cout << "  Size: " << file.fileSize << " bytes" << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// FindRecordCommand - 查找 MFT 记录号
// ============================================================================
DEFINE_COMMAND_BASE(FindRecordCommand, "findrecord |name |file", TRUE)
REGISTER_COMMAND(FindRecordCommand);

void FindRecordCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 2) {
		cout << "Usage: findrecord <drive_letter> <file_path>" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& filePath = GET_ARG_STRING(1);

		if (driveStr.empty() || filePath.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		cout << "=== Find MFT Record Number ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Path: " << filePath << endl;

		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "Failed to open volume " << driveLetter << ":/" << endl;
			return;
		}

		MFTParser parser(&reader);
		PathResolver resolver(&reader, &parser);

		cout << "Searching for file..." << endl;
		ULONGLONG recordNumber = resolver.FindFileRecordByPath(filePath);

		if (recordNumber == 0) {
			cout << "\n[NOT FOUND] File not found in MFT." << endl;
			return;
		}

		cout << "\n=== File Found ===" << endl;
		cout << "MFT Record Number: " << recordNumber << endl;

		vector<BYTE> record;
		if (reader.ReadMFT(recordNumber, record)) {
			FILE_RECORD_HEADER* header = (FILE_RECORD_HEADER*)record.data();
			bool isDeleted = ((header->Flags & 0x01) == 0);
			bool isDirectory = ((header->Flags & 0x02) != 0);

			cout << "Status: " << (isDeleted ? "DELETED" : "ACTIVE") << endl;
			cout << "Type: " << (isDirectory ? "Directory" : "File") << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// FindUserFilesCommand - 查找用户文件
// ============================================================================
DEFINE_COMMAND_BASE(FindUserFilesCommand, "finduserfiles |name |name", TRUE)
REGISTER_COMMAND(FindUserFilesCommand);

void FindUserFilesCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 1) {
		cout << "Usage: finduserfiles <drive_letter> [limit]" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);

		size_t displayLimit = 100;
		if (HAS_ARG(1)) {
			string& limitStr = GET_ARG_STRING(1);
			displayLimit = stoull(limitStr);
		}

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		cout << "=== Find User Files ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;

		vector<DeletedFileInfo> allFiles;
		if (DeletedFileScanner::IsCacheValid(driveLetter, 60)) {
			cout << "Loading from cache..." << endl;
			DeletedFileScanner::LoadFromCache(allFiles, driveLetter);
		}

		if (allFiles.empty()) {
			cout << "Scanning deleted files..." << endl;
			FileRestore fr;
			allFiles = fr.ScanDeletedFiles(driveLetter, 0);
			if (!allFiles.empty()) {
				DeletedFileScanner::SaveToCache(allFiles, driveLetter);
			}
		}

		if (allFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		auto userFiles = DeletedFileScanner::FilterUserFiles(allFiles);

		if (userFiles.empty()) {
			cout << "\nNo user files found." << endl;
			return;
		}

		cout << "\n=== Found " << userFiles.size() << " user file(s) ===" << endl;

		map<wstring, vector<DeletedFileInfo>> filesByType;
		for (const auto& file : userFiles) {
			wstring ext = DeletedFileScanner::GetFileExtension(file.fileName);
			transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
			filesByType[ext].push_back(file);
		}

		cout << "\n=== File Type Summary ===" << endl;
		for (const auto& pair : filesByType) {
			wcout << L"  " << pair.first << L": " << pair.second.size() << L" files" << endl;
		}

		cout << "\n=== File List ===" << endl;
		size_t displayCount = min(userFiles.size(), displayLimit);
		for (size_t i = 0; i < displayCount; i++) {
			const auto& file = userFiles[i];
			cout << "\n[" << (i + 1) << "] MFT Record #" << file.recordNumber << endl;
			wcout << L"  Name: " << file.fileName << endl;
			cout << "  Size: " << file.fileSize << " bytes" << endl;
		}

		if (userFiles.size() > displayLimit) {
			cout << "\n(Showing first " << displayLimit << " of " << userFiles.size() << " files)" << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// BatchRestoreCommand - 批量恢复文件
// ============================================================================
DEFINE_COMMAND_BASE(BatchRestoreCommand, "batchrestore |name |name |file", TRUE)
REGISTER_COMMAND(BatchRestoreCommand);

void BatchRestoreCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 3) {
		cout << "Usage: batchrestore <drive_letter> <record_numbers> <output_directory>" << endl;
		cout << "Record numbers format: comma-separated list" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& recordsStr = GET_ARG_STRING(1);
		string& outputDir = GET_ARG_STRING(2);

		if (driveStr.empty() || recordsStr.empty() || outputDir.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		// Parse record numbers
		vector<ULONGLONG> recordNumbers;
		size_t pos = 0;
		string recordsCopy = recordsStr;
		while ((pos = recordsCopy.find(',')) != string::npos) {
			string token = recordsCopy.substr(0, pos);
			if (!token.empty()) {
				try {
					recordNumbers.push_back(stoull(token));
				}
				catch (...) {}
			}
			recordsCopy.erase(0, pos + 1);
		}
		if (!recordsCopy.empty()) {
			try {
				recordNumbers.push_back(stoull(recordsCopy));
			}
			catch (...) {}
		}

		if (recordNumbers.empty()) {
			cout << "No valid record numbers provided." << endl;
			return;
		}

		CreateDirectoryA(outputDir.c_str(), NULL);

		cout << "=== Batch File Recovery ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Output Directory: " << outputDir << endl;
		cout << "Files to restore: " << recordNumbers.size() << endl;

		FileRestore* fileRestore = new FileRestore();

		if (!fileRestore->OpenDrive(driveLetter)) {
			cout << "Failed to open volume " << driveLetter << ":/" << endl;
			delete fileRestore;
			return;
		}

		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "Failed to open MFT reader" << endl;
			delete fileRestore;
			return;
		}

		MFTBatchReader batchReader;
		if (!batchReader.Initialize(&reader)) {
			cout << "Failed to initialize batch reader." << endl;
			delete fileRestore;
			return;
		}

		MFTParser parser(&reader);

		cout << "Pre-loading MFT records..." << endl;
		map<ULONGLONG, vector<BYTE>> preloadedRecords;
		for (ULONGLONG recordNum : recordNumbers) {
			vector<BYTE> record;
			if (batchReader.ReadMFTRecord(recordNum, record)) {
				preloadedRecords[recordNum] = record;
			}
		}
		cout << "Pre-loaded " << preloadedRecords.size() << " records." << endl;

		size_t successCount = 0;
		size_t failCount = 0;
		size_t skipCount = 0;

		ProgressBar progress(recordNumbers.size(), 40);
		progress.Show();

		for (size_t i = 0; i < recordNumbers.size(); i++) {
			ULONGLONG recordNum = recordNumbers[i];

			progress.Update(i + 1, successCount);

			auto it = preloadedRecords.find(recordNum);
			if (it == preloadedRecords.end() || it->second.empty()) {
				failCount++;
				continue;
			}

			vector<BYTE>& record = it->second;
			ULONGLONG parentDir;
			wstring fileName = parser.GetFileNameFromRecord(record, parentDir);

			if (fileName.empty()) {
				fileName = L"file_" + to_wstring(recordNum);
			}

			string fileNameStr(fileName.begin(), fileName.end());
			string outputPath = outputDir;
			if (outputPath.back() != '\\' && outputPath.back() != '/') {
				outputPath += '\\';
			}
			outputPath += fileNameStr;

			DWORD fileAttr = GetFileAttributesA(outputPath.c_str());
			if (fileAttr != INVALID_FILE_ATTRIBUTES) {
				size_t dotPos = outputPath.find_last_of('.');
				string baseName = (dotPos != string::npos) ? outputPath.substr(0, dotPos) : outputPath;
				string extension = (dotPos != string::npos) ? outputPath.substr(dotPos) : "";

				int suffix = 1;
				do {
					outputPath = baseName + "_" + to_string(suffix) + extension;
					suffix++;
					fileAttr = GetFileAttributesA(outputPath.c_str());
				} while (fileAttr != INVALID_FILE_ATTRIBUTES && suffix < 1000);
			}

			OverwriteDetectionResult result = fileRestore->DetectFileOverwrite(driveLetter, recordNum);

			if (!result.isFullyAvailable && !result.isPartiallyAvailable) {
				skipCount++;
				continue;
			}

			bool success = fileRestore->RestoreFileByRecordNumber(driveLetter, recordNum, outputPath);

			if (success) {
				successCount++;
			}
			else {
				failCount++;
			}
		}

		progress.Finish();

		cout << "\n=== Batch Recovery Summary ===" << endl;
		cout << "Total files: " << recordNumbers.size() << endl;
		cout << "Successfully restored: " << successCount << endl;
		cout << "Failed to restore: " << failCount << endl;
		cout << "Skipped (overwritten): " << skipCount << endl;
		cout << "\nOutput directory: " << outputDir << endl;

		batchReader.ClearCache();
		delete fileRestore;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
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

// 静态变量存储扫描结果（内存模式）
static vector<CarvedFileInfo> lastCarveResults;
static char lastCarveDrive = 0;

// 缓存和内存映射管理（大数据集模式）
static CarvedResultsCache* resultsCache = nullptr;
static MemoryMappedResults* mappedResults = nullptr;
static bool useMemoryMapping = false;
static size_t currentPageIndex = 0;
static size_t pageSize = 50;  // 每页显示50条记录

// 辅助函数：获取总记录数
static size_t GetTotalResultCount() {
    if (useMemoryMapping && mappedResults && mappedResults->IsValid()) {
        return (size_t)mappedResults->GetTotalRecords();
    } else if (resultsCache) {
        ULONGLONG count;
        char drive;
        if (resultsCache->GetCacheInfo(count, drive)) {
            return (size_t)count;
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

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		// 创建输出目录
		CreateDirectoryA(outputDir.c_str(), NULL);

		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "Failed to open volume " << driveLetter << ":" << endl;
			return;
		}

		FileCarver carver(&reader);
		carver.SetAsyncMode(useAsync);

		vector<CarvedFileInfo> results;

		// 解析文件类型
		vector<string> types;
		if (fileTypeArg == "all") {
			// 获取所有类型
			for (const auto& t : carver.GetSupportedTypes()) {
				size_t dashPos = t.find(" - ");
				if (dashPos != string::npos) {
					types.push_back(t.substr(0, dashPos));
				}
			}
		} else {
			// 支持逗号分隔的多类型：jpg,png,gif
			string typeCopy = fileTypeArg;
			size_t pos = 0;
			while ((pos = typeCopy.find(',')) != string::npos) {
				string type = typeCopy.substr(0, pos);
				if (!type.empty()) {
					types.push_back(type);
				}
				typeCopy.erase(0, pos + 1);
			}
			if (!typeCopy.empty()) {
				types.push_back(typeCopy);
			}
		}

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
		cout << "\nRun 'carve' first to scan for files." << endl;
		return;
	}

	try {
		string& indexStr = GET_ARG_STRING(0);
		string& outputPath = GET_ARG_STRING(1);

		size_t index = stoull(indexStr);

		if (lastCarveResults.empty()) {
			cout << "No carving results available. Run 'carve' first." << endl;
			return;
		}

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

void CarveCommandThreadPool::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 3) {
		cout << "Usage: carvepool <drive> <type|types|all> <output_dir> [threads] [notimestamp]" << endl;
		cout << "\nThread Pool Parallel Signature Scanner with Timestamp Sorting" << endl;
		cout << "Files are sorted by: 1) Confidence (high first) 2) Date (earlier first)\n" << endl;
		cout << "Examples:" << endl;
		cout << "  carvepool C zip D:\\recovered\\" << endl;
		cout << "  carvepool C jpg,png D:\\recovered\\ 8" << endl;
		cout << "  carvepool D all D:\\recovered\\ 0 notimestamp" << endl;
		cout << "\nOptions:" << endl;
		cout << "  [threads]     - Worker threads (0 = auto-detect)" << endl;
		cout << "  [notimestamp] - Skip timestamp extraction (faster)" << endl;
		cout << "\nUse 'carvetypes' to see supported file types." << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& fileTypeArg = GET_ARG_STRING(1);
		string& outputDir = GET_ARG_STRING(2);

		// 解析选项
		int threadCount = 0;
		bool extractTimestamps = true;

		if (HAS_ARG(3)) {
			string& arg3 = GET_ARG_STRING(3);
			if (arg3 == "notimestamp" || arg3 == "nots") {
				extractTimestamps = false;
			} else {
				try {
					threadCount = stoi(arg3);
					if (threadCount < 0) threadCount = 0;
					if (threadCount > 64) threadCount = 64;
				} catch (...) {}
			}
		}

		if (HAS_ARG(4)) {
			string& arg4 = GET_ARG_STRING(4);
			if (arg4 == "notimestamp" || arg4 == "nots") {
				extractTimestamps = false;
			}
		}

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		// 创建输出目录
		CreateDirectoryA(outputDir.c_str(), NULL);

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

		// 解析文件类型
		vector<string> types;
		if (fileTypeArg == "all") {
			for (const auto& t : carver.GetSupportedTypes()) {
				size_t dashPos = t.find(" - ");
				if (dashPos != string::npos) {
					types.push_back(t.substr(0, dashPos));
				}
			}
		} else {
			string typeCopy = fileTypeArg;
			size_t pos = 0;
			while ((pos = typeCopy.find(',')) != string::npos) {
				string type = typeCopy.substr(0, pos);
				if (!type.empty()) {
					types.push_back(type);
				}
				typeCopy.erase(0, pos + 1);
			}
			if (!typeCopy.empty()) {
				types.push_back(typeCopy);
			}
		}

		// 使用线程池扫描
		results = carver.ScanForFileTypesThreadPool(types, CARVE_SMART, 1000);

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
