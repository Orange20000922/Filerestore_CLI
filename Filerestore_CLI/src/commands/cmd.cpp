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
#include <algorithm>
#include <map>
using namespace std;
vector<LPVOID> HelpCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> QueueDLLsCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> GetProcessFuncAddressCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> ExitCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> PrintAllFunction::Arglist = vector<LPVOID>();
vector<LPVOID> ListDeletedFilesCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> RestoreByRecordCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> DiagnoseMFTCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> DetectOverwriteCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> SearchDeletedFilesCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> DiagnoseFileCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> SearchUsnCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> FilterSizeCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> FindRecordCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> FindUserFilesCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> BatchRestoreCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> SetLanguageCommand::ArgsList = vector<LPVOID>();
vector<LPVOID> ScanUsnCommand::ArgsList = vector<LPVOID>();
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
		string& cmdName = *(string*)ArgsList[0];

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
			cout << "功能:" << endl;
			cout << "  - 实时进度条显示" << endl;
			cout << "  - 自动跳过已覆盖的文件" << endl;
			cout << "  - 自动处理文件名冲突" << endl;
			cout << "  - 显示详细恢复统计\n" << endl;
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
			cout << "说明:" << endl;
			cout << "  - 自动缓存扫描结果，加速后续查询" << endl;
			cout << "  - 显示文件名、大小、路径等信息" << endl;
			cout << "  - 大型驱动器扫描可能需要数分钟\n" << endl;
		}
		else if (cmdName == "searchdeleted") {
			cout << "\n=== searchdeleted - 搜索已删除文件 ===\n" << endl;
			cout << "用法: searchdeleted <drive> <pattern> [extension] [filter_level]\n" << endl;
			cout << "参数:" << endl;
			cout << "  <drive>        - 驱动器字母" << endl;
			cout << "  <pattern>      - 文件名搜索模式 (支持通配符*)" << endl;
			cout << "  [extension]    - 文件扩展名 (可选, 如: .pdf)" << endl;
			cout << "  [filter_level] - 过滤级别: none/skip/exclude (可选)\n" << endl;
			cout << "示例:" << endl;
			cout << "  searchdeleted C document" << endl;
			cout << "  searchdeleted C report .pdf" << endl;
			cout << "  searchdeleted D * .jpg skip\n" << endl;
			cout << "过滤级别:" << endl;
			cout << "  none    - 重建所有文件路径 (最慢，最完整)" << endl;
			cout << "  skip    - 跳过系统文件路径重建 (默认，平衡)" << endl;
			cout << "  exclude - 排除系统文件 (最快，结果最少)\n" << endl;
		}
		else if (cmdName == "searchusn") {
			cout << "\n=== searchusn - USN日志搜索 ===\n" << endl;
			cout << "用法: searchusn <drive> <filename> [exact]\n" << endl;
			cout << "参数:" << endl;
			cout << "  <drive>     - 驱动器字母" << endl;
			cout << "  <filename>  - 要搜索的文件名" << endl;
			cout << "  [exact]     - 精确匹配模式 (可选)\n" << endl;
			cout << "示例:" << endl;
			cout << "  searchusn C document.docx" << endl;
			cout << "  searchusn D report.pdf exact\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 仅搜索最近的文件操作记录" << endl;
			cout << "  - 比扫描整个MFT快得多" << endl;
			cout << "  - 适合查找刚删除的文件" << endl;
			cout << "  - 显示删除时间和父目录信息\n" << endl;
		}
		else if (cmdName == "filtersize") {
			cout << "\n=== filtersize - 按大小筛选文件 ===\n" << endl;
			cout << "用法: filtersize <drive> <min_size> <max_size> [limit]\n" << endl;
			cout << "参数:" << endl;
			cout << "  <drive>     - 驱动器字母" << endl;
			cout << "  <min_size>  - 最小大小 (支持单位: B/K/M/G)" << endl;
			cout << "  <max_size>  - 最大大小" << endl;
			cout << "  [limit]     - 最大显示数量 (可选, 默认100)\n" << endl;
			cout << "示例:" << endl;
			cout << "  filtersize C 1M 100M       - 1MB到100MB之间的文件" << endl;
			cout << "  filtersize D 0B 1K         - 小于1KB的文件" << endl;
			cout << "  filtersize C 1G 10G 50     - 1GB到10GB，显示50个\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 自动使用缓存加速查询" << endl;
			cout << "  - 支持多种大小单位" << endl;
			cout << "  - 显示文件大小的友好格式\n" << endl;
		}
		else if (cmdName == "finduserfiles") {
			cout << "\n=== finduserfiles - 查找用户文件 ===\n" << endl;
			cout << "用法: finduserfiles <drive> [limit]\n" << endl;
			cout << "参数:" << endl;
			cout << "  <drive>  - 驱动器字母" << endl;
			cout << "  [limit]  - 最大显示数量 (可选, 默认100)\n" << endl;
			cout << "示例:" << endl;
			cout << "  finduserfiles C" << endl;
			cout << "  finduserfiles D 200\n" << endl;
			cout << "用户文件类型:" << endl;
			cout << "  - 文档: .doc, .docx, .pdf, .txt, .xls, .xlsx, .ppt, .pptx" << endl;
			cout << "  - 图片: .jpg, .png, .gif, .bmp, .raw, .tiff" << endl;
			cout << "  - 视频: .mp4, .avi, .mkv, .mov, .wmv" << endl;
			cout << "  - 音频: .mp3, .wav, .flac, .aac" << endl;
			cout << "  - 压缩: .zip, .rar, .7z, .tar, .gz\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 自动按文件类型分组统计" << endl;
			cout << "  - 过滤掉系统文件和临时文件\n" << endl;
		}
		else if (cmdName == "findrecord") {
			cout << "\n=== findrecord - 查找MFT记录号 ===\n" << endl;
			cout << "用法: findrecord <drive> <file_path>\n" << endl;
			cout << "参数:" << endl;
			cout << "  <drive>      - 驱动器字母" << endl;
			cout << "  <file_path>  - 文件完整路径\n" << endl;
			cout << "示例:" << endl;
			cout << "  findrecord C C:\\Windows\\System32\\notepad.exe" << endl;
			cout << "  findrecord D D:\\Documents\\report.docx\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 查找文件对应的MFT记录号" << endl;
			cout << "  - 显示文件状态(活动/已删除)" << endl;
			cout << "  - 可用于后续的恢复操作\n" << endl;
		}
		else if (cmdName == "diagnosefile") {
			cout << "\n=== diagnosefile - 文件诊断 ===\n" << endl;
			cout << "用法: diagnosefile <drive> <filename>\n" << endl;
			cout << "参数:" << endl;
			cout << "  <drive>     - 驱动器字母" << endl;
			cout << "  <filename>  - 要诊断的文件名\n" << endl;
			cout << "示例:" << endl;
			cout << "  diagnosefile C document.docx" << endl;
			cout << "  diagnosefile D photo.jpg\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 搜索MFT中所有匹配的文件" << endl;
			cout << "  - 显示活动和已删除的文件" << endl;
			cout << "  - 尝试重建完整路径" << endl;
			cout << "  - 限制显示前50个匹配结果\n" << endl;
		}
		else if (cmdName == "detectoverwrite") {
			cout << "\n=== detectoverwrite - 覆盖检测 ===\n" << endl;
			cout << "用法: detectoverwrite <drive> <record_number> [mode]\n" << endl;
			cout << "参数:" << endl;
			cout << "  <drive>          - 驱动器字母" << endl;
			cout << "  <record_number>  - MFT记录号" << endl;
			cout << "  [mode]           - 检测模式 (可选)\n" << endl;
			cout << "检测模式:" << endl;
			cout << "  fast      - 快速采样检测 (适合大文件)" << endl;
			cout << "  balanced  - 智能检测 (默认，推荐)" << endl;
			cout << "  thorough  - 完整检测 (最准确，最慢)\n" << endl;
			cout << "示例:" << endl;
			cout << "  detectoverwrite C 12345" << endl;
			cout << "  detectoverwrite D 99999 thorough\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 自动检测存储类型(HDD/SSD/NVMe)" << endl;
			cout << "  - 智能选择检测策略" << endl;
			cout << "  - 显示可恢复百分比" << endl;
			cout << "  - 提供恢复可行性建议\n" << endl;
		}
		else if (cmdName == "diagnosemft") {
			cout << "\n=== diagnosemft - MFT诊断 ===\n" << endl;
			cout << "用法: diagnosemft <drive>\n" << endl;
			cout << "参数:" << endl;
			cout << "  <drive>  - 驱动器字母\n" << endl;
			cout << "示例:" << endl;
			cout << "  diagnosemft C" << endl;
			cout << "  diagnosemft D\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 分析MFT碎片化程度" << endl;
			cout << "  - 显示MFT大小和位置信息" << endl;
			cout << "  - 评估文件系统健康状况\n" << endl;
		}
		else if (cmdName == "scanusn") {
			cout << "\n=== scanusn - USN日志扫描 ===\n" << endl;
			cout << "用法: scanusn <drive> <max_hours>\n" << endl;
			cout << "参数:" << endl;
			cout << "  <drive>      - 驱动器字母" << endl;
			cout << "  <max_hours>  - 扫描时间范围(小时)\n" << endl;
			cout << "示例:" << endl;
			cout << "  scanusn C 24   - 扫描最近24小时的文件操作" << endl;
			cout << "  scanusn D 1    - 扫描最近1小时\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 显示最近的文件删除记录" << endl;
			cout << "  - 包含文件名和时间戳" << endl;
			cout << "  - 比全盘扫描快得多\n" << endl;
		}
		else if (cmdName == "printallfunc") {
			cout << "\n=== printallfunc - 打印PE函数 ===\n" << endl;
			cout << "用法: printallfunc <file_path>\n" << endl;
			cout << "参数:" << endl;
			cout << "  <file_path>  - PE文件路径 (.exe或.dll)\n" << endl;
			cout << "示例:" << endl;
			cout << "  printallfunc C:\\Windows\\System32\\kernel32.dll\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 列出所有导入函数" << endl;
			cout << "  - 按DLL分组显示" << endl;
			cout << "  - 显示函数地址\n" << endl;
		}
		else if (cmdName == "queuedllsname") {
			cout << "\n=== queuedllsname - 列出依赖DLL ===\n" << endl;
			cout << "用法: queuedllsname <file_path>\n" << endl;
			cout << "参数:" << endl;
			cout << "  <file_path>  - PE文件路径\n" << endl;
			cout << "示例:" << endl;
			cout << "  queuedllsname C:\\Program Files\\app.exe\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 显示所有依赖的DLL文件\n" << endl;
		}
		else if (cmdName == "getfuncaddr") {
			cout << "\n=== getfuncaddr - 获取函数地址 ===\n" << endl;
			cout << "用法: getfuncaddr <file_path> <function_name>\n" << endl;
			cout << "参数:" << endl;
			cout << "  <file_path>       - PE文件路径" << endl;
			cout << "  <function_name>   - 函数名称\n" << endl;
			cout << "示例:" << endl;
			cout << "  getfuncaddr C:\\Windows\\System32\\kernel32.dll CreateFileA\n" << endl;
		}
		else if (cmdName == "printallcommand") {
			cout << "\n=== printallcommand - 列出所有命令 ===\n" << endl;
			cout << "用法: printallcommand -list\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 显示所有注册的命令及其参数格式\n" << endl;
		}
		else if (cmdName == "exit") {
			cout << "\n=== exit - 退出程序 ===\n" << endl;
			cout << "用法: exit\n" << endl;
			cout << "说明:" << endl;
			cout << "  - 安全退出程序" << endl;
			cout << "  - 自动清理资源和卸载模块\n" << endl;
		}
		else if (cmdName == "setlang") {
			cout << "\n=== setlang - Language Settings / 语言设置 ===\n" << endl;
			cout << "用法 / Usage: setlang [language_code]\n" << endl;
			cout << "参数 / Parameters:" << endl;
			cout << "  [language_code]  - 语言代码 / Language code (可选 / optional)\n" << endl;
			cout << "示例 / Examples:" << endl;
			cout << "  setlang        - Show current language / 显示当前语言" << endl;
			cout << "  setlang en     - Switch to English / 切换到英文" << endl;
			cout << "  setlang zh     - Switch to Chinese / 切换到中文\n" << endl;
			cout << "支持的语言 / Supported Languages:" << endl;
			cout << "  en  - English" << endl;
			cout << "  zh  - 中文 (Chinese)\n" << endl;
			cout << "说明 / Notes:" << endl;
			cout << "  - Language files are stored in: langs\\<language_code>.json" << endl;
			cout << "  - 语言文件存储在: langs\\<语言代码>.json" << endl;
			cout << "  - Some messages may still appear in previous language" << endl;
			cout << "  - 某些消息可能仍以先前的语言显示\n" << endl;
		}
		else if (cmdName == "help") {
			cout << "\n=== help - 帮助系统 ===\n" << endl;
			cout << "用法: help [command]\n" << endl;
			cout << "参数:" << endl;
			cout << "  [command]  - 命令名称 (可选)\n" << endl;
			cout << "示例:" << endl;
			cout << "  help                  - 显示所有命令概览" << endl;
			cout << "  help restorebyrecord  - 显示restorebyrecord详细帮助\n" << endl;
		}
		else {
			cout << "\n未知命令: " << cmdName << endl;
			cout << "使用 'help' 查看所有可用命令。\n" << endl;
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
		CLI::SetShouldExit(true);  // 设置退出标志而不是直接退出
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
			ZeroMemory(&ft,sizeof(FILETIME));
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

// ==================== SearchUsnCommand Implementation ====================

SearchUsnCommand::SearchUsnCommand()
{
	FlagHasArgs = TRUE;
}

SearchUsnCommand::~SearchUsnCommand()
{
}

void SearchUsnCommand::AcceptArgs(vector<LPVOID> argslist)
{
	SearchUsnCommand::ArgsList = argslist;
}

void SearchUsnCommand::Execute(string command)
{
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.size() < 2) {
		cout << "Usage: searchusn <drive_letter> <filename> [exact]" << endl;
		cout << "Examples:" << endl;
		cout << "  searchusn C document        - Search for files containing 'document'" << endl;
		cout << "  searchusn C report.pdf exact - Exact match for 'report.pdf'" << endl;
		return;
	}

	try {
		string& driveStr = *(string*)ArgsList[0];
		string& filenameStr = *(string*)ArgsList[1];

		bool exactMatch = false;
		if (ArgsList.size() >= 3) {
			string& matchMode = *(string*)ArgsList[2];
			exactMatch = (matchMode == "exact" || matchMode == "e");
		}

		if (driveStr.empty() || filenameStr.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		// Convert filename to wstring
		wstring searchName(filenameStr.begin(), filenameStr.end());

		cout << "=== USN Journal Search ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		wcout << L"Searching for: " << searchName << endl;
		cout << "Match mode: " << (exactMatch ? "Exact" : "Partial") << endl;
		cout << endl;

		UsnJournalReader reader;
		if (!reader.Open(driveLetter)) {
			cout << "Failed to open USN Journal for drive " << driveLetter << ":" << endl;
			cout << "Error: " << reader.GetLastError() << endl;
			cout << "\nPossible reasons:" << endl;
			cout << "  1. Drive does not support USN Journal" << endl;
			cout << "  2. Insufficient privileges (try running as Administrator)" << endl;
			cout << "  3. USN Journal is disabled on this volume" << endl;
			return;
		}

		cout << "Searching USN Journal..." << endl;
		auto results = reader.SearchDeletedByName(searchName, exactMatch);

		if (results.empty()) {
			cout << "\nNo deleted files found matching '" << filenameStr << "'" << endl;
			cout << "\nNote: USN Journal only contains recent file operations." << endl;
			cout << "For older files, use: diagnosefile " << driveLetter << " " << filenameStr << endl;
			return;
		}

		cout << "\n=== Found " << results.size() << " deleted file(s) ===" << endl;

		for (size_t i = 0; i < results.size(); i++) {
			const auto& info = results[i];
			cout << "\n[" << (i + 1) << "] MFT Record #" << info.FileReferenceNumber << endl;
			wcout << L"  Name: " << info.FileName << endl;
			cout << "  Parent Record: " << info.ParentFileReferenceNumber << endl;

			// Convert FILETIME to readable format
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
		cout << "To attempt recovery:" << endl;
		cout << "  1. Check if data is available: detectoverwrite " << driveLetter << " <record_number>" << endl;
		cout << "  2. Restore the file: restorebyrecord " << driveLetter << " <record_number> <output_path>" << endl;

	} catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	} catch (...) {
		cout << "[ERROR] Unknown exception in SearchUsnCommand" << endl;
	}
}

BOOL SearchUsnCommand::HasArgs()
{
	return FlagHasArgs;
}

BOOL SearchUsnCommand::CheckName(string input)
{
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}

// ==================== FilterSizeCommand Implementation ====================

FilterSizeCommand::FilterSizeCommand()
{
	FlagHasArgs = TRUE;
}

FilterSizeCommand::~FilterSizeCommand()
{
}

void FilterSizeCommand::AcceptArgs(vector<LPVOID> argslist)
{
	FilterSizeCommand::ArgsList = argslist;
}

void FilterSizeCommand::Execute(string command)
{
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.size() < 3) {
		cout << "Usage: filtersize <drive_letter> <min_size> <max_size> [limit]" << endl;
		cout << "Size format: number + unit (B/K/M/G)" << endl;
		cout << "Examples:" << endl;
		cout << "  filtersize C 1M 100M       - Files between 1MB and 100MB" << endl;
		cout << "  filtersize C 0B 1K         - Files smaller than 1KB" << endl;
		cout << "  filtersize C 1G 10G 50     - Files between 1GB and 10GB (show first 50)" << endl;
		return;
	}

	try {
		string& driveStr = *(string*)ArgsList[0];
		string& minSizeStr = *(string*)ArgsList[1];
		string& maxSizeStr = *(string*)ArgsList[2];

		size_t displayLimit = 100;
		if (ArgsList.size() >= 4) {
			string& limitStr = *(string*)ArgsList[3];
			displayLimit = stoull(limitStr);
		}

		auto parseSize = [](const string& sizeStr) -> ULONGLONG {
			string numPart;
			char unit = 'B';
			for (char c : sizeStr) {
				if (isdigit(c) || c == '.') {
					numPart += c;
				} else {
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
		cout << endl;

		// Load deleted files from cache or scan
		vector<DeletedFileInfo> allFiles;
		if (DeletedFileScanner::IsCacheValid(driveLetter, 60)) {
			cout << "Loading from cache..." << endl;
			if (DeletedFileScanner::LoadFromCache(allFiles, driveLetter)) {
				cout << "Cache loaded: " << allFiles.size() << " files" << endl;
			}
		}

		if (allFiles.empty()) {
			cout << "No cache available. Scanning deleted files..." << endl;
			cout << "This may take a while. Run 'listdeleted " << driveLetter << " <limit>' first to build cache." << endl;
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

		// Filter by size
		cout << "\nFiltering files..." << endl;
		auto filtered = DeletedFileScanner::FilterBySize(allFiles, minSize, maxSize);

		if (filtered.empty()) {
			cout << "\nNo files found in the specified size range." << endl;
			return;
		}

		cout << "\n=== Found " << filtered.size() << " file(s) ===" << endl;

		size_t displayCount = min(filtered.size(), displayLimit);
		for (size_t i = 0; i < displayCount; i++) {
			const auto& file = filtered[i];
			cout << "\n[" << (i + 1) << "] MFT Record #" << file.recordNumber << endl;
			wcout << L"  Name: " << file.fileName << endl;
			cout << "  Size: " << file.fileSize << " bytes";

			if (file.fileSize >= 1024 * 1024 * 1024) {
				cout << " (" << (file.fileSize / (1024.0 * 1024 * 1024)) << " GB)";
			} else if (file.fileSize >= 1024 * 1024) {
				cout << " (" << (file.fileSize / (1024.0 * 1024)) << " MB)";
			} else if (file.fileSize >= 1024) {
				cout << " (" << (file.fileSize / 1024.0) << " KB)";
			}
			cout << endl;

			if (!file.filePath.empty()) {
				wcout << L"  Path: " << file.filePath << endl;
			}
		}

		if (filtered.size() > displayLimit) {
			cout << "\n(Showing first " << displayLimit << " of " << filtered.size() << " files)" << endl;
			cout << "Use: filtersize " << driveLetter << " " << minSizeStr << " " << maxSizeStr << " <limit> to show more" << endl;
		}

		cout << "\n=== Recovery Instructions ===" << endl;
		cout << "To restore a file: restorebyrecord " << driveLetter << " <record_number> <output_path>" << endl;

	} catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	} catch (...) {
		cout << "[ERROR] Unknown exception in FilterSizeCommand" << endl;
	}
}

BOOL FilterSizeCommand::HasArgs()
{
	return FlagHasArgs;
}

BOOL FilterSizeCommand::CheckName(string input)
{
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}

// ==================== FindRecordCommand Implementation ====================

FindRecordCommand::FindRecordCommand()
{
	FlagHasArgs = TRUE;
}

FindRecordCommand::~FindRecordCommand()
{
}

void FindRecordCommand::AcceptArgs(vector<LPVOID> argslist)
{
	FindRecordCommand::ArgsList = argslist;
}

void FindRecordCommand::Execute(string command)
{
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.size() < 2) {
		cout << "Usage: findrecord <drive_letter> <file_path>" << endl;
		cout << "Examples:" << endl;
		cout << "  findrecord C C:\\Windows\\System32\\notepad.exe" << endl;
		cout << "  findrecord D D:\\Documents\\report.docx" << endl;
		return;
	}

	try {
		string& driveStr = *(string*)ArgsList[0];
		string& filePath = *(string*)ArgsList[1];

		if (driveStr.empty() || filePath.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		cout << "=== Find MFT Record Number ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Path: " << filePath << endl;
		cout << endl;

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
			cout << "\nPossible reasons:" << endl;
			cout << "  1. File does not exist at the specified path" << endl;
			cout << "  2. File has been deleted (try diagnosefile or searchdeleted)" << endl;
			cout << "  3. Invalid path format" << endl;
			return;
		}

		cout << "\n=== File Found ===" << endl;
		cout << "MFT Record Number: " << recordNumber << endl;

		// Get file details
		vector<BYTE> record;
		if (reader.ReadMFT(recordNumber, record)) {
			FILE_RECORD_HEADER* header = (FILE_RECORD_HEADER*)record.data();
			bool isDeleted = ((header->Flags & 0x01) == 0);
			bool isDirectory = ((header->Flags & 0x02) != 0);

			cout << "Status: " << (isDeleted ? "DELETED" : "ACTIVE") << endl;
			cout << "Type: " << (isDirectory ? "Directory" : "File") << endl;

			if (!isDeleted) {
				cout << "\n=== Available Operations ===" << endl;
				cout << "Check file integrity: diagnosefile " << driveLetter << " <filename>" << endl;
			} else {
				cout << "\n=== Recovery Options ===" << endl;
				cout << "1. Check if recoverable: detectoverwrite " << driveLetter << " " << recordNumber << endl;
				cout << "2. Restore file: restorebyrecord " << driveLetter << " " << recordNumber << " <output_path>" << endl;
			}
		}

	} catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	} catch (...) {
		cout << "[ERROR] Unknown exception in FindRecordCommand" << endl;
	}
}

BOOL FindRecordCommand::HasArgs()
{
	return FlagHasArgs;
}

BOOL FindRecordCommand::CheckName(string input)
{
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}

// ==================== FindUserFilesCommand Implementation ====================

FindUserFilesCommand::FindUserFilesCommand()
{
	FlagHasArgs = TRUE;
}

FindUserFilesCommand::~FindUserFilesCommand()
{
}

void FindUserFilesCommand::AcceptArgs(vector<LPVOID> argslist)
{
	FindUserFilesCommand::ArgsList = argslist;
}

void FindUserFilesCommand::Execute(string command)
{
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.size() < 1) {
		cout << "Usage: finduserfiles <drive_letter> [limit]" << endl;
		cout << "Examples:" << endl;
		cout << "  finduserfiles C        - Find all user files on C:" << endl;
		cout << "  finduserfiles D 100    - Find user files on D: (show first 100)" << endl;
		return;
	}

	try {
		string& driveStr = *(string*)ArgsList[0];

		size_t displayLimit = 100;
		if (ArgsList.size() >= 2) {
			string& limitStr = *(string*)ArgsList[1];
			displayLimit = stoull(limitStr);
		}

		if (driveStr.empty()) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		char driveLetter = driveStr[0];

		cout << "=== Find User Files ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "User file types: Documents, Images, Videos, Audio, Archives" << endl;
		cout << endl;

		// Load deleted files from cache or scan
		vector<DeletedFileInfo> allFiles;
		if (DeletedFileScanner::IsCacheValid(driveLetter, 60)) {
			cout << "Loading from cache..." << endl;
			if (DeletedFileScanner::LoadFromCache(allFiles, driveLetter)) {
				cout << "Cache loaded: " << allFiles.size() << " files" << endl;
			}
		}

		if (allFiles.empty()) {
			cout << "No cache available. Scanning deleted files..." << endl;
			cout << "This may take a while..." << endl;
			FileRestore fr;
			allFiles = fr.ScanDeletedFiles(driveLetter, 0);

			if (!allFiles.empty()) {
				cout << "Saving to cache..." << endl;
				DeletedFileScanner::SaveToCache(allFiles, driveLetter);
			}
		}

		if (allFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		// Filter user files
		cout << "\nFiltering user files..." << endl;
		auto userFiles = DeletedFileScanner::FilterUserFiles(allFiles);

		if (userFiles.empty()) {
			cout << "\nNo user files found." << endl;
			return;
		}

		cout << "\n=== Found " << userFiles.size() << " user file(s) ===" << endl;

		// Group by file type
		map<wstring, vector<DeletedFileInfo>> filesByType;
		for (const auto& file : userFiles) {
			wstring ext = DeletedFileScanner::GetFileExtension(file.fileName);
			transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
			filesByType[ext].push_back(file);
		}

		// Display summary
		cout << "\n=== File Type Summary ===" << endl;
		for (const auto& pair : filesByType) {
			wcout << L"  " << pair.first << L": " << pair.second.size() << L" files" << endl;
		}

		// Display detailed list
		cout << "\n=== File List ===" << endl;
		size_t displayCount = min(userFiles.size(), displayLimit);
		for (size_t i = 0; i < displayCount; i++) {
			const auto& file = userFiles[i];
			cout << "\n[" << (i + 1) << "] MFT Record #" << file.recordNumber << endl;
			wcout << L"  Name: " << file.fileName << endl;
			cout << "  Size: " << file.fileSize << " bytes";

			if (file.fileSize >= 1024 * 1024) {
				cout << " (" << (file.fileSize / (1024.0 * 1024)) << " MB)";
			} else if (file.fileSize >= 1024) {
				cout << " (" << (file.fileSize / 1024.0) << " KB)";
			}
			cout << endl;

			if (!file.filePath.empty() && file.filePath != L"<path unavailable>") {
				wcout << L"  Path: " << file.filePath << endl;
			}
		}

		if (userFiles.size() > displayLimit) {
			cout << "\n(Showing first " << displayLimit << " of " << userFiles.size() << " files)" << endl;
			cout << "Use: finduserfiles " << driveLetter << " <limit> to show more" << endl;
		}

		cout << "\n=== Recovery Instructions ===" << endl;
		cout << "To restore a file: restorebyrecord " << driveLetter << " <record_number> <output_path>" << endl;
		cout << "To batch restore: batchrestore " << driveLetter << " <record1,record2,...> <output_directory>" << endl;

	} catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	} catch (...) {
		cout << "[ERROR] Unknown exception in FindUserFilesCommand" << endl;
	}
}

BOOL FindUserFilesCommand::HasArgs()
{
	return FlagHasArgs;
}

BOOL FindUserFilesCommand::CheckName(string input)
{
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}

// ==================== BatchRestoreCommand Implementation ====================

BatchRestoreCommand::BatchRestoreCommand()
{
	FlagHasArgs = TRUE;
}

BatchRestoreCommand::~BatchRestoreCommand()
{
}

void BatchRestoreCommand::AcceptArgs(vector<LPVOID> argslist)
{
	BatchRestoreCommand::ArgsList = argslist;
}

void BatchRestoreCommand::Execute(string command)
{
	if (!CheckName(command)) {
		return;
	}

	if (ArgsList.size() < 3) {
		cout << "Usage: batchrestore <drive_letter> <record_numbers> <output_directory>" << endl;
		cout << "Record numbers format: comma-separated list" << endl;
		cout << "Examples:" << endl;
		cout << "  batchrestore C 12345,12346,12347 C:\\recovered\\" << endl;
		cout << "  batchrestore D 1000,2000,3000 D:\\backup\\" << endl;
		return;
	}

	try {
		string& driveStr = *(string*)ArgsList[0];
		string& recordsStr = *(string*)ArgsList[1];
		string& outputDir = *(string*)ArgsList[2];

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
				} catch (...) {
					cout << "Invalid record number: " << token << endl;
				}
			}
			recordsCopy.erase(0, pos + 1);
		}
		if (!recordsCopy.empty()) {
			try {
				recordNumbers.push_back(stoull(recordsCopy));
			} catch (...) {
				cout << "Invalid record number: " << recordsCopy << endl;
			}
		}

		if (recordNumbers.empty()) {
			cout << "No valid record numbers provided." << endl;
			return;
		}

		// Create output directory if it doesn't exist
		CreateDirectoryA(outputDir.c_str(), NULL);

		cout << "=== Batch File Recovery ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Output Directory: " << outputDir << endl;
		cout << "Files to restore: " << recordNumbers.size() << endl;
		cout << endl;

		FileRestore* fileRestore = new FileRestore();

		// 优化：预先打开卷，避免每次恢复都重新打开
		if (!fileRestore->OpenDrive(driveLetter)) {
			cout << "Failed to open volume " << driveLetter << ":/" << endl;
			delete fileRestore;
			return;
		}
		cout << "Volume opened successfully." << endl;

		// 创建 MFTReader 和 MFTBatchReader 用于批量读取文件名
		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "Failed to open MFT reader for volume " << driveLetter << ":/" << endl;
			delete fileRestore;
			return;
		}

		// 优化：使用 MFTBatchReader 进行缓存读取
		MFTBatchReader batchReader;
		if (!batchReader.Initialize(&reader)) {
			cout << "Failed to initialize batch reader." << endl;
			delete fileRestore;
			return;
		}

		MFTParser parser(&reader);

		// 优化：预先批量读取所有需要的 MFT 记录
		cout << "Pre-loading MFT records..." << endl;
		map<ULONGLONG, vector<BYTE>> preloadedRecords;
		size_t preloadSuccess = 0;
		for (ULONGLONG recordNum : recordNumbers) {
			vector<BYTE> record;
			if (batchReader.ReadMFTRecord(recordNum, record)) {
				preloadedRecords[recordNum] = record;
				preloadSuccess++;
			}
		}
		cout << "Pre-loaded " << preloadSuccess << " / " << recordNumbers.size() << " records." << endl;
		cout << endl;

		size_t successCount = 0;
		size_t failCount = 0;
		size_t skipCount = 0;

		ProgressBar progress(recordNumbers.size(), 40);
		progress.Show();

		for (size_t i = 0; i < recordNumbers.size(); i++) {
			ULONGLONG recordNum = recordNumbers[i];

			progress.Update(i + 1, successCount);

			// 从预加载的缓存中获取记录
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

			// Convert wstring to string for output path
			string fileNameStr(fileName.begin(), fileName.end());
			string outputPath = outputDir;
			if (outputPath.back() != '\\' && outputPath.back() != '/') {
				outputPath += '\\';
			}
			outputPath += fileNameStr;

			// Check if already exists
			DWORD fileAttr = GetFileAttributesA(outputPath.c_str());
			if (fileAttr != INVALID_FILE_ATTRIBUTES) {
				// File exists, add number suffix
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

			// Quick overwrite check (fast mode) - 使用已打开的卷
			OverwriteDetectionResult result = fileRestore->DetectFileOverwrite(driveLetter, recordNum);

			if (!result.isFullyAvailable && !result.isPartiallyAvailable) {
				skipCount++;
				continue;
			}

			// Attempt recovery - 使用已打开的卷
			bool success = fileRestore->RestoreFileByRecordNumber(driveLetter, recordNum, outputPath);

			if (success) {
				successCount++;
			} else {
				failCount++;
			}
		}

		progress.Finish();

		cout << "\n=== Batch Recovery Summary ===" << endl;
		cout << "Total files: " << recordNumbers.size() << endl;
		cout << "Successfully restored: " << successCount << endl;
		cout << "Failed to restore: " << failCount << endl;
		cout << "Skipped (overwritten): " << skipCount << endl;
		cout << "Cache hit rate: " << batchReader.GetCacheSize() << " records cached" << endl;
		cout << "\nOutput directory: " << outputDir << endl;

		// 清理
		batchReader.ClearCache();
		delete fileRestore;

	} catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	} catch (...) {
		cout << "[ERROR] Unknown exception in BatchRestoreCommand" << endl;
	}
}

BOOL BatchRestoreCommand::HasArgs()
{
	return FlagHasArgs;
}

BOOL BatchRestoreCommand::CheckName(string input)
{
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}

// ==================== SetLanguageCommand Implementation ====================

SetLanguageCommand::SetLanguageCommand()
{
	FlagHasArgs = TRUE;
}

SetLanguageCommand::~SetLanguageCommand()
{
}

void SetLanguageCommand::AcceptArgs(vector<LPVOID> argslist)
{
	SetLanguageCommand::ArgsList = argslist;
}

void SetLanguageCommand::Execute(string command)
{
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

		cout << "Usage / 用法: setlang <language_code>" << endl;
		cout << "Example / 示例:" << endl;
		cout << "  setlang en   - Switch to English" << endl;
		cout << "  setlang zh   - 切换到中文\n" << endl;
		return;
	}

	try {
		string& langStr = *(string*)ArgsList[0];

		// Convert to wstring
		wstring langCode(langStr.begin(), langStr.end());

		cout << "\nSwitching language to / 切换语言至: " << langStr << "..." << endl;

		if (!LocalizationManager::Instance().SetLanguage(langCode)) {
			cout << "\n[ERROR] Failed to load language / 加载语言失败: " << langStr << endl;
			cout << "Please ensure the language file exists at: langs\\" << langStr << ".json" << endl;
			cout << "请确保语言文件存在于: langs\\" << langStr << ".json\n" << endl;
			return;
		}

		cout << "\n=== Language Changed Successfully / 语言切换成功 ===" << endl;
		wcout << L"Current language / 当前语言: " << LocalizationManager::Instance().GetCurrentLanguage() << endl;
		cout << "\nNote: Some messages may still appear in the previous language until you run new commands." << endl;
		cout << "注意: 某些消息可能仍然以先前的语言显示，直到您运行新命令。\n" << endl;

	} catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	} catch (...) {
		cout << "[ERROR] Unknown exception in SetLanguageCommand" << endl;
	}
}

BOOL SetLanguageCommand::HasArgs()
{
	return FlagHasArgs;
}

BOOL SetLanguageCommand::CheckName(string input)
{
	if (input.compare(name) == 0) {
		return TRUE;
	}
	return FALSE;
}
