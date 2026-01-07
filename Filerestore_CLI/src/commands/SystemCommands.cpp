// SystemCommands.cpp - 系统相关命令实现
// 包含: PrintAllFunction, PrintAllCommand, HelpCommand, ExitCommand, SetLanguageCommand

#include "cmd.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <Windows.h>
#include <map>
#include "cli.h"
#include "ImageTable.h"
#include "LocalizationManager.h"

using namespace std;

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

		cout << "=== ML 数据集生成命令 ===" << endl;
		cout << "  mlscan            - 扫描目录/卷生成ML训练数据集" << endl;
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
	else if (cmdName == "mlscan") {
		cout << "\n=== mlscan - ML数据集生成器 ===\n" << endl;
		cout << "用法: mlscan <path/drive> [options]\n" << endl;
		cout << "参数:" << endl;
		cout << "  <path/drive>     - 扫描路径或驱动器字母 (如: D: 或 C:\\Users\\Documents)\n" << endl;
		cout << "选项:" << endl;
		cout << "  --types=t1,t2,...  指定扫描的文件类型 (默认: 全部支持类型)" << endl;
		cout << "  --max=N            每类最大样本数 (默认: 2000)" << endl;
		cout << "  --output=file      输出文件名 (默认: ml_dataset.csv)" << endl;
		cout << "  --binary           导出为二进制格式 (更高效)" << endl;
		cout << "  --threads=N        工作线程数 (默认: 8)" << endl;
		cout << "  --include-path     在输出中包含文件路径\n" << endl;
		cout << "示例:" << endl;
		cout << "  mlscan D:                                    - 扫描D盘全部类型" << endl;
		cout << "  mlscan D: --types=pdf,jpg,png --max=1000    - 指定类型和数量" << endl;
		cout << "  mlscan C:\\Documents --output=docs.csv       - 扫描指定目录" << endl;
		cout << "  mlscan D: --binary --output=dataset.bin     - 导出二进制格式\n" << endl;
		cout << "支持的文件类型:" << endl;
		cout << "  pdf, doc, xls, ppt, html, txt, xml, jpg, gif, png\n" << endl;
		cout << "工作流程:" << endl;
		cout << "  1. 运行 mlscan 生成数据集 CSV/BIN 文件" << endl;
		cout << "  2. 将文件复制到 ml/data/ 目录" << endl;
		cout << "  3. 运行: python ml/src/train.py --csv <file.csv>" << endl;
		cout << "  4. 训练完成后将 ONNX 模型复制到程序目录\n" << endl;
		cout << "性能特性:" << endl;
		cout << "  - 多线程并行特征提取" << endl;
		cout << "  - 与 Python 端特征提取算法完全一致" << endl;
		cout << "  - 261维特征: 256字节频率 + 5统计特征" << endl;
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
