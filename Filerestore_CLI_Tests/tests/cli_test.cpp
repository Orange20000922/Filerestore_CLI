#include <gtest/gtest.h>
#include "../../Filerestore_CLI/src/core/cli.h"
#include "../../Filerestore_CLI/src/commands/CommandMacros.h"
#include <string>
#include <vector>

// ============================================================================
// CLI 参数解析测试
// ============================================================================

class CLITest : public ::testing::Test {
protected:
    CLI cli;

    void SetUp() override {
        // 每个测试前初始化 CLI
        CLI::shouldExit = false;
    }

    void TearDown() override {
        // 清理
        CLI::shouldExit = false;
    }
};

// ============================================================================
// 基础命令测试
// ============================================================================

TEST_F(CLITest, HelpCommand) {
    // 测试帮助命令不会崩溃
    EXPECT_NO_THROW(cli.Run("help"));
}

TEST_F(CLITest, ExitCommand) {
    // 测试退出命令设置退出标志
    cli.Run("exit");
    EXPECT_TRUE(CLI::ShouldExit());
}

TEST_F(CLITest, InvalidCommand) {
    // 测试无效命令不会崩溃
    EXPECT_NO_THROW(cli.Run("nonexistent_command_xyz"));
}

TEST_F(CLITest, EmptyCommand) {
    // 测试空命令不会崩溃
    EXPECT_NO_THROW(cli.Run(""));
}

TEST_F(CLITest, CommandWithExtraSpaces) {
    // 测试带多余空格的命令
    EXPECT_NO_THROW(cli.Run("  help  "));
    EXPECT_NO_THROW(cli.Run("help   "));
}

// ============================================================================
// 参数验证测试
// ============================================================================

TEST_F(CLITest, CommandWithMissingParameters) {
    // 测试缺少必填参数的命令
    EXPECT_NO_THROW(cli.Run("listdeleted"));  // 缺少 drive
    EXPECT_NO_THROW(cli.Run("carvepool"));    // 缺少所有参数
    EXPECT_NO_THROW(cli.Run("recover"));      // 缺少 drive
}

TEST_F(CLITest, CommandWithValidParameters) {
    // 测试带有效参数的命令（不会实际执行，只测试解析）
    EXPECT_NO_THROW(cli.Run("help listdeleted"));
    EXPECT_NO_THROW(cli.Run("help recover"));
}

TEST_F(CLITest, DriveParameterFormats) {
    // 测试不同的驱动器参数格式
    // 注意：这些会尝试解析，但因为可能没有权限，主要测试不崩溃
    EXPECT_NO_THROW(cli.Run("checkdrive D:"));
    EXPECT_NO_THROW(cli.Run("checkdrive D"));
    EXPECT_NO_THROW(cli.Run("checkdrive d:"));  // 小写
}

TEST_F(CLITest, InvalidDriveParameter) {
    // 测试无效驱动器参数
    EXPECT_NO_THROW(cli.Run("checkdrive XYZ:"));
    EXPECT_NO_THROW(cli.Run("checkdrive 123"));
    EXPECT_NO_THROW(cli.Run("checkdrive "));
}

// ============================================================================
// 命令匹配测试
// ============================================================================

TEST_F(CLITest, CommandPrefixMatching) {
    // CLI 使用迭代筛选算法，测试前缀匹配
    EXPECT_NO_THROW(cli.Run("carve"));      // 可能匹配多个
    EXPECT_NO_THROW(cli.Run("carvelist"));  // 精确匹配
}

TEST_F(CLITest, CaseSensitivity) {
    // 测试命令是否大小写敏感（应该不敏感）
    EXPECT_NO_THROW(cli.Run("HELP"));
    EXPECT_NO_THROW(cli.Run("Help"));
    EXPECT_NO_THROW(cli.Run("hElP"));
}

// ============================================================================
// 复杂命令测试
// ============================================================================

TEST_F(CLITest, MultiParameterCommand) {
    // 测试多参数命令解析
    EXPECT_NO_THROW(cli.Run("carvepool D: jpg D:\\out 8"));
    EXPECT_NO_THROW(cli.Run("recover D: test.txt D:\\out"));
    EXPECT_NO_THROW(cli.Run("restorebyrecord D: 12345 D:\\out"));
}

TEST_F(CLITest, PathWithSpaces) {
    // 测试带空格的路径参数
    EXPECT_NO_THROW(cli.Run("recover D: \"my document.txt\" \"D:\\My Output\""));
}

TEST_F(CLITest, SpecialCharactersInParameters) {
    // 测试特殊字符
    EXPECT_NO_THROW(cli.Run("searchdeleted D: *.txt"));
    EXPECT_NO_THROW(cli.Run("searchdeleted D: test?.doc"));
}

// ============================================================================
// CommandHelper 测试
// ============================================================================

#include "../../Filerestore_CLI/src/tui/CommandHelper.h"

TEST(CommandHelperTest, GetAllCommandNames) {
    auto names = CommandHelper::GetAllCommandNames();

    // 至少应该有一些基础命令
    EXPECT_GT(names.size(), 0);

    // 检查关键命令是否存在
    bool hasHelp = std::find(names.begin(), names.end(), "help") != names.end();
    bool hasRecover = std::find(names.begin(), names.end(), "recover") != names.end();

    EXPECT_TRUE(hasHelp);
    EXPECT_TRUE(hasRecover);
}

TEST(CommandHelperTest, GetCommandDescription) {
    // 测试获取命令描述
    auto desc = CommandHelper::GetDescription("recover");
    EXPECT_FALSE(desc.empty());
}

TEST(CommandHelperTest, GetCommandUsage) {
    // 测试获取命令用法
    auto usage = CommandHelper::GetUsage("carvepool");
    EXPECT_FALSE(usage.empty());
    EXPECT_NE(usage.find("carvepool"), std::string::npos);
}

TEST(CommandHelperTest, MatchCommandsPrefix) {
    // 测试前缀匹配
    auto matches = CommandHelper::MatchCommands("carve");
    EXPECT_GE(matches.size(), 1);

    // 所有匹配项应该以 "carve" 开头
    for (const auto& match : matches) {
        EXPECT_EQ(match.find("carve"), 0);
    }
}

TEST(CommandHelperTest, MatchCommandsEmpty) {
    // 空前缀应该返回所有命令
    auto all = CommandHelper::MatchCommands("");
    EXPECT_GT(all.size(), 10);
}

TEST(CommandHelperTest, MatchCommandsNoMatch) {
    // 不存在的前缀
    auto matches = CommandHelper::MatchCommands("zzz_nonexistent");
    EXPECT_EQ(matches.size(), 0);
}

TEST(CommandHelperTest, GetParams) {
    // 测试获取命令参数
    auto params = CommandHelper::GetParams("recover");
    EXPECT_GE(params.size(), 1);

    // 第一个参数应该是 drive
    EXPECT_EQ(params[0].name, "drive");
    EXPECT_TRUE(params[0].required);
}

TEST(CommandHelperTest, AssembleCommand) {
    // 测试命令组装
    std::string params[] = {"D:", "test.txt", "D:\\out"};
    auto assembled = CommandHelper::AssembleCommand("recover", params, 3);

    EXPECT_EQ(assembled, "recover D: test.txt D:\\out");
}

// ============================================================================
// 边界条件测试
// ============================================================================

TEST_F(CLITest, VeryLongCommand) {
    // 测试非常长的命令
    std::string longCmd = "help " + std::string(10000, 'a');
    EXPECT_NO_THROW(cli.Run(longCmd));
}

TEST_F(CLITest, ManyParameters) {
    // 测试大量参数
    std::string manyParams = "help";
    for (int i = 0; i < 100; i++) {
        manyParams += " param" + std::to_string(i);
    }
    EXPECT_NO_THROW(cli.Run(manyParams));
}

TEST_F(CLITest, UnicodeCharacters) {
    // 测试 Unicode 字符（文件名可能包含中文）
    EXPECT_NO_THROW(cli.Run("searchdeleted D: 测试文件.txt"));
    EXPECT_NO_THROW(cli.Run("recover D: 我的文档.docx D:\\输出"));
}
