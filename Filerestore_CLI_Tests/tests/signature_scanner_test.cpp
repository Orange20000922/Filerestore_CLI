#include <gtest/gtest.h>
#include "../../Filerestore_CLI/src/fileRestore/SignatureScanThreadPool.h"
#include <vector>
#include <cstring>

// ============================================================================
// 签名匹配测试（SIMD 优化验证）
// ============================================================================

class SignatureScannerTest : public ::testing::Test {
protected:
    SignatureScanThreadPool* scanner;

    void SetUp() override {
        // 创建扫描器实例（用于访问 MatchSignature 方法）
        scanner = new SignatureScanThreadPool(8);  // 8 threads
    }

    void TearDown() override {
        delete scanner;
    }

    // 辅助：创建测试数据
    std::vector<BYTE> CreateTestData(const std::vector<BYTE>& prefix, size_t totalSize) {
        std::vector<BYTE> data(totalSize, 0xCC);  // 填充 0xCC
        std::copy(prefix.begin(), prefix.end(), data.begin());
        return data;
    }
};

// ============================================================================
// 基础签名匹配测试
// ============================================================================

TEST_F(SignatureScannerTest, MatchZipSignature) {
    // ZIP 签名: 50 4B 03 04
    std::vector<BYTE> zipSig = {0x50, 0x4B, 0x03, 0x04};
    auto data = CreateTestData(zipSig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), zipSig));
}

TEST_F(SignatureScannerTest, MatchPdfSignature) {
    // PDF 签名: 25 50 44 46 (%PDF)
    std::vector<BYTE> pdfSig = {0x25, 0x50, 0x44, 0x46};
    auto data = CreateTestData(pdfSig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), pdfSig));
}

TEST_F(SignatureScannerTest, MatchJpgSignature) {
    // JPG 签名: FF D8 FF
    std::vector<BYTE> jpgSig = {0xFF, 0xD8, 0xFF};
    auto data = CreateTestData(jpgSig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), jpgSig));
}

TEST_F(SignatureScannerTest, MatchPngSignature) {
    // PNG 签名: 89 50 4E 47 0D 0A 1A 0A (8 字节)
    std::vector<BYTE> pngSig = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    auto data = CreateTestData(pngSig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), pngSig));
}

TEST_F(SignatureScannerTest, MatchGifSignature) {
    // GIF 签名: 47 49 46 38 (GIF8)
    std::vector<BYTE> gifSig = {0x47, 0x49, 0x46, 0x38};
    auto data = CreateTestData(gifSig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), gifSig));
}

TEST_F(SignatureScannerTest, MatchRarSignature) {
    // RAR 签名: 52 61 72 21 1A 07 (6 字节)
    std::vector<BYTE> rarSig = {0x52, 0x61, 0x72, 0x21, 0x1A, 0x07};
    auto data = CreateTestData(rarSig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), rarSig));
}

TEST_F(SignatureScannerTest, Match7zSignature) {
    // 7z 签名: 37 7A BC AF 27 1C (6 字节)
    std::vector<BYTE> sig7z = {0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C};
    auto data = CreateTestData(sig7z, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), sig7z));
}

// ============================================================================
// 不匹配测试
// ============================================================================

TEST_F(SignatureScannerTest, NoMatchWrongSignature) {
    // 期望 ZIP，但提供 PDF 数据
    std::vector<BYTE> zipSig = {0x50, 0x4B, 0x03, 0x04};
    std::vector<BYTE> pdfData = {0x25, 0x50, 0x44, 0x46, 0x00, 0x00};

    EXPECT_FALSE(scanner->MatchSignature(pdfData.data(), pdfData.size(), zipSig));
}

TEST_F(SignatureScannerTest, NoMatchPartialSignature) {
    // 签名的前半部分匹配，但后半部分不匹配
    std::vector<BYTE> zipSig = {0x50, 0x4B, 0x03, 0x04};
    std::vector<BYTE> wrongData = {0x50, 0x4B, 0xFF, 0xFF};

    EXPECT_FALSE(scanner->MatchSignature(wrongData.data(), wrongData.size(), zipSig));
}

// ============================================================================
// 边界条件测试
// ============================================================================

TEST_F(SignatureScannerTest, ExactSizeMatch) {
    // 数据大小刚好等于签名大小
    std::vector<BYTE> sig = {0x50, 0x4B, 0x03, 0x04};
    std::vector<BYTE> data = {0x50, 0x4B, 0x03, 0x04};

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), sig));
}

TEST_F(SignatureScannerTest, DataSmallerThanSignature) {
    // 数据小于签名 - 应该返回 false
    std::vector<BYTE> sig = {0x50, 0x4B, 0x03, 0x04};
    std::vector<BYTE> data = {0x50, 0x4B};  // 只有 2 字节

    EXPECT_FALSE(scanner->MatchSignature(data.data(), data.size(), sig));
}

TEST_F(SignatureScannerTest, EmptySignature) {
    // 空签名 - 应该匹配任何数据（根据实现逻辑）
    std::vector<BYTE> emptySig;
    std::vector<BYTE> data = {0x50, 0x4B, 0x03, 0x04};

    // 注意：实际行为取决于实现，memcmp(p1, p2, 0) 返回 0 (相等)
    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), emptySig));
}

TEST_F(SignatureScannerTest, VeryShortSignature) {
    // 极短签名（1 字节）- 应该使用标量路径
    std::vector<BYTE> shortSig = {0xFF};
    std::vector<BYTE> data = {0xFF, 0x00, 0x00};

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), shortSig));
}

TEST_F(SignatureScannerTest, TwoByteSignature) {
    // 2 字节签名（如 BMP: 42 4D）
    std::vector<BYTE> bmpSig = {0x42, 0x4D};
    auto data = CreateTestData(bmpSig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), bmpSig));
}

TEST_F(SignatureScannerTest, SignatureAt16ByteBoundary) {
    // 测试 16 字节边界的签名（SSE2 优化边界）
    std::vector<BYTE> sig(16, 0xAA);
    auto data = CreateTestData(sig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), sig));
}

// ============================================================================
// SIMD 优化验证（标量 vs SIMD 等价性）
// ============================================================================

TEST_F(SignatureScannerTest, SimdEquivalenceShort) {
    // 4 字节签名：应该触发 SIMD 路径（如果支持 SSE2）
    std::vector<BYTE> sig = {0x12, 0x34, 0x56, 0x78};
    auto matchData = CreateTestData(sig, 1024);
    auto noMatchData = CreateTestData({0xFF, 0xFF, 0xFF, 0xFF}, 1024);

    // 验证匹配和不匹配都正确
    EXPECT_TRUE(scanner->MatchSignature(matchData.data(), matchData.size(), sig));
    EXPECT_FALSE(scanner->MatchSignature(noMatchData.data(), noMatchData.size(), sig));
}

TEST_F(SignatureScannerTest, SimdEquivalenceMedium) {
    // 8 字节签名（PNG 长度）
    std::vector<BYTE> sig = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    auto matchData = CreateTestData(sig, 1024);
    auto noMatchData = CreateTestData({0x01, 0x02, 0x03, 0x04, 0xFF, 0xFF, 0xFF, 0xFF}, 1024);

    EXPECT_TRUE(scanner->MatchSignature(matchData.data(), matchData.size(), sig));
    EXPECT_FALSE(scanner->MatchSignature(noMatchData.data(), noMatchData.size(), sig));
}

TEST_F(SignatureScannerTest, SimdEquivalenceLong) {
    // 12 字节签名（超过大多数实际签名）
    std::vector<BYTE> sig = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    auto matchData = CreateTestData(sig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(matchData.data(), matchData.size(), sig));
}

// ============================================================================
// 性能基准测试（可选，需要大数据集）
// ============================================================================

TEST_F(SignatureScannerTest, DISABLED_PerformanceBenchmark) {
    // 测试大量匹配操作的性能（禁用，仅手动运行）
    std::vector<BYTE> sig = {0x50, 0x4B, 0x03, 0x04};
    auto data = CreateTestData(sig, 1024 * 1024);  // 1 MB

    for (int i = 0; i < 10000; i++) {
        scanner->MatchSignature(data.data(), data.size(), sig);
    }
    // 运行后检查日志或使用性能分析工具
}

// ============================================================================
// 特殊字节模式测试
// ============================================================================

TEST_F(SignatureScannerTest, AllZeros) {
    std::vector<BYTE> sig(8, 0x00);
    auto data = CreateTestData(sig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), sig));
}

TEST_F(SignatureScannerTest, AllOnes) {
    std::vector<BYTE> sig(8, 0xFF);
    auto data = CreateTestData(sig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), sig));
}

TEST_F(SignatureScannerTest, AlternatingPattern) {
    std::vector<BYTE> sig = {0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55};
    auto data = CreateTestData(sig, 1024);

    EXPECT_TRUE(scanner->MatchSignature(data.data(), data.size(), sig));
}

// ============================================================================
// 内存对齐测试（非对齐访问）
// ============================================================================

TEST_F(SignatureScannerTest, UnalignedAccess) {
    // 创建故意不对齐的数据（偏移 1 字节）
    std::vector<BYTE> buffer(1024 + 1);
    std::vector<BYTE> sig = {0x12, 0x34, 0x56, 0x78};

    // 从偏移 1 开始写入签名（非 16 字节对齐）
    std::copy(sig.begin(), sig.end(), buffer.begin() + 1);

    EXPECT_TRUE(scanner->MatchSignature(buffer.data() + 1, buffer.size() - 1, sig));
}

TEST_F(SignatureScannerTest, UnalignedAccessLong) {
    // 测试非对齐的长签名
    std::vector<BYTE> buffer(1024 + 3);
    std::vector<BYTE> sig = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};

    // 从偏移 3 开始（非对齐）
    std::copy(sig.begin(), sig.end(), buffer.begin() + 3);

    EXPECT_TRUE(scanner->MatchSignature(buffer.data() + 3, buffer.size() - 3, sig));
}
