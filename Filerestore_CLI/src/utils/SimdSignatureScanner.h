#pragma once
#include <Windows.h>
#include <vector>
#include <set>
#include <unordered_map>
#include <cstdint>
#include "CpuFeatures.h"

// ============================================================================
// SIMD 优化的首字节扫描器
// 使用 AVX2/SSE2 加速多模式首字节匹配
// ============================================================================

class SimdSignatureScanner {
public:
    // 匹配结果：数据中匹配首字节的位置
    struct MatchResult {
        size_t offset;      // 在数据中的偏移
        BYTE matchedByte;   // 匹配的首字节
    };

private:
    // 要搜索的首字节集合
    std::vector<BYTE> targetBytes_;

    // 首字节查找表（256 位 bitmap，每个字节一个 bit）
    // 用于快速判断某个字节是否在目标集合中
    uint64_t byteBitmap_[4] = { 0 };  // 256 bits = 4 * 64 bits

    // 当前使用的 SIMD 级别
    CpuFeatures::SimdLevel simdLevel_;

    // 构建 bitmap
    void BuildBitmap() {
        // 清空
        for (int i = 0; i < 4; i++) byteBitmap_[i] = 0;

        // 设置每个目标字节对应的 bit
        for (BYTE b : targetBytes_) {
            int idx = b / 64;
            int bit = b % 64;
            byteBitmap_[idx] |= (1ULL << bit);
        }
    }

    // 检查字节是否在目标集合中
    inline bool IsTargetByte(BYTE b) const {
        int idx = b / 64;
        int bit = b % 64;
        return (byteBitmap_[idx] & (1ULL << bit)) != 0;
    }

    // ========== 各种实现 ==========

    // 标量实现（回退方案）
    void ScanScalar(const BYTE* data, size_t size, std::vector<MatchResult>& results) const;

    // SSE2 实现（128-bit，一次处理 16 字节）
    void ScanSSE2(const BYTE* data, size_t size, std::vector<MatchResult>& results) const;

    // AVX2 实现（256-bit，一次处理 32 字节）
    void ScanAVX2(const BYTE* data, size_t size, std::vector<MatchResult>& results) const;

    // AVX-512 实现（512-bit，一次处理 64 字节）
    void ScanAVX512(const BYTE* data, size_t size, std::vector<MatchResult>& results) const;

public:
    // 构造函数
    SimdSignatureScanner();

    // 从签名索引初始化（提取所有首字节）
    void Initialize(const std::unordered_map<BYTE, std::vector<const void*>>& signatureIndex);

    // 直接设置目标首字节
    void SetTargetBytes(const std::vector<BYTE>& bytes);
    void SetTargetBytes(const std::set<BYTE>& bytes);

    // 执行扫描 - 返回所有匹配首字节的位置
    // 调用者需要在这些位置进行完整的签名验证
    std::vector<MatchResult> Scan(const BYTE* data, size_t size) const;

    // 带预分配的扫描（减少内存分配）
    void Scan(const BYTE* data, size_t size, std::vector<MatchResult>& results) const;

    // 获取当前 SIMD 级别
    CpuFeatures::SimdLevel GetSimdLevel() const { return simdLevel_; }

    // 获取目标字节数量
    size_t GetTargetByteCount() const { return targetBytes_.size(); }

    // 强制设置 SIMD 级别（用于测试）
    void ForceSimdLevel(CpuFeatures::SimdLevel level) { simdLevel_ = level; }
};
