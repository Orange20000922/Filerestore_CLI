#include "SimdSignatureScanner.h"
#include <immintrin.h>
#include <algorithm>

// ============================================================================
// 构造函数
// ============================================================================
SimdSignatureScanner::SimdSignatureScanner() {
    // 检测 CPU 能力并选择最佳实现
    simdLevel_ = CpuFeatures::Instance().GetBestSimdLevel();
}

// ============================================================================
// 初始化方法
// ============================================================================
void SimdSignatureScanner::Initialize(
    const std::unordered_map<BYTE, std::vector<const void*>>& signatureIndex) {
    targetBytes_.clear();
    for (const auto& pair : signatureIndex) {
        targetBytes_.push_back(pair.first);
    }
    BuildBitmap();
}

void SimdSignatureScanner::SetTargetBytes(const std::vector<BYTE>& bytes) {
    targetBytes_ = bytes;
    BuildBitmap();
}

void SimdSignatureScanner::SetTargetBytes(const std::set<BYTE>& bytes) {
    targetBytes_.assign(bytes.begin(), bytes.end());
    BuildBitmap();
}

// ============================================================================
// 主扫描接口
// ============================================================================
std::vector<SimdSignatureScanner::MatchResult> SimdSignatureScanner::Scan(
    const BYTE* data, size_t size) const {
    std::vector<MatchResult> results;
    results.reserve(size / 1024);  // 预估：每 1KB 约 1 个匹配
    Scan(data, size, results);
    return results;
}

void SimdSignatureScanner::Scan(const BYTE* data, size_t size,
    std::vector<MatchResult>& results) const {
    results.clear();

    if (!data || size == 0 || targetBytes_.empty()) {
        return;
    }

    // 根据 SIMD 级别选择实现
    switch (simdLevel_) {
    case CpuFeatures::SimdLevel::AVX512:
        ScanAVX512(data, size, results);
        break;
    case CpuFeatures::SimdLevel::AVX2:
        ScanAVX2(data, size, results);
        break;
    case CpuFeatures::SimdLevel::SSE42:
    case CpuFeatures::SimdLevel::SSE2:
        ScanSSE2(data, size, results);
        break;
    default:
        ScanScalar(data, size, results);
        break;
    }
}

// ============================================================================
// 标量实现（回退方案）
// ============================================================================
void SimdSignatureScanner::ScanScalar(const BYTE* data, size_t size,
    std::vector<MatchResult>& results) const {
    for (size_t i = 0; i < size; ++i) {
        if (IsTargetByte(data[i])) {
            results.push_back({ i, data[i] });
        }
    }
}

// ============================================================================
// SSE2 实现（128-bit，一次处理 16 字节）
// ============================================================================
void SimdSignatureScanner::ScanSSE2(const BYTE* data, size_t size,
    std::vector<MatchResult>& results) const {

    // 处理对齐的部分
    size_t alignedSize = size & ~15ULL;  // 向下对齐到 16 字节
    size_t i = 0;

    // 为每个目标字节创建 SSE 寄存器
    // 由于目标字节数量有限（通常 < 20），直接遍历
    std::vector<__m128i> targetVecs;
    targetVecs.reserve(targetBytes_.size());
    for (BYTE b : targetBytes_) {
        targetVecs.push_back(_mm_set1_epi8(static_cast<char>(b)));
    }

    // 主循环：每次处理 16 字节
    for (; i < alignedSize; i += 16) {
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));

        // 对每个目标字节进行比较，合并结果
        __m128i anyMatch = _mm_setzero_si128();
        for (const __m128i& target : targetVecs) {
            __m128i cmp = _mm_cmpeq_epi8(chunk, target);
            anyMatch = _mm_or_si128(anyMatch, cmp);
        }

        // 提取匹配掩码
        int mask = _mm_movemask_epi8(anyMatch);
        if (mask != 0) {
            // 有匹配，逐位检查
            while (mask) {
                // 找到最低位的 1
                unsigned long pos;
                _BitScanForward(&pos, mask);

                size_t matchOffset = i + pos;
                results.push_back({ matchOffset, data[matchOffset] });

                // 清除最低位
                mask &= (mask - 1);
            }
        }
    }

    // 处理剩余字节（标量）
    for (; i < size; ++i) {
        if (IsTargetByte(data[i])) {
            results.push_back({ i, data[i] });
        }
    }
}

// ============================================================================
// AVX2 实现（256-bit，一次处理 32 字节）
// ============================================================================
void SimdSignatureScanner::ScanAVX2(const BYTE* data, size_t size,
    std::vector<MatchResult>& results) const {

    size_t alignedSize = size & ~31ULL;  // 向下对齐到 32 字节
    size_t i = 0;

    // 为每个目标字节创建 AVX2 寄存器
    std::vector<__m256i> targetVecs;
    targetVecs.reserve(targetBytes_.size());
    for (BYTE b : targetBytes_) {
        targetVecs.push_back(_mm256_set1_epi8(static_cast<char>(b)));
    }

    // 主循环：每次处理 32 字节
    for (; i < alignedSize; i += 32) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));

        // 对每个目标字节进行比较，合并结果
        __m256i anyMatch = _mm256_setzero_si256();
        for (const __m256i& target : targetVecs) {
            __m256i cmp = _mm256_cmpeq_epi8(chunk, target);
            anyMatch = _mm256_or_si256(anyMatch, cmp);
        }

        // 提取匹配掩码（32 位）
        int mask = _mm256_movemask_epi8(anyMatch);
        if (mask != 0) {
            // 有匹配，逐位检查
            while (mask) {
                unsigned long pos;
                _BitScanForward(&pos, mask);

                size_t matchOffset = i + pos;
                results.push_back({ matchOffset, data[matchOffset] });

                mask &= (mask - 1);
            }
        }
    }

    // 处理剩余字节（使用 SSE2 或标量）
    if (i < size) {
        // 剩余 < 32 字节，用标量处理
        for (; i < size; ++i) {
            if (IsTargetByte(data[i])) {
                results.push_back({ i, data[i] });
            }
        }
    }

    // 清除 AVX 状态，避免性能惩罚
    _mm256_zeroupper();
}

// ============================================================================
// AVX-512 实现（512-bit，一次处理 64 字节）
// ============================================================================
void SimdSignatureScanner::ScanAVX512(const BYTE* data, size_t size,
    std::vector<MatchResult>& results) const {

#ifdef __AVX512F__
    size_t alignedSize = size & ~63ULL;  // 向下对齐到 64 字节
    size_t i = 0;

    // 为每个目标字节创建 AVX-512 寄存器
    std::vector<__m512i> targetVecs;
    targetVecs.reserve(targetBytes_.size());
    for (BYTE b : targetBytes_) {
        targetVecs.push_back(_mm512_set1_epi8(static_cast<char>(b)));
    }

    // 主循环：每次处理 64 字节
    for (; i < alignedSize; i += 64) {
        __m512i chunk = _mm512_loadu_si512(data + i);

        // 对每个目标字节进行比较，合并结果
        __mmask64 anyMask = 0;
        for (const __m512i& target : targetVecs) {
            __mmask64 cmp = _mm512_cmpeq_epi8_mask(chunk, target);
            anyMask |= cmp;
        }

        if (anyMask != 0) {
            // 有匹配，逐位检查
            while (anyMask) {
                unsigned long pos;
                _BitScanForward64(&pos, anyMask);

                size_t matchOffset = i + pos;
                results.push_back({ matchOffset, data[matchOffset] });

                anyMask &= (anyMask - 1);
            }
        }
    }

    // 处理剩余字节
    for (; i < size; ++i) {
        if (IsTargetByte(data[i])) {
            results.push_back({ i, data[i] });
        }
    }
#else
    // 编译器不支持 AVX-512，回退到 AVX2
    ScanAVX2(data, size, results);
#endif
}
