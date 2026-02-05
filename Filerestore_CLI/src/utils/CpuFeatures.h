#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <intrin.h>

// ============================================================================
// CPU 特性检测工具
// 用于运行时检测 CPU 支持的 SIMD 指令集
// ============================================================================

class CpuFeatures {
private:
    // CPU 特性标志
    bool hasSSE2_ = false;
    bool hasSSE41_ = false;
    bool hasSSE42_ = false;
    bool hasAVX_ = false;
    bool hasAVX2_ = false;
    bool hasAVX512F_ = false;
    bool hasAVX512BW_ = false;
    bool hasPopcnt_ = false;
    bool hasBMI1_ = false;
    bool hasBMI2_ = false;

    // CPU 信息
    std::string cpuVendor_;
    std::string cpuBrand_;
    int family_ = 0;
    int model_ = 0;
    int stepping_ = 0;

    // 单例实例
    static CpuFeatures* instance_;

    // 私有构造函数
    CpuFeatures() {
        Detect();
    }

    // 执行 CPUID 指令
    void Cpuid(int info[4], int funcId, int subFuncId = 0) {
        __cpuidex(info, funcId, subFuncId);
    }

    // 检测 CPU 特性
    void Detect() {
        int info[4] = { 0 };

        // 获取最大支持的 CPUID 功能号
        Cpuid(info, 0);
        int maxFunc = info[0];

        // 获取 CPU 厂商字符串
        char vendor[13] = { 0 };
        *reinterpret_cast<int*>(vendor) = info[1];
        *reinterpret_cast<int*>(vendor + 4) = info[3];
        *reinterpret_cast<int*>(vendor + 8) = info[2];
        cpuVendor_ = vendor;

        // 获取 CPU 特性 (Function 1)
        if (maxFunc >= 1) {
            Cpuid(info, 1);

            family_ = ((info[0] >> 8) & 0xF) + ((info[0] >> 20) & 0xFF);
            model_ = ((info[0] >> 4) & 0xF) | (((info[0] >> 16) & 0xF) << 4);
            stepping_ = info[0] & 0xF;

            // ECX 特性
            hasSSE41_ = (info[2] & (1 << 19)) != 0;
            hasSSE42_ = (info[2] & (1 << 20)) != 0;
            hasPopcnt_ = (info[2] & (1 << 23)) != 0;
            hasAVX_ = (info[2] & (1 << 28)) != 0;

            // EDX 特性
            hasSSE2_ = (info[3] & (1 << 26)) != 0;
        }

        // 获取扩展特性 (Function 7)
        if (maxFunc >= 7) {
            Cpuid(info, 7, 0);

            // EBX 特性
            hasBMI1_ = (info[1] & (1 << 3)) != 0;
            hasAVX2_ = (info[1] & (1 << 5)) != 0;
            hasBMI2_ = (info[1] & (1 << 8)) != 0;
            hasAVX512F_ = (info[1] & (1 << 16)) != 0;
            hasAVX512BW_ = (info[1] & (1 << 30)) != 0;
        }

        // 获取扩展 CPUID 信息
        Cpuid(info, 0x80000000);
        int maxExtFunc = info[0];

        // 获取 CPU 品牌字符串
        if (maxExtFunc >= 0x80000004) {
            char brand[49] = { 0 };
            Cpuid(reinterpret_cast<int*>(brand), 0x80000002);
            Cpuid(reinterpret_cast<int*>(brand + 16), 0x80000003);
            Cpuid(reinterpret_cast<int*>(brand + 32), 0x80000004);
            cpuBrand_ = brand;

            // 移除前导空格
            size_t start = cpuBrand_.find_first_not_of(' ');
            if (start != std::string::npos) {
                cpuBrand_ = cpuBrand_.substr(start);
            }
        }

        // 验证 AVX/AVX2 是否真正可用（需要 OS 支持）
        if (hasAVX_ || hasAVX2_) {
            // 检查 XGETBV 是否可用
            Cpuid(info, 1);
            bool osxsave = (info[2] & (1 << 27)) != 0;

            if (osxsave) {
                // 检查 XCR0 寄存器
                unsigned long long xcr0 = _xgetbv(0);
                bool avxEnabled = (xcr0 & 0x6) == 0x6;  // XMM 和 YMM 状态保存
                bool avx512Enabled = (xcr0 & 0xE6) == 0xE6;  // 包括 ZMM 状态

                if (!avxEnabled) {
                    hasAVX_ = false;
                    hasAVX2_ = false;
                }
                if (!avx512Enabled) {
                    hasAVX512F_ = false;
                    hasAVX512BW_ = false;
                }
            }
            else {
                // OS 不支持 XSAVE，禁用所有 AVX
                hasAVX_ = false;
                hasAVX2_ = false;
                hasAVX512F_ = false;
                hasAVX512BW_ = false;
            }
        }
    }

public:
    // 获取单例实例
    static CpuFeatures& Instance() {
        if (!instance_) {
            instance_ = new CpuFeatures();
        }
        return *instance_;
    }

    // 特性查询
    bool HasSSE2() const { return hasSSE2_; }
    bool HasSSE41() const { return hasSSE41_; }
    bool HasSSE42() const { return hasSSE42_; }
    bool HasAVX() const { return hasAVX_; }
    bool HasAVX2() const { return hasAVX2_; }
    bool HasAVX512F() const { return hasAVX512F_; }
    bool HasAVX512BW() const { return hasAVX512BW_; }
    bool HasPopcnt() const { return hasPopcnt_; }
    bool HasBMI1() const { return hasBMI1_; }
    bool HasBMI2() const { return hasBMI2_; }

    // CPU 信息查询
    const std::string& Vendor() const { return cpuVendor_; }
    const std::string& Brand() const { return cpuBrand_; }
    int Family() const { return family_; }
    int Model() const { return model_; }

    // 获取最佳可用的 SIMD 级别
    enum class SimdLevel {
        SCALAR,     // 无 SIMD
        SSE2,       // 128-bit (16 字节)
        SSE42,      // 128-bit + 字符串指令
        AVX2,       // 256-bit (32 字节)
        AVX512      // 512-bit (64 字节)
    };

    SimdLevel GetBestSimdLevel() const {
        if (hasAVX512F_ && hasAVX512BW_) return SimdLevel::AVX512;
        if (hasAVX2_) return SimdLevel::AVX2;
        if (hasSSE42_) return SimdLevel::SSE42;
        if (hasSSE2_) return SimdLevel::SSE2;
        return SimdLevel::SCALAR;
    }

    // 获取 SIMD 级别的字符串描述
    static std::string SimdLevelToString(SimdLevel level) {
        switch (level) {
        case SimdLevel::SCALAR: return "Scalar (no SIMD)";
        case SimdLevel::SSE2: return "SSE2 (128-bit)";
        case SimdLevel::SSE42: return "SSE4.2 (128-bit + string ops)";
        case SimdLevel::AVX2: return "AVX2 (256-bit)";
        case SimdLevel::AVX512: return "AVX-512 (512-bit)";
        default: return "Unknown";
        }
    }

    // 获取完整的 CPU 信息报告
    std::string GetReport() const {
        std::string report;
        report += "CPU: " + cpuBrand_ + "\n";
        report += "Vendor: " + cpuVendor_ + "\n";
        report += "Family/Model/Stepping: " + std::to_string(family_) + "/" +
            std::to_string(model_) + "/" + std::to_string(stepping_) + "\n";
        report += "\nSIMD Support:\n";
        report += "  SSE2:     " + std::string(hasSSE2_ ? "Yes" : "No") + "\n";
        report += "  SSE4.1:   " + std::string(hasSSE41_ ? "Yes" : "No") + "\n";
        report += "  SSE4.2:   " + std::string(hasSSE42_ ? "Yes" : "No") + "\n";
        report += "  AVX:      " + std::string(hasAVX_ ? "Yes" : "No") + "\n";
        report += "  AVX2:     " + std::string(hasAVX2_ ? "Yes" : "No") + "\n";
        report += "  AVX-512F: " + std::string(hasAVX512F_ ? "Yes" : "No") + "\n";
        report += "  AVX-512BW:" + std::string(hasAVX512BW_ ? "Yes" : "No") + "\n";
        report += "\nBit Manipulation:\n";
        report += "  POPCNT:   " + std::string(hasPopcnt_ ? "Yes" : "No") + "\n";
        report += "  BMI1:     " + std::string(hasBMI1_ ? "Yes" : "No") + "\n";
        report += "  BMI2:     " + std::string(hasBMI2_ ? "Yes" : "No") + "\n";
        report += "\nBest SIMD Level: " + SimdLevelToString(GetBestSimdLevel()) + "\n";
        return report;
    }
};

// 静态成员初始化（需要在 .cpp 文件中定义）
// CpuFeatures* CpuFeatures::instance_ = nullptr;
