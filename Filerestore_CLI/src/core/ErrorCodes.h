#pragma once
#include <string>
#include <cstdint>

namespace FR {

// 错误代码枚举 - 使用分类编号
enum class ErrorCode : uint32_t {
    Success = 0,

    // 系统错误 (0x0100xxxx)
    SystemInsufficientPrivileges = 0x01000001,
    SystemVolumeNotFound = 0x01000002,
    SystemDiskAccessDenied = 0x01000003,
    SystemInvalidDriveLetter = 0x01000004,

    // 文件系统错误 (0x0200xxxx)
    FSMFTCorrupted = 0x02000001,
    FSRecordNotFound = 0x02000002,
    FSInvalidClusterNumber = 0x02000003,
    FSAttributeNotFound = 0x02000004,
    FSInvalidRecordNumber = 0x02000005,

    // I/O 错误 (0x0500xxxx)
    IOReadFailed = 0x05000001,
    IOWriteFailed = 0x05000002,
    IOSeekFailed = 0x05000003,
    IOHandleInvalid = 0x05000004,

    // 逻辑错误 (0x0400xxxx)
    LogicInvalidArgument = 0x04000001,
    LogicOperationCancelled = 0x04000002,
    LogicBufferTooSmall = 0x04000003,
    LogicOutOfRange = 0x04000004,

    // 内存错误 (0x0600xxxx)
    MemoryAllocationFailed = 0x06000001,
    MemoryCorrupted = 0x06000002,

    // 文件恢复特定错误 (0x0700xxxx)
    RecoveryFileOverwritten = 0x07000001,
    RecoveryFileFragmented = 0x07000002,
    RecoverySignatureNotFound = 0x07000003,
    RecoveryIntegrityCheckFailed = 0x07000004,
};

// 错误信息类
class ErrorInfo {
public:
    ErrorCode code;
    std::string message;
    std::string context;
    uint32_t systemErrorCode;  // Windows GetLastError() 或其他系统错误码

    // 默认构造函数 - 成功状态
    ErrorInfo()
        : code(ErrorCode::Success)
        , message("")
        , context("")
        , systemErrorCode(0)
    {}

    // 完整构造函数
    ErrorInfo(ErrorCode ec,
              const std::string& msg = "",
              const std::string& ctx = "",
              uint32_t sysErr = 0)
        : code(ec)
        , message(msg)
        , context(ctx)
        , systemErrorCode(sysErr)
    {}

    // 判断是否成功
    bool IsSuccess() const { return code == ErrorCode::Success; }

    // 转换为字符串描述
    std::string ToString() const;

    // 获取错误代码的名称
    std::string GetErrorCodeName() const;

    // 获取错误类别
    std::string GetCategory() const;
};

// 辅助函数：从 Windows GetLastError() 创建 ErrorInfo
ErrorInfo MakeSystemError(ErrorCode code, const std::string& message, const std::string& context);

// 辅助函数：创建简单错误
ErrorInfo MakeError(ErrorCode code, const std::string& message);

} // namespace FR
