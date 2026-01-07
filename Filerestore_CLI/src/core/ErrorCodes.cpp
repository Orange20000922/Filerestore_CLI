#include "ErrorCodes.h"
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace FR {

// 获取错误代码的名称
std::string ErrorInfo::GetErrorCodeName() const {
    switch (code) {
        // 成功
        case ErrorCode::Success:
            return "Success";

        // 系统错误
        case ErrorCode::SystemInsufficientPrivileges:
            return "SystemInsufficientPrivileges";
        case ErrorCode::SystemVolumeNotFound:
            return "SystemVolumeNotFound";
        case ErrorCode::SystemDiskAccessDenied:
            return "SystemDiskAccessDenied";
        case ErrorCode::SystemInvalidDriveLetter:
            return "SystemInvalidDriveLetter";

        // 文件系统错误
        case ErrorCode::FSMFTCorrupted:
            return "FSMFTCorrupted";
        case ErrorCode::FSRecordNotFound:
            return "FSRecordNotFound";
        case ErrorCode::FSInvalidClusterNumber:
            return "FSInvalidClusterNumber";
        case ErrorCode::FSAttributeNotFound:
            return "FSAttributeNotFound";
        case ErrorCode::FSInvalidRecordNumber:
            return "FSInvalidRecordNumber";

        // I/O 错误
        case ErrorCode::IOReadFailed:
            return "IOReadFailed";
        case ErrorCode::IOWriteFailed:
            return "IOWriteFailed";
        case ErrorCode::IOSeekFailed:
            return "IOSeekFailed";
        case ErrorCode::IOHandleInvalid:
            return "IOHandleInvalid";

        // 逻辑错误
        case ErrorCode::LogicInvalidArgument:
            return "LogicInvalidArgument";
        case ErrorCode::LogicOperationCancelled:
            return "LogicOperationCancelled";
        case ErrorCode::LogicBufferTooSmall:
            return "LogicBufferTooSmall";
        case ErrorCode::LogicOutOfRange:
            return "LogicOutOfRange";

        // 内存错误
        case ErrorCode::MemoryAllocationFailed:
            return "MemoryAllocationFailed";
        case ErrorCode::MemoryCorrupted:
            return "MemoryCorrupted";

        // 文件恢复特定错误
        case ErrorCode::RecoveryFileOverwritten:
            return "RecoveryFileOverwritten";
        case ErrorCode::RecoveryFileFragmented:
            return "RecoveryFileFragmented";
        case ErrorCode::RecoverySignatureNotFound:
            return "RecoverySignatureNotFound";
        case ErrorCode::RecoveryIntegrityCheckFailed:
            return "RecoveryIntegrityCheckFailed";

        default:
            return "UnknownError";
    }
}

// 获取错误类别
std::string ErrorInfo::GetCategory() const {
    uint32_t category = (static_cast<uint32_t>(code) >> 16) & 0xFF;

    switch (category) {
        case 0x01:
            return "System";
        case 0x02:
            return "FileSystem";
        case 0x04:
            return "Logic";
        case 0x05:
            return "IO";
        case 0x06:
            return "Memory";
        case 0x07:
            return "Recovery";
        default:
            return "Unknown";
    }
}

// 转换为字符串描述
std::string ErrorInfo::ToString() const {
    if (IsSuccess()) {
        return "Success";
    }

    std::ostringstream oss;
    oss << "[" << GetCategory() << "] " << GetErrorCodeName();

    if (!message.empty()) {
        oss << ": " << message;
    }

    if (!context.empty()) {
        oss << " (Context: " << context << ")";
    }

    if (systemErrorCode != 0) {
        oss << " [System Error: 0x" << std::hex << std::setw(8)
            << std::setfill('0') << systemErrorCode << "]";

#ifdef _WIN32
        // 在 Windows 上，尝试获取系统错误消息
        LPSTR messageBuffer = nullptr;
        size_t size = FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            systemErrorCode,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPSTR)&messageBuffer,
            0,
            NULL);

        if (size > 0 && messageBuffer != nullptr) {
            std::string sysMsg(messageBuffer, size);
            // 移除末尾的换行符
            while (!sysMsg.empty() && (sysMsg.back() == '\n' || sysMsg.back() == '\r')) {
                sysMsg.pop_back();
            }
            oss << " - " << sysMsg;
            LocalFree(messageBuffer);
        }
#endif
    }

    return oss.str();
}

// 从 Windows GetLastError() 创建 ErrorInfo
ErrorInfo MakeSystemError(ErrorCode code, const std::string& message, const std::string& context) {
#ifdef _WIN32
    DWORD lastError = GetLastError();
    return ErrorInfo(code, message, context, lastError);
#else
    return ErrorInfo(code, message, context, 0);
#endif
}

// 创建简单错误
ErrorInfo MakeError(ErrorCode code, const std::string& message) {
    return ErrorInfo(code, message, "", 0);
}

} // namespace FR
