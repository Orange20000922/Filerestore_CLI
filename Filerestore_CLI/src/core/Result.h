#pragma once
#include "ErrorCodes.h"
#include <optional>
#include <utility>
#include <stdexcept>
#include <functional>

namespace FR {

// Result<T> - 表示操作结果的类型，要么成功返回值 T，要么失败返回错误信息
template<typename T>
class Result {
private:
    std::optional<T> value_;
    ErrorInfo error_;

public:
    // 私有构造函数 - 使用静态工厂方法创建
    Result() = default;

    // 静态工厂方法：创建成功结果
    static Result<T> Success(T val) {
        Result<T> r;
        r.value_ = std::move(val);
        r.error_ = ErrorInfo();  // Success
        return r;
    }

    // 静态工厂方法：创建失败结果
    static Result<T> Failure(ErrorInfo err) {
        Result<T> r;
        r.error_ = err;
        return r;
    }

    // 静态工厂方法：创建失败结果（简化版）
    static Result<T> Failure(ErrorCode code, const std::string& message = "") {
        return Failure(MakeError(code, message));
    }

    // 判断操作是否成功
    bool IsSuccess() const { return error_.IsSuccess() && value_.has_value(); }

    // 判断操作是否失败
    bool IsFailure() const { return !IsSuccess(); }

    // 获取成功值（如果失败则抛出异常 - 慎用）
    T& Value() {
        if (!value_.has_value()) {
            throw std::runtime_error("Attempting to get value from failed Result: " + error_.ToString());
        }
        return value_.value();
    }

    const T& Value() const {
        if (!value_.has_value()) {
            throw std::runtime_error("Attempting to get value from failed Result: " + error_.ToString());
        }
        return value_.value();
    }

    // 获取错误信息
    const ErrorInfo& Error() const { return error_; }

    // 安全获取值：如果失败则返回默认值
    T ValueOr(T defaultValue) const {
        return value_.value_or(std::move(defaultValue));
    }

    // 安全获取值：如果失败则返回 nullptr（仅适用于指针类型）
    template<typename U = T>
    typename std::enable_if<std::is_pointer<U>::value, U>::type
    ValueOrNull() const {
        return value_.value_or(nullptr);
    }

    // 链式操作：如果成功则执行函数 f，否则传递错误
    template<typename F>
    auto Then(F f) -> Result<decltype(f(std::declval<T>()))> {
        using RetType = decltype(f(std::declval<T>()));

        if (IsFailure()) {
            return Result<RetType>::Failure(error_);
        }

        return Result<RetType>::Success(f(value_.value()));
    }

    // 链式操作：如果失败则执行错误处理函数
    Result<T>& OnError(std::function<void(const ErrorInfo&)> errorHandler) {
        if (IsFailure()) {
            errorHandler(error_);
        }
        return *this;
    }

    // 操作符重载：布尔转换（允许 if (result) 语法）
    explicit operator bool() const { return IsSuccess(); }
};

// Result<void> 特化 - 用于不返回值的操作
template<>
class Result<void> {
private:
    ErrorInfo error_;

public:
    Result() = default;

    // 静态工厂方法：创建成功结果
    static Result<void> Success() {
        Result<void> r;
        r.error_ = ErrorInfo();  // Success
        return r;
    }

    // 静态工厂方法：创建失败结果
    static Result<void> Failure(ErrorInfo err) {
        Result<void> r;
        r.error_ = err;
        return r;
    }

    // 静态工厂方法：创建失败结果（简化版）
    static Result<void> Failure(ErrorCode code, const std::string& message = "") {
        return Failure(MakeError(code, message));
    }

    // 判断操作是否成功
    bool IsSuccess() const { return error_.IsSuccess(); }

    // 判断操作是否失败
    bool IsFailure() const { return !IsSuccess(); }

    // 获取错误信息
    const ErrorInfo& Error() const { return error_; }

    // 链式操作：如果失败则执行错误处理函数
    Result<void>& OnError(std::function<void(const ErrorInfo&)> errorHandler) {
        if (IsFailure()) {
            errorHandler(error_);
        }
        return *this;
    }

    // 操作符重载：布尔转换（允许 if (result) 语法）
    explicit operator bool() const { return IsSuccess(); }
};

} // namespace FR
