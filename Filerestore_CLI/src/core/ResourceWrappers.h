#pragma once
#include <Windows.h>
#include <utility>

namespace FR {

// RAII Handle 包装器 - 自动管理 Windows HANDLE 生命周期
class ScopedHandle {
private:
    HANDLE handle_;

public:
    // 构造函数 - 接受一个 HANDLE
    explicit ScopedHandle(HANDLE h = INVALID_HANDLE_VALUE)
        : handle_(h)
    {}

    // 析构函数 - 自动关闭 HANDLE
    ~ScopedHandle() {
        Close();
    }

    // 禁止拷贝构造
    ScopedHandle(const ScopedHandle&) = delete;
    ScopedHandle& operator=(const ScopedHandle&) = delete;

    // 支持移动构造
    ScopedHandle(ScopedHandle&& other) noexcept
        : handle_(other.handle_)
    {
        other.handle_ = INVALID_HANDLE_VALUE;
    }

    // 支持移动赋值
    ScopedHandle& operator=(ScopedHandle&& other) noexcept {
        if (this != &other) {
            Close();
            handle_ = other.handle_;
            other.handle_ = INVALID_HANDLE_VALUE;
        }
        return *this;
    }

    // 获取原始 HANDLE
    HANDLE Get() const {
        return handle_;
    }

    // 判断 HANDLE 是否有效
    bool IsValid() const {
        return handle_ != INVALID_HANDLE_VALUE && handle_ != NULL;
    }

    // 释放所有权（调用者负责关闭）
    HANDLE Release() {
        HANDLE h = handle_;
        handle_ = INVALID_HANDLE_VALUE;
        return h;
    }

    // 重置为新的 HANDLE（关闭旧的）
    void Reset(HANDLE newHandle = INVALID_HANDLE_VALUE) {
        Close();
        handle_ = newHandle;
    }

    // 关闭 HANDLE
    void Close() {
        if (IsValid()) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
    }

    // 隐式转换为 HANDLE（方便使用）
    operator HANDLE() const {
        return handle_;
    }

    // 获取 HANDLE 地址（用于输出参数）
    HANDLE* GetAddressOf() {
        Close();  // 确保之前的 HANDLE 已关闭
        return &handle_;
    }
};

// RAII 内存块包装器 - 自动管理 new[] 分配的内存
template<typename T>
class ScopedArray {
private:
    T* ptr_;
    size_t size_;

public:
    // 构造函数
    explicit ScopedArray(size_t size = 0)
        : ptr_(size > 0 ? new T[size] : nullptr)
        , size_(size)
    {}

    // 析构函数 - 自动释放内存
    ~ScopedArray() {
        delete[] ptr_;
    }

    // 禁止拷贝
    ScopedArray(const ScopedArray&) = delete;
    ScopedArray& operator=(const ScopedArray&) = delete;

    // 支持移动
    ScopedArray(ScopedArray&& other) noexcept
        : ptr_(other.ptr_)
        , size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    ScopedArray& operator=(ScopedArray&& other) noexcept {
        if (this != &other) {
            delete[] ptr_;
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // 获取原始指针
    T* Get() const {
        return ptr_;
    }

    // 获取大小
    size_t Size() const {
        return size_;
    }

    // 判断是否有效
    bool IsValid() const {
        return ptr_ != nullptr;
    }

    // 释放所有权
    T* Release() {
        T* p = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return p;
    }

    // 重置
    void Reset(size_t newSize = 0) {
        delete[] ptr_;
        ptr_ = newSize > 0 ? new T[newSize] : nullptr;
        size_ = newSize;
    }

    // 数组访问操作符
    T& operator[](size_t index) {
        return ptr_[index];
    }

    const T& operator[](size_t index) const {
        return ptr_[index];
    }

    // 隐式转换为指针
    operator T*() const {
        return ptr_;
    }
};

// RAII LocalFree 包装器 - 用于 Windows API 分配的内存
template<typename T>
class ScopedLocalAlloc {
private:
    T* ptr_;

public:
    explicit ScopedLocalAlloc(T* p = nullptr)
        : ptr_(p)
    {}

    ~ScopedLocalAlloc() {
        if (ptr_) {
            LocalFree(ptr_);
        }
    }

    // 禁止拷贝
    ScopedLocalAlloc(const ScopedLocalAlloc&) = delete;
    ScopedLocalAlloc& operator=(const ScopedLocalAlloc&) = delete;

    // 支持移动
    ScopedLocalAlloc(ScopedLocalAlloc&& other) noexcept
        : ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
    }

    ScopedLocalAlloc& operator=(ScopedLocalAlloc&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                LocalFree(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* Get() const {
        return ptr_;
    }

    bool IsValid() const {
        return ptr_ != nullptr;
    }

    T* Release() {
        T* p = ptr_;
        ptr_ = nullptr;
        return p;
    }

    void Reset(T* newPtr = nullptr) {
        if (ptr_) {
            LocalFree(ptr_);
        }
        ptr_ = newPtr;
    }

    operator T*() const {
        return ptr_;
    }

    T** GetAddressOf() {
        if (ptr_) {
            LocalFree(ptr_);
            ptr_ = nullptr;
        }
        return &ptr_;
    }
};

} // namespace FR
