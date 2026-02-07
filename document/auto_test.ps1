#!/usr/bin/env pwsh
# Filerestore_CLI 自动化测试脚本
# 用途：功能测试、性能分析、回归测试

param(
    [string]$ExePath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe",
    [string]$TestDrive = "D:",
    [switch]$SkipSlowTests
)

$ErrorActionPreference = "Continue"
$TestResults = @()

# 颜色输出函数
function Write-TestHeader($message) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $message -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

function Write-TestPass($message) {
    Write-Host "[PASS] $message" -ForegroundColor Green
}

function Write-TestFail($message) {
    Write-Host "[FAIL] $message" -ForegroundColor Red
}

function Write-TestInfo($message) {
    Write-Host "[INFO] $message" -ForegroundColor Yellow
}

# 执行命令并记录结果
function Invoke-TestCommand {
    param(
        [string]$TestName,
        [string]$Command,
        [int]$ExpectedExitCode = 0,
        [string]$ExpectedOutputPattern = $null,
        [int]$TimeoutSeconds = 30
    )

    Write-TestInfo "Running: $TestName"
    Write-TestInfo "Command: $Command"

    $startTime = Get-Date

    try {
        # 执行命令
        $output = & $ExePath --cmd $Command 2>&1
        $exitCode = $LASTEXITCODE
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds

        # 验证退出码
        $exitCodeMatch = ($exitCode -eq $ExpectedExitCode)

        # 验证输出模式（如果指定）
        $outputMatch = $true
        if ($ExpectedOutputPattern) {
            $outputMatch = ($output -match $ExpectedOutputPattern)
        }

        # 记录结果
        $result = @{
            TestName = $TestName
            Command = $Command
            ExitCode = $exitCode
            ExpectedExitCode = $ExpectedExitCode
            Duration = [math]::Round($duration, 2)
            Success = ($exitCodeMatch -and $outputMatch)
            Output = $output -join "`n"
        }

        $script:TestResults += $result

        # 输出结果
        if ($result.Success) {
            Write-TestPass "$TestName (${duration}s)"
        } else {
            Write-TestFail "$TestName"
            if (!$exitCodeMatch) {
                Write-Host "  Expected exit code: $ExpectedExitCode, Got: $exitCode" -ForegroundColor Red
            }
            if (!$outputMatch) {
                Write-Host "  Output pattern not found: $ExpectedOutputPattern" -ForegroundColor Red
            }
        }

        return $result
    }
    catch {
        Write-TestFail "$TestName - Exception: $_"
        return @{
            TestName = $TestName
            Command = $Command
            Success = $false
            Error = $_.Exception.Message
        }
    }
}

# 分析日志文件获取性能指标
function Get-PerformanceMetrics {
    param([string]$LogPath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\debug.log")

    Write-TestHeader "Performance Metrics from Log"

    if (Test-Path $LogPath) {
        # 读取最后 200 行日志
        $logLines = Get-Content $LogPath -Tail 200 -Encoding UTF8

        # 提取扫描速度
        $scanSpeed = $logLines | Select-String "已扫描.*找到" | Select-Object -Last 5
        if ($scanSpeed) {
            Write-TestInfo "Recent scan results:"
            $scanSpeed | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        }

        # 提取路径缓存命中率
        $cacheHit = $logLines | Select-String "命中率" | Select-Object -Last 1
        if ($cacheHit) {
            Write-TestInfo "Cache performance: $cacheHit"
        }

        # 提取警告/错误
        $warnings = $logLines | Select-String "\[WARNING\]|\[ERROR\]" | Select-Object -Last 10
        if ($warnings) {
            Write-Host "`nRecent warnings/errors:" -ForegroundColor Yellow
            $warnings | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
        }
    }
    else {
        Write-TestInfo "Log file not found: $LogPath"
    }
}

# ============================================================================
# 测试套件
# ============================================================================

Write-Host @"

  ______ _ _                     _                  _____ _      _____
 |  ____(_) |                   | |                / ____| |    |_   _|
 | |__   _| | ___ _ __ ___  ___| |_ ___  _ __ ___| |    | |     | |
 |  __| | | |/ _ \ '__/ _ \/ __| __/ _ \| '__/ _ \ |    | |     | |
 | |    | | |  __/ | |  __/\__ \ || (_) | | |  __/ |____| |____ _| |_
 |_|    |_|_|\___|_|  \___||___/\__\___/|_|  \___|\_____|______|_____|

                    Automated Test Suite

"@ -ForegroundColor Cyan

Write-TestInfo "Test executable: $ExePath"
Write-TestInfo "Test drive: $TestDrive"
Write-TestInfo "Skip slow tests: $SkipSlowTests"

# ============================================================================
# 基础功能测试
# ============================================================================

Write-TestHeader "Basic Functionality Tests"

Invoke-TestCommand -TestName "Help Command" -Command "help"
Invoke-TestCommand -TestName "Exit Command" -Command "exit"
Invoke-TestCommand -TestName "Invalid Command" -Command "nonexistent_cmd" -ExpectedExitCode 1

# ============================================================================
# 参数验证测试
# ============================================================================

Write-TestHeader "Parameter Validation Tests"

Invoke-TestCommand -TestName "ListDeleted - Missing Drive" -Command "listdeleted"
Invoke-TestCommand -TestName "Recover - Invalid Drive" -Command "recover XYZ:"
Invoke-TestCommand -TestName "CarvePool - Missing Args" -Command "carvepool"

# ============================================================================
# 功能测试（需要有效驱动器）
# ============================================================================

if ($TestDrive -and (Test-Path "$TestDrive\")) {
    Write-TestHeader "Functional Tests (Drive: $TestDrive)"

    # 快速测试
    Invoke-TestCommand -TestName "CheckDrive" -Command "checkdrive $TestDrive"

    if (!$SkipSlowTests) {
        # 慢速测试（需要较长时间）
        Write-TestInfo "Running slow tests (can be skipped with -SkipSlowTests)"

        Invoke-TestCommand -TestName "ListDeleted - Quick Scan" `
            -Command "listdeleted $TestDrive all" `
            -TimeoutSeconds 60

        Invoke-TestCommand -TestName "CarveList - Check Cache" `
            -Command "carvelist"
    }
}
else {
    Write-TestInfo "Skipping functional tests - invalid test drive: $TestDrive"
}

# ============================================================================
# 性能分析
# ============================================================================

Get-PerformanceMetrics

# ============================================================================
# 测试报告
# ============================================================================

Write-TestHeader "Test Summary"

$totalTests = $TestResults.Count
$passedTests = ($TestResults | Where-Object { $_.Success }).Count
$failedTests = $totalTests - $passedTests
$avgDuration = ($TestResults | Where-Object { $_.Duration } | Measure-Object -Property Duration -Average).Average

Write-Host "Total tests:   $totalTests" -ForegroundColor White
Write-Host "Passed:        $passedTests" -ForegroundColor Green
Write-Host "Failed:        $failedTests" -ForegroundColor $(if ($failedTests -eq 0) { "Green" } else { "Red" })
Write-Host "Avg Duration:  $([math]::Round($avgDuration, 2))s" -ForegroundColor Yellow

# 失败的测试详情
if ($failedTests -gt 0) {
    Write-Host "`nFailed Tests:" -ForegroundColor Red
    $TestResults | Where-Object { !$_.Success } | ForEach-Object {
        Write-Host "  - $($_.TestName): Exit Code $($_.ExitCode)" -ForegroundColor Red
    }
}

# 导出详细报告到 JSON
$reportPath = "test_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$TestResults | ConvertTo-Json -Depth 5 | Out-File $reportPath -Encoding UTF8
Write-TestInfo "Detailed report saved to: $reportPath"

# 返回退出码
if ($failedTests -eq 0) {
    Write-Host "`n[SUCCESS] All tests passed!" -ForegroundColor Green
    exit 0
}
else {
    Write-Host "`n[FAILURE] $failedTests test(s) failed" -ForegroundColor Red
    exit 1
}
