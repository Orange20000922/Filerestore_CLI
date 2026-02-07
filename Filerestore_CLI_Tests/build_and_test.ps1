#!/usr/bin/env pwsh
# Google Test 单元测试构建和运行脚本

param(
    [string]$Configuration = "Debug",
    [string]$TestFilter = "*"
)

$ErrorActionPreference = "Stop"

$MSBuildPath = "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe"
$ProjectPath = "D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI_Tests\Filerestore_CLI_Tests.vcxproj"
$TestExePath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\$Configuration\Tests\Filerestore_CLI_Tests.exe"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Filerestore_CLI Unit Test Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuration: $Configuration" -ForegroundColor Yellow
Write-Host "Test Filter:   $TestFilter" -ForegroundColor Yellow

# 步骤 1: 恢复 NuGet 包
Write-Host "`n[1/3] Restoring NuGet packages..." -ForegroundColor Green
if (!(Test-Path "D:\Users\21405\source\repos\Filerestore_CLI\packages\gtest.1.14.0")) {
    Write-Host "  Installing Google Test via NuGet..." -ForegroundColor Yellow
    nuget restore $ProjectPath
} else {
    Write-Host "  Google Test already installed" -ForegroundColor Gray
}

# 步骤 2: 构建测试项目
Write-Host "`n[2/3] Building test project..." -ForegroundColor Green
$buildArgs = @(
    $ProjectPath,
    "/p:Configuration=$Configuration",
    "/p:Platform=x64",
    "/t:Build",
    "/v:minimal",
    "/m"
)

& $MSBuildPath $buildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[FAIL] Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "  Build succeeded" -ForegroundColor Green

# 步骤 3: 运行测试
Write-Host "`n[3/3] Running tests..." -ForegroundColor Green

if (!(Test-Path $TestExePath)) {
    Write-Host "[ERROR] Test executable not found: $TestExePath" -ForegroundColor Red
    exit 1
}

$testArgs = @(
    "--gtest_color=yes"
)

if ($TestFilter -ne "*") {
    $testArgs += "--gtest_filter=$TestFilter"
}

Write-Host "`nExecuting: $TestExePath $($testArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

& $TestExePath $testArgs

$testExitCode = $LASTEXITCODE

# 总结
Write-Host "`n========================================" -ForegroundColor Cyan
if ($testExitCode -eq 0) {
    Write-Host "  All tests PASSED!" -ForegroundColor Green
} else {
    Write-Host "  Some tests FAILED (exit code: $testExitCode)" -ForegroundColor Red
}
Write-Host "========================================" -ForegroundColor Cyan

exit $testExitCode
