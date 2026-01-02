@echo off
REM ========================================
REM 文件恢复工具测试脚本 (批处理版本)
REM File Recovery Tool Test Script (Batch)
REM ========================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   文件恢复工具测试脚本
echo   File Recovery Tool Test Script
echo ========================================
echo.

REM 配置参数
set TEST_DIR=C:\Temp
set TEST_FILE=test_recovery_file.txt
set FULL_PATH=%TEST_DIR%\%TEST_FILE%
set PROGRAM_PATH=x64\Debug\ImportTableAnalyzer.exe

REM ========== Step 1: 创建测试目录和文件 ==========
echo ========== Step 1: Creating Test File ==========
echo.

if not exist "%TEST_DIR%" (
    mkdir "%TEST_DIR%"
    echo [+] Created test directory: %TEST_DIR%
)

REM 创建测试文件
echo ====================================== > "%FULL_PATH%"
echo Test File for Recovery Tool >> "%FULL_PATH%"
echo Created at: %date% %time% >> "%FULL_PATH%"
echo ====================================== >> "%FULL_PATH%"
echo. >> "%FULL_PATH%"
echo This is a test file to verify deleted file scanning and recovery. >> "%FULL_PATH%"
echo. >> "%FULL_PATH%"

REM 添加更多内容使文件稍大一些
for /L %%i in (1,1,100) do (
    echo Line %%i: Test data - Random number !RANDOM! >> "%FULL_PATH%"
)

if exist "%FULL_PATH%" (
    echo [+] Created test file: %FULL_PATH%
    for %%A in ("%FULL_PATH%") do echo     Size: %%~zA bytes
    echo     Time: %date% %time%
    echo.
) else (
    echo [-] ERROR: Failed to create test file!
    pause
    exit /b 1
)

REM ========== Step 2: 等待用户确认删除 ==========
echo [!] Press ANY KEY to DELETE the test file...
pause > nul

REM ========== Step 3: 删除文件 ==========
echo.
echo ========== Step 2: Deleting Test File ==========
echo [*] Deleting file (permanent deletion)...

del /F /Q "%FULL_PATH%" > nul 2>&1

if not exist "%FULL_PATH%" (
    echo [+] File deleted successfully!
    echo     Deleted at: %date% %time%
    echo.
) else (
    echo [-] ERROR: Failed to delete file!
    pause
    exit /b 1
)

REM 等待文件系统刷新
echo [*] Waiting for filesystem to flush...
timeout /t 2 /nobreak > nul

REM ========== Step 4: 启动恢复程序 ==========
echo.
echo ========== Step 3: Launching Recovery Program ==========
echo.

REM 查找程序 - 自动选择最新编译的版本
echo [*] Searching for program executable...

REM 使用 PowerShell 查找最新的可执行文件
set FOUND_PATH=
for /f "delims=" %%i in ('powershell -NoProfile -Command "$paths = @('%PROGRAM_PATH%', 'x64\Release\ImportTableAnalyzer.exe', 'x64\Debug\ImportTableAnalyzer.exe', 'Debug\ImportTableAnalyzer.exe', 'Release\ImportTableAnalyzer.exe', 'ImportTableAnalyzer.exe'); $files = $paths | Where-Object { Test-Path $_ } | ForEach-Object { Get-Item $_ }; if ($files) { ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName } else { '' }"') do set FOUND_PATH=%%i

REM 如果 PowerShell 失败，使用传统方法（优先 Release）
if not defined FOUND_PATH (
    if exist "x64\Release\ImportTableAnalyzer.exe" (
        set FOUND_PATH=x64\Release\ImportTableAnalyzer.exe
    ) else if exist "x64\Debug\ImportTableAnalyzer.exe" (
        set FOUND_PATH=x64\Debug\ImportTableAnalyzer.exe
    ) else if exist "%PROGRAM_PATH%" (
        set FOUND_PATH=%PROGRAM_PATH%
    ) else if exist "Debug\ImportTableAnalyzer.exe" (
        set FOUND_PATH=Debug\ImportTableAnalyzer.exe
    ) else if exist "Release\ImportTableAnalyzer.exe" (
        set FOUND_PATH=Release\ImportTableAnalyzer.exe
    ) else if exist "ImportTableAnalyzer.exe" (
        set FOUND_PATH=ImportTableAnalyzer.exe
    )
)

if defined FOUND_PATH (
    echo [+] Found program: !FOUND_PATH!

    REM 显示程序编译时间
    for %%A in ("!FOUND_PATH!") do (
        echo     File size: %%~zA bytes
        echo     Modified: %%~tA
    )
    echo.
    echo [*] Suggested test commands:
    echo     listdeleted C none
    echo     searchdeleted C test .txt
    echo     searchdeleted C test_recovery
    echo.
    echo [*] Test file details:
    echo     Original path: %FULL_PATH%
    echo     File name: %TEST_FILE%
    echo     Directory: %TEST_DIR%
    echo     Drive: C:
    echo.
    echo [*] Launching program in 3 seconds...
    timeout /t 3 /nobreak > nul
    echo.

    start "" "!FOUND_PATH!"
    echo [+] Program launched!
) else (
    echo [-] ERROR: Program not found!
    echo.
    echo     Searched paths:
    echo     - %PROGRAM_PATH%
    echo     - x64\Release\ImportTableAnalyzer.exe
    echo     - Debug\ImportTableAnalyzer.exe
    echo     - Release\ImportTableAnalyzer.exe
    echo     - ImportTableAnalyzer.exe
    echo.
    echo     Please build the project first!
    echo.
    pause
    exit /b 1
)

echo.
echo ========== Test Setup Complete ==========
echo.
echo Test completed at: %date% %time%
echo.
pause
