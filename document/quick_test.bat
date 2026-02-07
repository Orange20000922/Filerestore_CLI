@echo off
REM Filerestore_CLI Quick Test Script
REM Usage: quick_test.bat [drive_letter]

setlocal enabledelayedexpansion

set EXE=..\x64\Release\Filerestore_CLI.exe
set DRIVE=%1
if "%DRIVE%"=="" set DRIVE=D:

echo ========================================
echo Filerestore_CLI Quick Test
echo ========================================
echo.

echo [TEST 1] Help command
%EXE% --cmd "help"
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Help command executed successfully
) else (
    echo [FAIL] Help command failed with exit code %ERRORLEVEL%
)
echo.

echo [TEST 2] Invalid command
%EXE% --cmd "invalid_cmd_xyz"
if %ERRORLEVEL% NEQ 0 (
    echo [PASS] Invalid command correctly rejected
) else (
    echo [FAIL] Invalid command should have failed
)
echo.

echo [TEST 3] Check drive %DRIVE%
%EXE% --cmd "checkdrive %DRIVE%"
if %ERRORLEVEL% EQU 0 (
    echo [PASS] CheckDrive executed
) else (
    echo [FAIL] CheckDrive failed
)
echo.

echo [TEST 4] List carved results
%EXE% --cmd "carvelist"
if %ERRORLEVEL% EQU 0 (
    echo [PASS] CarveList executed
) else (
    echo [FAIL] CarveList failed
)
echo.

echo ========================================
echo Checking debug.log...
echo ========================================
findstr /C:"Direct command mode" ..\x64\Release\debug.log | find /C ":"
echo command executions found in log
echo.

echo ========================================
echo Tests completed!
echo ========================================
pause
