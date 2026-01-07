@echo off
REM ============================================================================
REM ONNX Runtime Download Script for Filerestore_CLI
REM ============================================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "DEPS_DIR=%PROJECT_DIR%deps"
set "ONNX_DIR=%DEPS_DIR%\onnxruntime"

REM ONNX Runtime version
set "ONNX_VERSION=1.19.2"

REM Choose GPU or CPU version
echo ============================================
echo ONNX Runtime Installer for Filerestore_CLI
echo ============================================
echo.
echo Select version:
echo   1. GPU version (CUDA 12.x required)
echo   2. CPU only version
echo.
set /p CHOICE="Enter choice (1 or 2): "

if "%CHOICE%"=="1" (
    set "PACKAGE_NAME=onnxruntime-win-x64-gpu-%ONNX_VERSION%"
    set "DOWNLOAD_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ONNX_VERSION%/onnxruntime-win-x64-gpu-%ONNX_VERSION%.zip"
) else (
    set "PACKAGE_NAME=onnxruntime-win-x64-%ONNX_VERSION%"
    set "DOWNLOAD_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ONNX_VERSION%/onnxruntime-win-x64-%ONNX_VERSION%.zip"
)

echo.
echo Downloading %PACKAGE_NAME%...
echo URL: %DOWNLOAD_URL%
echo.

REM Create deps directory
if not exist "%DEPS_DIR%" mkdir "%DEPS_DIR%"

REM Download using PowerShell
set "ZIP_FILE=%DEPS_DIR%\%PACKAGE_NAME%.zip"

powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%ZIP_FILE%'}"

if not exist "%ZIP_FILE%" (
    echo ERROR: Download failed!
    echo Please download manually from:
    echo   %DOWNLOAD_URL%
    echo And extract to:
    echo   %ONNX_DIR%
    pause
    exit /b 1
)

echo Download complete. Extracting...

REM Extract
if exist "%ONNX_DIR%" rmdir /s /q "%ONNX_DIR%"
powershell -Command "& {Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%DEPS_DIR%' -Force}"

REM Rename to standard directory
if exist "%DEPS_DIR%\%PACKAGE_NAME%" (
    move "%DEPS_DIR%\%PACKAGE_NAME%" "%ONNX_DIR%"
)

REM Verify installation
if exist "%ONNX_DIR%\include\onnxruntime_cxx_api.h" (
    echo.
    echo ============================================
    echo Installation successful!
    echo ============================================
    echo ONNX Runtime installed to: %ONNX_DIR%
    echo.
    echo Directory structure:
    dir /b "%ONNX_DIR%"
    echo.
    echo You can now build the project with ML support.
    echo The build system will automatically detect ONNX Runtime.
) else (
    echo.
    echo ERROR: Installation verification failed!
    echo Please check the extracted files.
)

REM Cleanup
del "%ZIP_FILE%" 2>nul

echo.
pause
