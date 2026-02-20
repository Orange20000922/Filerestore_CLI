# Filerestore_CLI Release 打包脚本
# 用法:
#   .\package_release.ps1                         # 默认 v1.0.0，CPU only
#   .\package_release.ps1 -Version "1.0.1"
#   .\package_release.ps1 -IncludeCUDA            # 附带 CUDA DLL
#   .\package_release.ps1 -SignCert "AABBCCDD..."  # Authenticode 签名（需 signtool）
#   .\package_release.ps1 -Build                  # 打包前先执行 Release 构建

param(
    [string]$Version    = "1.0.0",
    [switch]$IncludeCUDA = $false,
    [string]$SignCert   = "",          # Authenticode 证书 SHA1 指纹（留空=不签名）
    [string]$SignTimestamp = "http://timestamp.digicert.com",
    [switch]$Build      = $false       # 是否在打包前先构建
)

$ErrorActionPreference = "Stop"

# ── 路径配置 ────────────────────────────────────────────────
$ProjectRoot  = $PSScriptRoot
$VcxprojPath  = "$ProjectRoot\Filerestore_CLI\Filerestore_CLI.vcxproj"
$ReleaseDir   = "$ProjectRoot\x64\Release"
$OutputDir    = "$ProjectRoot\release_packages"
$PackageName  = "Filerestore_CLI_v${Version}_x64"
if ($IncludeCUDA) { $PackageName += "_cuda" }
$PackageDir   = "$OutputDir\$PackageName"
$ZipPath      = "$OutputDir\$PackageName.zip"
$Msbuild      = "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Filerestore_CLI Release Packager" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Version    : $Version"
Write-Host "  CUDA       : $IncludeCUDA"
Write-Host "  Sign       : $(if ($SignCert) { $SignCert.Substring(0, [Math]::Min(8,$SignCert.Length)) + '...' } else { '(skip)' })"
Write-Host "  Output     : $PackageDir"
Write-Host ""

# ── 可选：先构建 ─────────────────────────────────────────────
if ($Build) {
    if (-not (Test-Path $Msbuild)) {
        Write-Error "MSBuild not found at $Msbuild"
        exit 1
    }
    Write-Host "Building Release x64..." -ForegroundColor Green
    & $Msbuild $VcxprojPath /p:Configuration=Release /p:Platform=x64 /t:Build /v:minimal
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed (exit $LASTEXITCODE)"
        exit 1
    }
    Write-Host "Build completed." -ForegroundColor Green
    Write-Host ""
}

# ── 检查主程序 ───────────────────────────────────────────────
if (-not (Test-Path "$ReleaseDir\Filerestore_CLI.exe")) {
    Write-Host "Error: Filerestore_CLI.exe not found at $ReleaseDir" -ForegroundColor Red
    Write-Host "Run with -Build to compile first, or build manually." -ForegroundColor Red
    exit 1
}

# ── 可选：Authenticode 签名 ──────────────────────────────────
if ($SignCert) {
    $signtool = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if (-not $signtool) {
        # 尝试从 Windows SDK 找
        $signtool = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin" -Filter signtool.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -Last 1
    }
    if (-not $signtool) {
        Write-Warning "signtool.exe not found, skipping signing."
        $SignCert = ""
    } else {
        $signtoolPath = if ($signtool -is [string]) { $signtool } else { $signtool.FullName }
        Write-Host "Signing Filerestore_CLI.exe..." -ForegroundColor Green
        & $signtoolPath sign /sha1 $SignCert /fd SHA256 /tr $SignTimestamp /td SHA256 "$ReleaseDir\Filerestore_CLI.exe"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Signing failed"
            exit 1
        }
        Write-Host "Signed OK." -ForegroundColor Green
        Write-Host ""
    }
}

# ── 创建目录结构 ─────────────────────────────────────────────
if (Test-Path $PackageDir) {
    Write-Host "Removing existing package directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $PackageDir
}
New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null
New-Item -ItemType Directory -Force -Path "$PackageDir\models" | Out-Null
New-Item -ItemType Directory -Force -Path "$PackageDir\langs"  | Out-Null

Write-Host "Copying files..." -ForegroundColor Green

# ── 1. 主程序 ────────────────────────────────────────────────
Write-Host "  [+] Filerestore_CLI.exe"
Copy-Item "$ReleaseDir\Filerestore_CLI.exe" "$PackageDir\"

# ── 2. ONNX Runtime（可选，不存在则跳过）───────────────────────
if (Test-Path "$ReleaseDir\onnxruntime.dll") {
    Write-Host "  [+] onnxruntime.dll (ML inference, CPU)"
    Copy-Item "$ReleaseDir\onnxruntime.dll" "$PackageDir\"
} else {
    Write-Host "  [-] onnxruntime.dll not found, skipping (ML disabled)" -ForegroundColor Yellow
}

# ── 3. CUDA DLL（可选）──────────────────────────────────────
if ($IncludeCUDA) {
    $cudaDlls = @(
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_tensorrt.dll"
    )
    foreach ($dll in $cudaDlls) {
        if (Test-Path "$ReleaseDir\$dll") {
            Write-Host "  [+] $dll (CUDA)" -ForegroundColor Yellow
            Copy-Item "$ReleaseDir\$dll" "$PackageDir\"
        } else {
            Write-Warning "  [-] $dll not found, skipping"
        }
    }
}

# ── 4. ML 模型 ───────────────────────────────────────────────
if (Test-Path "$ReleaseDir\models") {
    $modelCount = (Get-ChildItem "$ReleaseDir\models" -File).Count
    Write-Host "  [+] models/ ($modelCount files)"
    Copy-Item "$ReleaseDir\models\*" "$PackageDir\models\" -Recurse
} else {
    Write-Host "  [-] models/ not found, skipping" -ForegroundColor Yellow
}

# ── 5. 语言文件 ──────────────────────────────────────────────
if (Test-Path "$ReleaseDir\langs") {
    $langCount = (Get-ChildItem "$ReleaseDir\langs" -File).Count
    Write-Host "  [+] langs/ ($langCount files)"
    Copy-Item "$ReleaseDir\langs\*" "$PackageDir\langs\" -Recurse
} else {
    Write-Host "  [-] langs/ not found, skipping" -ForegroundColor Yellow
}

# ── 6. README.txt ────────────────────────────────────────────
Write-Host "  [+] README.txt"
$hasML  = Test-Path "$PackageDir\onnxruntime.dll"
$mlNote = if ($hasML) { "Included (onnxruntime.dll)" } else { "Not included (ML commands disabled)" }

$ReadmeContent = @"
Filerestore_CLI v$Version
========================
NTFS File Recovery Tool — Windows x64

GitHub  : https://github.com/Orange20000922/Filerestore_CLI
License : MIT

REQUIREMENTS
------------
  - Windows 10/11 x64
  - Administrator privileges (raw disk access)
  - NTFS file system
  - ML inference: $mlNote

QUICK START
-----------
Run as Administrator:

  Filerestore_CLI.exe          # Interactive CLI
  Filerestore_CLI.exe --tui    # TUI graphical interface (recommended)

COMMANDS
--------
  help                                  Show all commands and descriptions

  [ USN Targeted Recovery ]
  usnlist <drive>                       List recently deleted files with MFT
                                        verification and confidence scoring
  usnrecover <drive> <target> <output>  Recover file by index / filename /
                                        MFT record number
  recover <drive> [file] [output]       Smart recovery wizard (USN + MFT +
                                        signature joint scan)

  [ Signature Carving ]
  carve <drive> <type> [output]         Single-threaded signature scan
  carvepool <drive> <types> <output>    Thread-pool scan (recommended, ~2700 MB/s)
    types: jpg,png,gif,zip,pdf,docx...  Comma-separated; "all" = all supported
  Example:
    carvepool C all D:\recovered\       Scan all types on C:
    carvepool D jpg,png D:\recovered\ 8 Use 8 threads

  [ Real-time Monitoring ]
  monitor <drive>                       Start real-time deletion monitor (USN)
  Filerestore_CLI.exe --monitor-daemon  Run as background monitor daemon

  [ Scan & Search ]
  scan <drive>                          Scan deleted MFT entries
  search <drive> <pattern>              Search deleted files by name pattern
  list                                  List previous scan results

  [ Diagnostics ]
  diag <drive>                          Disk health and SMART report
  pe <file>                             PE (EXE/DLL) structure analysis

  [ System ]
  setlang <en|zh>                       Switch language
  exit / quit                           Exit

BAD CLUSTER FILTERING (v1.0.0+)
---------------------------------
All recovery paths now perform per-cluster overwrite detection.
Partially overwritten files are recovered with bad clusters zero-filled,
and the output shows cluster health percentage:
  e.g.  簇健康: 850/1000 (85.0%) | 覆写簇: 150

KERNEL DRIVER BRIDGE (experimental)
-------------------------------------
Optional kernel-mode IRP capture for real-time MFT snapshots.
Requires signed driver and system test mode.
Disabled by default; see SECURITY.md on GitHub for details.

NOTES
-----
  - Recovered files may contain malicious content (e.g. recovered .exe).
    Scan with antivirus before opening.
  - Cache files (MFT snapshots, scan results) are stored locally,
    unencrypted. Keep output directories private.
"@

$ReadmeContent | Out-File -FilePath "$PackageDir\README.txt" -Encoding UTF8

# ── 7. 创建 ZIP ──────────────────────────────────────────────
Write-Host ""
Write-Host "Creating ZIP archive..." -ForegroundColor Green
if (Test-Path $ZipPath) { Remove-Item $ZipPath }
Compress-Archive -Path "$PackageDir\*" -DestinationPath $ZipPath -CompressionLevel Optimal

# ── 8. SHA256 校验文件 ───────────────────────────────────────
$sha256    = (Get-FileHash $ZipPath -Algorithm SHA256).Hash.ToLower()
$checksumFile = "$OutputDir\$PackageName.sha256"
"$sha256  $PackageName.zip" | Out-File -FilePath $checksumFile -Encoding ASCII
Write-Host "SHA256: $sha256" -ForegroundColor DarkGray

# ── 结果汇总 ─────────────────────────────────────────────────
$zipSizeMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Package created successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ZIP     : $ZipPath ($zipSizeMB MB)"
Write-Host "  SHA256  : $checksumFile"
Write-Host ""

Write-Host "Package contents:" -ForegroundColor Yellow
Get-ChildItem $PackageDir -Recurse | ForEach-Object {
    $rel  = $_.FullName.Replace($PackageDir, "").TrimStart("\")
    $size = if ($_.PSIsContainer) { "[DIR]" } else { "$([math]::Round($_.Length / 1KB, 1)) KB" }
    Write-Host "  $rel  $size"
}

Write-Host ""
Write-Host "Upload to GitHub Releases:" -ForegroundColor Green
Write-Host "  $ZipPath"
Write-Host "  $checksumFile"
