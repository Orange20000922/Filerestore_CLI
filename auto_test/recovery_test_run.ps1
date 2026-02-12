#!/usr/bin/env pwsh
# ============================================================================
# Recovery Performance Test - Run Recovery Tests
# ============================================================================
# Executes various recovery commands and captures results.
#
# Usage:
#   .\recovery_test_run.ps1
#   .\recovery_test_run.ps1 -TestDrive D -OutputDir "D:\recovery_test_results"
# ============================================================================

param(
    [string]$ExePath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe",
    [string]$LogPath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\debug.log",
    [string]$TestDrive = "D",
    [string]$OutputDir = "D:\recovery_test_results",
    [string]$MetadataPath = $null,  # Auto-detected if not specified
    [switch]$SkipUsnList,
    [switch]$SkipUsnRecover,
    [switch]$SkipCarvePool,
    [switch]$SkipRecover
)

$ErrorActionPreference = "Continue"

function Write-Header($msg) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

function Write-Info($msg)  { Write-Host "[INFO] $msg" -ForegroundColor Yellow }
function Write-OK($msg)    { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Err($msg)   { Write-Host "[ERR]  $msg" -ForegroundColor Red }

Write-Header "Recovery Test - Running Recovery Commands"
Write-Info "Executable: $ExePath"
Write-Info "Log file: $LogPath"
Write-Info "Test Drive: $TestDrive"
Write-Info "Output Dir: $OutputDir"

# Validate executable
if (-not (Test-Path $ExePath)) {
    Write-Err "Executable not found: $ExePath"
    Write-Info "Build Release x64 first."
    exit 1
}

# Find metadata file
if (-not $MetadataPath) {
    $MetadataPath = Join-Path $OutputDir "test_files_metadata.json"
}
if (-not (Test-Path $MetadataPath)) {
    Write-Err "Metadata file not found: $MetadataPath"
    Write-Info "Run recovery_test_create.ps1 first."
    exit 1
}

# Load metadata
$metadata = Get-Content $MetadataPath -Raw | ConvertFrom-Json
$testFiles = $metadata.files
$deleteTime = [DateTime]::Parse($metadata.deleteTime)
$now = Get-Date
$ageMinutes = ($now - $deleteTime).TotalMinutes

Write-Info "Loaded $($testFiles.Count) test file records"
Write-Info "Files deleted: $ageMinutes minutes ago"

# Create recovery output directory
$recoveryDir = Join-Path $OutputDir "recovered"
if (Test-Path $recoveryDir) {
    Remove-Item $recoveryDir -Recurse -Force
}
New-Item -ItemType Directory -Path $recoveryDir -Force | Out-Null

# Results tracking
$results = @{
    startTime = $now.ToString("yyyy-MM-dd HH:mm:ss")
    testDrive = $TestDrive
    testFileCount = $testFiles.Count
    deleteAgeMinutes = [math]::Round($ageMinutes, 1)
    usnlist = @{}
    usnrecover = @{}
    carvepool = @{}
    recover = @{}
}

# Clear old log
Remove-Item $LogPath -Force -ErrorAction SilentlyContinue

# ============================================================================
# Phase 1: USN List with Validation
# ============================================================================
if (-not $SkipUsnList) {
    Write-Header "Phase 1: USN List (usnlist --validate)"

    $cmd = "usnlist $TestDrive 1 --validate --pattern=test_"
    Write-Info "Command: $cmd"

    $startTime = Get-Date
    & $ExePath --cmd $cmd --test 2>$null
    $exitCode = $LASTEXITCODE
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds

    Write-Info "Exit code: $exitCode, Duration: $([math]::Round($duration, 1))s"

    # Parse log for results
    if (Test-Path $LogPath) {
        $logLines = Get-Content $LogPath -Encoding UTF8 -Tail 5000
        $outputLines = $logLines | Where-Object { $_ -match '\[OUTPUT\]' } | ForEach-Object {
            if ($_ -match '\[OUTPUT\]\s*(.*)$') { $Matches[1] }
        }

        # Find our test files
        $testFileResults = @{}
        $statusCounts = @{}

        foreach ($line in $outputLines) {
            if ($line -match 'test_\d{4}_') {
                if ($line -match '^\s*(\d+)\s+(\S+)\s.*\[(\w+)\]') {
                    $fileName = $Matches[2]
                    $status = $Matches[3]
                    $testFileResults[$fileName] = $status
                    if (-not $statusCounts.ContainsKey($status)) { $statusCounts[$status] = 0 }
                    $statusCounts[$status]++
                }
            }
        }

        $results.usnlist = @{
            durationSeconds = [math]::Round($duration, 1)
            exitCode = $exitCode
            testFilesFound = $testFileResults.Count
            statusCounts = $statusCounts
            files = $testFileResults
        }

        Write-OK "Found $($testFileResults.Count) test files in USN list"
        Write-Host "`n  Status distribution:" -ForegroundColor White
        foreach ($key in $statusCounts.Keys | Sort-Object) {
            Write-Host ("    {0,-20} {1,3}" -f $key, $statusCounts[$key])
        }
    }
}

# ============================================================================
# Phase 2: USN Recover
# ============================================================================
if (-not $SkipUsnRecover) {
    Write-Header "Phase 2: USN Recover (usnrecover)"

    $usnRecoverDir = Join-Path $recoveryDir "usnrecover"
    New-Item -ItemType Directory -Path $usnRecoverDir -Force | Out-Null

    # 使用真实文件名进行恢复测试（取前5个文件）
    $testFileNames = $testFiles | Select-Object -First 5 | ForEach-Object { $_.fileName }

    Write-Info "Testing recovery with specific filenames:"
    $testFileNames | ForEach-Object { Write-Host "  $_" }

    $successCount = 0
    $failCount = 0
    $totalDuration = 0

    foreach ($fileName in $testFileNames) {
        $cmd = "usnrecover $TestDrive `"$fileName`" `"$usnRecoverDir`""
        $startTime = Get-Date
        & $ExePath --cmd $cmd --test 2>$null
        $duration = ((Get-Date) - $startTime).TotalSeconds
        $totalDuration += $duration

        # Check if file was recovered
        $recoveredFile = Join-Path $usnRecoverDir $fileName
        if (Test-Path $recoveredFile) {
            $successCount++
            Write-Host "  [OK] $fileName ($([math]::Round($duration, 1))s)" -ForegroundColor Green
        } else {
            $failCount++
            Write-Host "  [FAIL] $fileName ($([math]::Round($duration, 1))s)" -ForegroundColor Red
        }
    }

    $recoveredFiles = Get-ChildItem $usnRecoverDir -File -ErrorAction SilentlyContinue
    $recoveredCount = ($recoveredFiles | Measure-Object).Count

    $results.usnrecover = @{
        durationSeconds = [math]::Round($totalDuration, 1)
        exitCode = 0
        filesRecovered = $recoveredCount
        successCount = $successCount
        failCount = $failCount
        outputDir = $usnRecoverDir
    }

    Write-OK "Recovered $successCount / $($testFileNames.Count) files to $usnRecoverDir"

    $startTime = Get-Date
    & $ExePath --cmd $cmd --test 2>$null
    $exitCode = $LASTEXITCODE
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds

    Write-Info "Exit code: $exitCode, Duration: $([math]::Round($duration, 1))s"

    # Count recovered files
    $recoveredFiles = Get-ChildItem $usnRecoverDir -File -ErrorAction SilentlyContinue
    $recoveredCount = ($recoveredFiles | Measure-Object).Count

    # Parse log for status
    $statusCounts = @{}
    if (Test-Path $LogPath) {
        $logLines = Get-Content $LogPath -Encoding UTF8 -Tail 5000

        # Match recovery status lines
        $logLines | Where-Object { $_ -match '\[OUTPUT\].*\[(SUCCESS|MFT_REUSED|NO_DATA|RESIDENT|MFT_REUSED_OK|PARTIAL)\]' } | ForEach-Object {
            if ($_ -match '\[(SUCCESS|MFT_REUSED|NO_DATA|RESIDENT|MFT_REUSED_OK|PARTIAL)\]') {
                $status = $Matches[1]
                if (-not $statusCounts.ContainsKey($status)) { $statusCounts[$status] = 0 }
                $statusCounts[$status]++
            }
        }
    }

    $results.usnrecover = @{
        durationSeconds = [math]::Round($duration, 1)
        exitCode = $exitCode
        filesRecovered = $recoveredCount
        statusCounts = $statusCounts
        outputDir = $usnRecoverDir
    }

    Write-OK "Recovered $recoveredCount files to $usnRecoverDir"
}

# ============================================================================
# Phase 3: CarvePool - Parse recent scan results from log/cache
# ============================================================================
if (-not $SkipCarvePool) {
    Write-Header "Phase 3: CarvePool (from cache/log)"

    Write-Info "Parsing recent CarvePool results from log..."
    Write-Info "(Full disk scan takes ~60 min, using cached results)"

    $foundCount = 0
    $deletedCount = 0
    $scanSpeed = 0
    $confidenceDist = @{ high = 0; medium = 0; low = 0 }

    if (Test-Path $LogPath) {
        $logLines = Get-Content $LogPath -Encoding UTF8 -Tail 10000

        # Find last scan summary
        $scanCompleteLine = $logLines | Where-Object { $_ -match 'Scan Complete|Files found:' } | Select-Object -Last 1
        if ($scanCompleteLine) {
            Write-Info "Found scan result: $scanCompleteLine"
        }

        # Parse "Files found: X" or "Total files found: X"
        $foundLine = $logLines | Where-Object { $_ -match 'files? found[:\s]+(\d+)' } | Select-Object -Last 1
        if ($foundLine -match '(\d+)') {
            $foundCount = [int]$Matches[1]
        }

        # Parse scan speed
        $speedLine = $logLines | Where-Object { $_ -match 'Scan speed[:\s]+([\d.]+)\s*MB/s' -or $_ -match 'Average speed[:\s]+([\d.]+)\s*MB/s' } | Select-Object -Last 1
        if ($speedLine -match '([\d.]+)\s*MB/s') {
            $scanSpeed = [double]$Matches[1]
        }

        # Parse deletion status counts
        $activeLine = $logLines | Where-Object { $_ -match 'Active files.*?:\s*(\d+)' } | Select-Object -Last 1
        $deletedLine = $logLines | Where-Object { $_ -match 'Deleted files.*?:\s*(\d+)' } | Select-Object -Last 1
        if ($deletedLine -match '(\d+)') {
            $deletedCount = [int]$Matches[1]
        }
    }

    # Also check carved results cache file
    $cacheFile = Join-Path $env:TEMP "carved_results_cache_$TestDrive.bin"
    $cacheInfo = ""
    if (Test-Path $cacheFile) {
        $cacheInfo = " (cache exists: $([math]::Round((Get-Item $cacheFile).Length / 1MB, 1)) MB)"
    }

    $results.carvepool = @{
        durationSeconds = 0
        exitCode = 0
        totalFound = $foundCount
        deletedOnly = $deletedCount
        scanSpeedMBps = $scanSpeed
        filesRecovered = 0
        confidenceDistribution = $confidenceDist
        fromCache = $true
        cacheFile = $cacheFile
    }

    Write-OK "Last scan found: $foundCount files, $deletedCount deleted-only$cacheInfo"
    Write-Info "Scan speed: $scanSpeed MB/s"
}

# ============================================================================
# Phase 4: Recover (Triple Validation)
# ============================================================================
if (-not $SkipRecover) {
    Write-Header "Phase 4: Recover (Triple Validation)"

    $tripleRecoverDir = Join-Path $recoveryDir "triple"
    New-Item -ItemType Directory -Path $tripleRecoverDir -Force | Out-Null

    # 使用真实文件名进行恢复测试（取前5个文件）
    $testFileNames = $testFiles | Select-Object -First 5 | ForEach-Object { $_.fileName }

    Write-Info "Testing triple validation with specific filenames:"
    $testFileNames | ForEach-Object { Write-Host "  $_" }

    $successCount = 0
    $failCount = 0
    $totalDuration = 0

    foreach ($fileName in $testFileNames) {
        $cmd = "recover $TestDrive `"$fileName`" `"$tripleRecoverDir`""
        $startTime = Get-Date
        & $ExePath --cmd $cmd --test 2>$null
        $duration = ((Get-Date) - $startTime).TotalSeconds
        $totalDuration += $duration

        Start-Sleep -Milliseconds 100
        $recoveredFiles = Get-ChildItem $tripleRecoverDir -File -ErrorAction SilentlyContinue
        $newFiles = $recoveredFiles | Where-Object { $_.LastWriteTime -gt $startTime }

        if ($newFiles.Count -gt 0) {
            $successCount++
            Write-Host "  [OK] $fileName -> $($newFiles[0].Name) ($([math]::Round($duration, 1))s)" -ForegroundColor Green
        } else {
            $failCount++
            Write-Host "  [FAIL] $fileName ($([math]::Round($duration, 1))s)" -ForegroundColor Red
        }
    }

    $recoveredCount = (Get-ChildItem $tripleRecoverDir -File -ErrorAction SilentlyContinue | Measure-Object).Count

    $results.recover = @{
        durationSeconds = [math]::Round($totalDuration, 1)
        exitCode = 0
        filesRecovered = $recoveredCount
        successCount = $successCount
        failCount = $failCount
        outputDir = $tripleRecoverDir
    }

    Write-OK "Recovered $successCount / $($testFileNames.Count) files with triple validation"
}

# ============================================================================
# Save Results
# ============================================================================
$results.endTime = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")

$resultsPath = Join-Path $OutputDir "recovery_test_results.json"
$results | ConvertTo-Json -Depth 5 | Out-File $resultsPath -Encoding UTF8

Write-Header "Summary"
Write-OK "Results saved: $resultsPath"
Write-Host "`n--- Recovery Summary ---" -ForegroundColor White
Write-Host ("  {0,-20} {1,8}s" -f "USN List", $results.usnlist.durationSeconds)
Write-Host ("  {0,-20} {1,8}s" -f "USN Recover", $results.usnrecover.durationSeconds)
Write-Host ("  {0,-20} {1,8}s" -f "CarvePool", $results.carvepool.durationSeconds)
Write-Host ("  {0,-20} {1,8}s" -f "Recover (Triple)", $results.recover.durationSeconds)

Write-Host "`n--- Next Step ---" -ForegroundColor White
Write-Host "  Run recovery_test_analyze.ps1 to analyze results and verify integrity"
Write-Host ""

exit 0
