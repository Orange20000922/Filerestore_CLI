#!/usr/bin/env pwsh
# ============================================================================
# Recovery Performance Test - Main Entry Point
# ============================================================================
# Orchestrates the full recovery performance testing workflow:
#   1. Create test files (various types/sizes)
#   2. Delete test files
#   3. Run recovery tests (usnlist, usnrecover, carvepool, recover)
#   4. Analyze results and generate report
#
# Usage:
#   .\recovery_performance_test.ps1
#   .\recovery_performance_test.ps1 -TestDrive D -TestFileCount 100
#   .\recovery_performance_test.ps1 -TestDrive D -SkipCreate  # Use existing test data
# ============================================================================

param(
    [string]$ExePath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe",
    [string]$LogPath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\debug.log",
    [string]$TestDrive = "D",
    [string]$OutputDir = "D:\recovery_test_results",
    [int]$TestFileCount = 50,
    [int]$WaitBeforeScanMinutes = 5,  # 改为分钟，默认5分钟等待USN刷新
    [switch]$SkipCreate,
    [switch]$SkipScan,
    [switch]$SkipAnalyze,
    [switch]$KeepRecovered,
    [switch]$QuickMode  # 快速模式：跳过等待，同时测试USN已有文件
)

$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Header($msg) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

function Write-Info($msg)  { Write-Host "[INFO] $msg" -ForegroundColor Yellow }
function Write-OK($msg)    { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Err($msg)   { Write-Host "[ERR]  $msg" -ForegroundColor Red }

# ============================================================================
# Main
# ============================================================================
$overallStart = Get-Date

Write-Header "Recovery Performance Test Suite"
Write-Info "Test Drive: $TestDrive"
Write-Info "Output Dir: $OutputDir"
Write-Info "File Count: $TestFileCount"
Write-Info "Executable: $ExePath"

# Validate executable
if (-not (Test-Path $ExePath)) {
    Write-Err "Executable not found: $ExePath"
    Write-Info "Build Release x64 first:"
    Write-Info "  powershell -Command `"& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' 'D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI\Filerestore_CLI.vcxproj' /p:Configuration=Release /p:Platform=x64`""
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# ============================================================================
# Phase 1: Create Test Files
# ============================================================================
if (-not $SkipCreate) {
    Write-Header "Phase 1: Creating Test Files"

    $createScript = Join-Path $scriptDir "recovery_test_create.ps1"
    if (-not (Test-Path $createScript)) {
        Write-Err "Create script not found: $createScript"
        exit 1
    }

    & $createScript -TestDrive $TestDrive -OutputDir $OutputDir -TestFileCount $TestFileCount
    # Check if metadata file was created
    $metadataPath = Join-Path $OutputDir "test_files_metadata.json"
    if (-not (Test-Path $metadataPath)) {
        Write-Err "Test file creation failed - metadata not found"
        exit 1
    }

    # Wait before scanning (USN journal needs time to register deletions)
    if ($QuickMode) {
        Write-Info "Quick mode: Skipping wait (USN may not have recent files)"
    } elseif ($WaitBeforeScanMinutes -gt 0) {
        $waitSeconds = $WaitBeforeScanMinutes * 60
        Write-Info "Waiting $WaitBeforeScanMinutes minutes for USN journal to update..."
        Write-Info "(This ensures deleted files appear in USN records)"

        $startTime = Get-Date
        $endTime = $startTime.AddSeconds($waitSeconds)

        while ((Get-Date) -lt $endTime) {
            $remaining = ($endTime - (Get-Date)).TotalSeconds
            if ($remaining -gt 60) {
                Write-Host "  Remaining: $([math]::Floor($remaining / 60)) min $([math]::Floor($remaining % 60)) sec" -NoNewline -ForegroundColor Gray
                Write-Host "`r" -NoNewline
            } else {
                Write-Host "  Remaining: $([math]::Floor($remaining)) sec    " -NoNewline -ForegroundColor Gray
                Write-Host "`r" -NoNewline
            }
            Start-Sleep -Seconds 5
        }
        Write-Host ""
        Write-OK "Wait complete"
    }
} else {
    Write-Info "Skipping test file creation (--SkipCreate)"
}

# ============================================================================
# Phase 2: Test Existing USN Files (different time ranges)
# ============================================================================
if (-not $SkipScan -and -not $QuickMode) {
    Write-Header "Phase 2a: Testing Existing USN Files"

    Write-Info "Analyzing existing deleted files in USN journal..."

    # Run usnlist for different time ranges
    $timeRanges = @(
        @{ Hours = 1;  Label = "1h" }
        @{ Hours = 24; Label = "24h" }
        @{ Hours = 168; Label = "7d" }
    )

    $usnStats = @{}

    foreach ($range in $timeRanges) {
        $cmd = "usnlist $TestDrive $($range.Hours) --validate"
        Write-Info "Running: $cmd"

        & $ExePath --cmd $cmd --test 2>$null

        # Parse log for this time range
        Start-Sleep -Milliseconds 500
        $logLines = Get-Content $LogPath -Encoding UTF8 -Tail 2000
        $outputLines = $logLines | Where-Object { $_ -match '\[OUTPUT\]' }

        $statusCounts = @{}
        $outputLines | ForEach-Object {
            if ($_ -match '\[(OK|SUCCESS|MFT_REUSED|NO_DATA|RESIDENT|SIG_MISMATCH)\]') {
                $status = $Matches[1]
                if (-not $statusCounts.ContainsKey($status)) { $statusCounts[$status] = 0 }
                $statusCounts[$status]++
            }
        }

        $total = ($statusCounts.Values | Measure-Object -Sum).Sum
        $ok = ($statusCounts.Keys | Where-Object { $_ -in @("OK", "SUCCESS", "RESIDENT") } | ForEach-Object { $statusCounts[$_] } | Measure-Object -Sum).Sum
        $rate = if ($total -gt 0) { [math]::Round($ok / $total * 100, 1) } else { 0 }

        $usnStats[$range.Label] = @{
            total = $total
            ok = $ok
            rate = $rate
            statusCounts = $statusCounts
        }

        Write-Host "  $($range.Label): $total files, $ok recoverable ($rate%)" -ForegroundColor $(if ($rate -ge 50) { "Green" } else { "Yellow" })
    }

    # Save USN stats
    $usnStatsPath = Join-Path $OutputDir "usn_existing_stats.json"
    $usnStats | ConvertTo-Json -Depth 3 | Out-File $usnStatsPath -Encoding UTF8
    Write-OK "USN stats saved: $usnStatsPath"
}

# ============================================================================
# Phase 3: Run Recovery Tests (on newly created files)
# ============================================================================
if (-not $SkipScan) {
    Write-Header "Phase 3: Running Recovery Tests (New Files)"

    $runScript = Join-Path $scriptDir "recovery_test_run.ps1"
    if (-not (Test-Path $runScript)) {
        Write-Err "Run script not found: $runScript"
        exit 1
    }

    & $runScript -ExePath $ExePath -LogPath $LogPath -TestDrive $TestDrive -OutputDir $OutputDir
} else {
    Write-Info "Skipping recovery tests (--SkipScan)"
}

# ============================================================================
# Phase 4: Analyze Results
# ============================================================================
if (-not $SkipAnalyze) {
    Write-Header "Phase 4: Analyzing Results"

    $analyzeScript = Join-Path $scriptDir "recovery_test_analyze.ps1"
    if (-not (Test-Path $analyzeScript)) {
        Write-Err "Analyze script not found: $analyzeScript"
        exit 1
    }

    & $analyzeScript -OutputDir $OutputDir -LogPath $LogPath
} else {
    Write-Info "Skipping analysis (--SkipAnalyze)"
}

# ============================================================================
# Cleanup
# ============================================================================
if (-not $KeepRecovered) {
    $recoveredDir = Join-Path $OutputDir "recovered"
    if (Test-Path $recoveredDir) {
        Write-Info "Cleaning up recovered files..."
        Remove-Item $recoveredDir -Recurse -Force
    }
}

# ============================================================================
# Final Summary
# ============================================================================
$overallEnd = Get-Date
$totalDuration = ($overallEnd - $overallStart).TotalSeconds

Write-Header "Test Suite Complete"
Write-OK "Total duration: $([math]::Round($totalDuration, 1)) seconds"

$reportPath = Join-Path $OutputDir "recovery_performance_report.json"
if (Test-Path $reportPath) {
    Write-OK "Report saved: $reportPath"

    # Show quick summary from report
    $report = Get-Content $reportPath -Raw | ConvertFrom-Json
    Write-Host ""
    Write-Host "=== Quick Summary ===" -ForegroundColor White
    Write-Host ("  Recovery Rate:      {0}%" -f $report.conclusions.shortTermRecoveryRate)
    Write-Host ("  Recommended:        {0}" -f $report.conclusions.recommendedStrategy)
    Write-Host ("  MFT-guided viable:  {0}" -f $report.conclusions.mftGuidedViable)
    Write-Host ""
} else {
    Write-Host "[WARN] Report file not found: $reportPath" -ForegroundColor Magenta
}

Write-Host "Files generated:" -ForegroundColor White
Get-ChildItem $OutputDir -File | ForEach-Object {
    Write-Host ("  {0,-40} {1,10:N0} bytes" -f $_.Name, $_.Length)
}

Write-Host ""
