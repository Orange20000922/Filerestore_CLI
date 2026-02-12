#!/usr/bin/env pwsh
# ============================================================================
# Recovery Performance Test - Result Analysis
# ============================================================================
# Analyzes recovery results and generates comprehensive JSON report.
# Verifies recovered file integrity using MD5 checksums.
#
# Usage:
#   .\recovery_test_analyze.ps1
#   .\recovery_test_analyze.ps1 -OutputDir "D:\recovery_test_results"
# ============================================================================

param(
    [string]$OutputDir = "D:\recovery_test_results",
    [string]$LogPath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\debug.log",
    [string]$MetadataPath = $null,
    [string]$ResultsPath = $null
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
function Write-Warn($msg)  { Write-Host "[WARN] $msg" -ForegroundColor Magenta }

# Auto-detect paths
if (-not $MetadataPath) {
    $MetadataPath = Join-Path $OutputDir "test_files_metadata.json"
}
if (-not $ResultsPath) {
    $ResultsPath = Join-Path $OutputDir "recovery_test_results.json"
}

Write-Header "Recovery Test - Analyzing Results"
Write-Info "Output Dir: $OutputDir"
Write-Info "Metadata: $MetadataPath"
Write-Info "Results: $ResultsPath"

# Validate input files
if (-not (Test-Path $MetadataPath)) {
    Write-Err "Metadata file not found: $MetadataPath"
    exit 1
}
if (-not (Test-Path $ResultsPath)) {
    Write-Err "Results file not found: $ResultsPath"
    exit 1
}

# Load data
$metadata = Get-Content $MetadataPath -Raw | ConvertFrom-Json
$rawResults = Get-Content $ResultsPath -Raw | ConvertFrom-Json

$testFiles = $metadata.files
$deleteTime = [DateTime]::Parse($metadata.deleteTime)
$createTime = [DateTime]::Parse($metadata.createTime)

Write-Info "Loaded $($testFiles.Count) test file records"
Write-Info "Delete time: $($deleteTime.ToString('yyyy-MM-dd HH:mm:ss'))"

# ============================================================================
# Build test file lookup (filename -> metadata)
# ============================================================================
$testFileLookup = @{}
foreach ($tf in $testFiles) {
    $testFileLookup[$tf.fileName] = $tf
    # Also add without timestamp suffix for pattern matching
    if ($tf.fileName -match '^(test_\d{4})_') {
        $baseName = $Matches[1]
        if (-not $testFileLookup.ContainsKey($baseName)) {
            $testFileLookup[$baseName] = $tf
        }
    }
}

# ============================================================================
# Initialize report structure
# ============================================================================
$report = @{
    testInfo = @{
        timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
        testDrive = $rawResults.testDrive
        testFileCount = $rawResults.testFileCount
        deleteAgeMinutes = $rawResults.deleteAgeMinutes
        durationSeconds = 0
    }
    testDataProfile = @{
        byType = $metadata.byType
        bySize = $metadata.bySize
        byDeleteAge = @{ "0-5min" = $testFiles.Count }  # All deleted around same time
    }
    recoveryResults = @{
        usnRecover = @{
            totalAttempted = $testFiles.Count
            success = @{}
            failure = @{}
            recoveryRate = 0
            avgConfidence = 0
            integrityCheck = @{ passed = 0; failed = 0; notChecked = 0 }
        }
        carvePool = @{
            totalFound = 0
            deletedOnly = 0
            recoveryRate = 0
            confidenceDistribution = @{ high = 0; medium = 0; low = 0 }
            integrityCheck = @{ passed = 0; failed = 0; notChecked = 0 }
        }
        tripleValidation = @{
            VAL_TRIPLE = 0
            VAL_MFT_SIGNATURE = 0
            VAL_USN_SIGNATURE = 0
            VAL_USN_MFT = 0
            VAL_SIGNATURE_ONLY = 0
            integrityCheck = @{ passed = 0; failed = 0; notChecked = 0 }
        }
    }
    performanceMetrics = @{
        usnListDurationMs = 0
        usnRecoverDurationMs = 0
        carvePoolDurationMs = 0
        tripleRecoverDurationMs = 0
    }
    fileIntegrity = @{
        verified = @()
        mismatches = @()
    }
    conclusions = @{
        shortTermRecoveryRate = 0
        recommendedStrategy = ""
        mftGuidedViable = $false
        confidenceDistribution = "unknown"
    }
}

# ============================================================================
# Analyze USN Recover Results
# ============================================================================
Write-Header "Analyzing USN Recover"

$usnRecoverDir = Join-Path $OutputDir "recovered\usnrecover"
$usnStatusCounts = $rawResults.usnrecover.statusCounts

if ($usnStatusCounts) {
    $successStatuses = @("SUCCESS", "RESIDENT", "MFT_REUSED_OK", "PARTIAL")
    $failureStatuses = @("MFT_REUSED", "NO_DATA", "SIG_MISMATCH", "READ_ERROR", "WRITE_ERROR")

    $successCount = 0
    $failureCount = 0

    foreach ($key in $usnStatusCounts.Keys) {
        if ($successStatuses -contains $key) {
            $report.recoveryResults.usnRecover.success[$key] = $usnStatusCounts[$key]
            $successCount += $usnStatusCounts[$key]
        } elseif ($failureStatuses -contains $key) {
            $report.recoveryResults.usnRecover.failure[$key] = $usnStatusCounts[$key]
            $failureCount += $usnStatusCounts[$key]
        }
    }

    $total = $successCount + $failureCount
    if ($total -gt 0) {
        $report.recoveryResults.usnRecover.recoveryRate = [math]::Round($successCount / $total * 100, 1)
    }

    Write-Info "USN Recover: $successCount success, $failureCount failure"
    Write-OK "Recovery rate: $($report.recoveryResults.usnRecover.recoveryRate)%"

    # Verify integrity of recovered files
    if (Test-Path $usnRecoverDir) {
        $recoveredFiles = Get-ChildItem $usnRecoverDir -File
        foreach ($rf in $recoveredFiles) {
            $matched = $false
            foreach ($tf in $testFiles) {
                if ($rf.Name -match [regex]::Escape($tf.fileName) -or
                    $rf.Name -match [regex]::Escape($tf.fileName -replace '\.[^.]+$', '')) {
                    $matched = $true

                    # Calculate MD5
                    try {
                        $md5 = [System.Security.Cryptography.MD5]::Create()
                        $bytes = [System.IO.File]::ReadAllBytes($rf.FullName)
                        $hash = $md5.ComputeHash($bytes)
                        $actualMd5 = [BitConverter]::ToString($hash).Replace("-", "").ToLower()

                        if ($actualMd5 -eq $tf.md5) {
                            $report.recoveryResults.usnRecover.integrityCheck.passed++
                            $report.fileIntegrity.verified += @{
                                fileName = $tf.fileName
                                method = "usnrecover"
                                originalSize = $tf.size
                                recoveredSize = $rf.Length
                                md5Match = $true
                            }
                        } else {
                            $report.recoveryResults.usnRecover.integrityCheck.failed++
                            $report.fileIntegrity.mismatches += @{
                                fileName = $tf.fileName
                                method = "usnrecover"
                                expectedMd5 = $tf.md5
                                actualMd5 = $actualMd5
                            }
                        }
                    } catch {
                        $report.recoveryResults.usnRecover.integrityCheck.notChecked++
                    }
                    break
                }
            }
        }
    }
}

# ============================================================================
# Analyze CarvePool Results
# ============================================================================
Write-Header "Analyzing CarvePool"

$carvePoolDir = Join-Path $OutputDir "recovered\carvepool"

$report.recoveryResults.carvePool.totalFound = $rawResults.carvepool.totalFound
$report.recoveryResults.carvePool.deletedOnly = $rawResults.carvepool.deletedOnly
$report.recoveryResults.carvePool.confidenceDistribution = $rawResults.carvepool.confidenceDistribution

if ($rawResults.carvepool.totalFound -gt 0) {
    $report.recoveryResults.carvePool.recoveryRate = [math]::Round(
        $rawResults.carvepool.deletedOnly / $rawResults.carvepool.totalFound * 100, 1)

    Write-Info "CarvePool: $($rawResults.carvepool.totalFound) found, $($rawResults.carvepool.deletedOnly) deleted-only"
    Write-OK "Deletion filter rate: $($report.recoveryResults.carvePool.recoveryRate)%"
}

# Verify integrity
if (Test-Path $carvePoolDir) {
    $recoveredFiles = Get-ChildItem $carvePoolDir -File
    foreach ($rf in $recoveredFiles) {
        foreach ($tf in $testFiles) {
            if ($rf.Name -match [regex]::Escape($tf.fileName) -or
                $rf.Name -match [regex]::Escape($tf.fileName -replace '\.[^.]+$', '')) {
                try {
                    $md5 = [System.Security.Cryptography.MD5]::Create()
                    $bytes = [System.IO.File]::ReadAllBytes($rf.FullName)
                    $hash = $md5.ComputeHash($bytes)
                    $actualMd5 = [BitConverter]::ToString($hash).Replace("-", "").ToLower()

                    if ($actualMd5 -eq $tf.md5) {
                        $report.recoveryResults.carvePool.integrityCheck.passed++
                    } else {
                        $report.recoveryResults.carvePool.integrityCheck.failed++
                    }
                } catch {
                    $report.recoveryResults.carvePool.integrityCheck.notChecked++
                }
                break
            }
        }
    }
}

# ============================================================================
# Analyze Triple Validation Results
# ============================================================================
Write-Header "Analyzing Triple Validation"

$tripleDir = Join-Path $OutputDir "recovered\triple"
$validationDist = $rawResults.recover.validationDistribution

if ($validationDist) {
    $report.recoveryResults.tripleValidation.VAL_TRIPLE = $validationDist.TRIPLE
    $report.recoveryResults.tripleValidation.VAL_MFT_SIGNATURE = $validationDist.MFT_SIGNATURE
    $report.recoveryResults.tripleValidation.VAL_USN_SIGNATURE = $validationDist.USN_SIGNATURE
    $report.recoveryResults.tripleValidation.VAL_USN_MFT = $validationDist.USN_MFT
    $report.recoveryResults.tripleValidation.VAL_SIGNATURE_ONLY = $validationDist.SIGNATURE_ONLY

    $total = $validationDist.TRIPLE + $validationDist.MFT_SIGNATURE + $validationDist.USN_SIGNATURE +
             $validationDist.USN_MFT + $validationDist.SIGNATURE_ONLY

    Write-Info "Triple validation total: $total"
    Write-Host "  TRIPLE:        $($validationDist.TRIPLE)" -ForegroundColor Green
    Write-Host "  MFT+Signature: $($validationDist.MFT_SIGNATURE)" -ForegroundColor Green
    Write-Host "  USN+Signature: $($validationDist.USN_SIGNATURE)" -ForegroundColor Yellow
    Write-Host "  USN+MFT:       $($validationDist.USN_MFT)" -ForegroundColor Yellow
    Write-Host "  Signature:     $($validationDist.SIGNATURE_ONLY)" -ForegroundColor Gray
}

# Verify integrity
if (Test-Path $tripleDir) {
    $recoveredFiles = Get-ChildItem $tripleDir -File
    foreach ($rf in $recoveredFiles) {
        foreach ($tf in $testFiles) {
            if ($rf.Name -match [regex]::Escape($tf.fileName) -or
                $rf.Name -match [regex]::Escape($tf.fileName -replace '\.[^.]+$', '')) {
                try {
                    $md5 = [System.Security.Cryptography.MD5]::Create()
                    $bytes = [System.IO.File]::ReadAllBytes($rf.FullName)
                    $hash = $md5.ComputeHash($bytes)
                    $actualMd5 = [BitConverter]::ToString($hash).Replace("-", "").ToLower()

                    if ($actualMd5 -eq $tf.md5) {
                        $report.recoveryResults.tripleValidation.integrityCheck.passed++
                    } else {
                        $report.recoveryResults.tripleValidation.integrityCheck.failed++
                    }
                } catch {
                    $report.recoveryResults.tripleValidation.integrityCheck.notChecked++
                }
                break
            }
        }
    }
}

# ============================================================================
# Performance Metrics
# ============================================================================
$report.performanceMetrics.usnListDurationMs = [math]::Round($rawResults.usnlist.durationSeconds * 1000)
$report.performanceMetrics.usnRecoverDurationMs = [math]::Round($rawResults.usnrecover.durationSeconds * 1000)
$report.performanceMetrics.carvePoolDurationMs = [math]::Round($rawResults.carvepool.durationSeconds * 1000)
$report.performanceMetrics.tripleRecoverDurationMs = [math]::Round($rawResults.recover.durationSeconds * 1000)

$totalDuration = $rawResults.usnlist.durationSeconds + $rawResults.usnrecover.durationSeconds +
                 $rawResults.carvepool.durationSeconds + $rawResults.recover.durationSeconds
$report.testInfo.durationSeconds = [math]::Round($totalDuration, 1)

# ============================================================================
# Conclusions
# ============================================================================
Write-Header "Generating Conclusions"

$usnRate = $report.recoveryResults.usnRecover.recoveryRate
$report.conclusions.shortTermRecoveryRate = $usnRate

# Determine recommended strategy
if ($usnRate -ge 80) {
    $report.conclusions.recommendedStrategy = "USN+MFT first, signature scan fallback"
    $report.conclusions.mftGuidedViable = $true
    Write-OK "MFT-guided scan: HIGH value ($usnRate pct recoverable)"
} elseif ($usnRate -ge 50) {
    $report.conclusions.recommendedStrategy = "USN+MFT parallel with signature scan"
    $report.conclusions.mftGuidedViable = $true
    Write-Host "[WARN] MFT-guided scan: MODERATE value ($usnRate pct recoverable)" -ForegroundColor Magenta
} else {
    $report.conclusions.recommendedStrategy = "Signature scan primary, MFT validation"
    $report.conclusions.mftGuidedViable = $false
    Write-Err "MFT-guided scan: LOW value ($usnRate pct recoverable)"
}

# Confidence distribution conclusion
$conf = $report.recoveryResults.carvePool.confidenceDistribution
if ($conf.high -gt $conf.medium -and $conf.high -gt $conf.low) {
    $report.conclusions.confidenceDistribution = "high-dominant"
} elseif ($conf.medium -gt $conf.low) {
    $report.conclusions.confidenceDistribution = "medium-dominant"
} else {
    $report.conclusions.confidenceDistribution = "low-dominant"
}

# ============================================================================
# Save Final Report
# ============================================================================
$reportPath = Join-Path $OutputDir "recovery_performance_report.json"
$report | ConvertTo-Json -Depth 6 | Out-File $reportPath -Encoding UTF8

Write-Header "Analysis Complete"
Write-OK "Report saved: $reportPath"

# Summary output
Write-Host "`n=== Recovery Performance Summary ===" -ForegroundColor White
Write-Host ""
Write-Host "Test Parameters:" -ForegroundColor Yellow
Write-Host ("  Files created:      {0}" -f $report.testInfo.testFileCount)
Write-Host ("  Delete age:         {0} minutes" -f $report.testInfo.deleteAgeMinutes)
Write-Host ("  Total duration:     {0} seconds" -f $report.testInfo.durationSeconds)
Write-Host ""
Write-Host "Recovery Results:" -ForegroundColor Yellow
Write-Host ("  USN Recover rate:   {0}%" -f $report.recoveryResults.usnRecover.recoveryRate)
Write-Host ("  CarvePool found:    {0} ({1} deleted)" -f $report.recoveryResults.carvePool.totalFound, $report.recoveryResults.carvePool.deletedOnly)
Write-Host ("  Triple validation:  {0} TRIPLE, {1} MFT+SIG" -f $report.recoveryResults.tripleValidation.VAL_TRIPLE, $report.recoveryResults.tripleValidation.VAL_MFT_SIGNATURE)
Write-Host ""
Write-Host "Integrity Check:" -ForegroundColor Yellow
$usnIntegrity = $report.recoveryResults.usnRecover.integrityCheck
$carveIntegrity = $report.recoveryResults.carvePool.integrityCheck
$tripleIntegrity = $report.recoveryResults.tripleValidation.integrityCheck
Write-Host ("  USN Recover:        {0} passed, {1} failed" -f $usnIntegrity.passed, $usnIntegrity.failed)
Write-Host ("  CarvePool:          {0} passed, {1} failed" -f $carveIntegrity.passed, $carveIntegrity.failed)
Write-Host ("  Triple Validation:  {0} passed, {1} failed" -f $tripleIntegrity.passed, $tripleIntegrity.failed)
Write-Host ""
Write-Host "Conclusion:" -ForegroundColor Green
Write-Host ("  Strategy:           {0}" -f $report.conclusions.recommendedStrategy)
Write-Host ("  MFT-guided viable:  {0}" -f $report.conclusions.mftGuidedViable)
Write-Host ""

exit 0
