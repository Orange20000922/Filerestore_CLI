# test_recover.ps1 - Automated recovery testing script
# Usage: .\test_recover.ps1 -Drive C -FileName "test.docx" -OutputDir "D:\recovered"

param(
    [Parameter(Mandatory=$true)]
    [string]$Drive,

    [Parameter(Mandatory=$true)]
    [string]$FileName,

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = ".\recovered_test",

    [Parameter(Mandatory=$false)]
    [switch]$Verbose
)

# Configuration
$ExePath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe"
$LogFile = "test_recover_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$ResultsFile = "test_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Test Case Structure
$TestResult = @{
    StartTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Drive = $Drive
    FileName = $FileName
    OutputDir = $OutputDir
    Phases = @{}
    Success = $false
    RecoveredFile = $null
    RecoveredSize = 0
    TotalDurationMs = 0
}

Write-Host "============================================"
Write-Host "Automated Recovery Test"
Write-Host "============================================"
Write-Host "Drive: $Drive"
Write-Host "FileName: $FileName"
Write-Host "Output: $OutputDir"
Write-Host ""

# Function to run command and capture output
function Invoke-RecoverTest {
    param(
        [string]$Drive,
        [string]$FileName,
        [string]$OutputDir
    )

    $startTime = Get-Date

    # Build command input (simulating user interaction)
    # recover <drive> <filename> <output_dir>
    $cmdArgs = "recover $Drive `"$FileName`" `"$OutputDir`""

    Write-Host "[TEST] Running: $cmdArgs"

    # Run the command
    $process = Start-Process -FilePath $ExePath -ArgumentList $cmdArgs -NoNewWindow -PassThru -Wait -RedirectStandardOutput "$OutputDir\stdout.txt" -RedirectStandardError "$OutputDir\stderr.txt"

    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalMilliseconds

    # Read output
    $stdout = ""
    $stderr = ""
    if (Test-Path "$OutputDir\stdout.txt") {
        $stdout = Get-Content "$OutputDir\stdout.txt" -Raw
    }
    if (Test-Path "$OutputDir\stderr.txt") {
        $stderr = Get-Content "$OutputDir\stderr.txt" -Raw
    }

    return @{
        ExitCode = $process.ExitCode
        Duration = $duration
        Stdout = $stdout
        Stderr = $stderr
    }
}

# Function to parse recovery output
function Parse-RecoveryOutput {
    param([string]$Output)

    $result = @{
        UsnRecordsFound = 0
        MftInfoEnriched = 0
        CandidatesFound = 0
        TripleValidation = 0
        DoubleValidation = 0
        SignatureOnly = 0
        RecoverySuccess = $false
        RecoveredSize = 0
    }

    # Parse USN records
    if ($Output -match "找到\s+(\d+)\s+条 USN 删除记录") {
        $result.UsnRecordsFound = [int]$Matches[1]
    }

    # Parse MFT enrichment
    if ($Output -match "成功获取\s+(\d+)\s+个文件的大小信息") {
        $result.MftInfoEnriched = [int]$Matches[1]
    }

    # Parse candidates
    if ($Output -match "找到\s+(\d+)\s+个候选文件") {
        $result.CandidatesFound = [int]$Matches[1]
    }

    # Parse validation results
    if ($Output -match "三角验证通过:\s+(\d+)") {
        $result.TripleValidation = [int]$Matches[1]
    }
    if ($Output -match "双重验证通过:\s+(\d+)") {
        $result.DoubleValidation = [int]$Matches[1]
    }
    if ($Output -match "仅签名验证:\s+(\d+)") {
        $result.SignatureOnly = [int]$Matches[1]
    }

    # Parse recovery result
    if ($Output -match "恢复成功") {
        $result.RecoverySuccess = $true
    }
    if ($Output -match "文件大小:\s+(\d+)\s+bytes") {
        $result.RecoveredSize = [int64]$Matches[1]
    }

    return $result
}

# Run the test
$startTotal = Get-Date

Write-Host ""
Write-Host "[PHASE 1] Running recover command..."
$execResult = Invoke-RecoverTest -Drive $Drive -FileName $FileName -OutputDir $OutputDir

$TestResult.TotalDurationMs = $execResult.Duration
$TestResult.ExitCode = $execResult.ExitCode

# Parse output
Write-Host "[PHASE 2] Parsing results..."
$parsed = Parse-RecoveryOutput -Output $execResult.Stdout

$TestResult.Phases = $parsed
$TestResult.Success = $parsed.RecoverySuccess
$TestResult.RecoveredSize = $parsed.RecoveredSize

# Check for recovered file
$recoveredFiles = Get-ChildItem -Path $OutputDir -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*$FileName*" -or $_.Extension -ne ".txt" }
if ($recoveredFiles) {
    $TestResult.RecoveredFile = $recoveredFiles[0].FullName
    $TestResult.RecoveredSize = $recoveredFiles[0].Length
}

$endTotal = Get-Date
$TestResult.EndTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Display results
Write-Host ""
Write-Host "============================================"
Write-Host "Test Results"
Write-Host "============================================"
Write-Host "Duration: $([math]::Round($TestResult.TotalDurationMs / 1000, 2)) seconds"
Write-Host ""
Write-Host "Phase Statistics:"
Write-Host "  USN Records Found:     $($parsed.UsnRecordsFound)"
Write-Host "  MFT Info Enriched:     $($parsed.MftInfoEnriched)"
Write-Host "  Candidates Found:      $($parsed.CandidatesFound)"
Write-Host "  Triple Validation:     $($parsed.TripleValidation)"
Write-Host "  Double Validation:     $($parsed.DoubleValidation)"
Write-Host "  Signature Only:        $($parsed.SignatureOnly)"
Write-Host ""
Write-Host "Recovery Result:"
if ($TestResult.Success) {
    Write-Host "  Status: SUCCESS" -ForegroundColor Green
    Write-Host "  File: $($TestResult.RecoveredFile)"
    Write-Host "  Size: $($TestResult.RecoveredSize) bytes"
} else {
    Write-Host "  Status: FAILED" -ForegroundColor Red
}

# Save results to JSON
$TestResult | ConvertTo-Json -Depth 5 | Out-File $ResultsFile -Encoding UTF8

Write-Host ""
Write-Host "Results saved to: $ResultsFile"
Write-Host "Full output saved to: $OutputDir\stdout.txt"

# Verbose output
if ($Verbose) {
    Write-Host ""
    Write-Host "============================================"
    Write-Host "Full Command Output"
    Write-Host "============================================"
    Write-Host $execResult.Stdout
}

# Return test result
return $TestResult
