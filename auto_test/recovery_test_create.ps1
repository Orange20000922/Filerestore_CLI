#!/usr/bin/env pwsh
# ============================================================================
# Recovery Performance Test - Test Data Creation
# ============================================================================
# Creates test files of various types and sizes, then deletes them.
# Records metadata for verification.
#
# Usage:
#   .\recovery_test_create.ps1
#   .\recovery_test_create.ps1 -TestDrive D -TestFileCount 100
# ============================================================================

param(
    [string]$TestDrive = "D",
    [string]$OutputDir = "D:\recovery_test_results",
    [int]$TestFileCount = 50
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

# File type configurations
$FileTypes = @(
    @{ Ext = "zip";  Header = $null; MinSize = 1024;      MaxSize = 512000   }
    @{ Ext = "pdf";  Header = "%PDF-1.4"; MinSize = 2048; MaxSize = 204800   }
    @{ Ext = "jpg";  Header = $null; MinSize = 51200;     MaxSize = 307200   }
    @{ Ext = "png";  Header = $null; MinSize = 10240;     MaxSize = 102400   }
    @{ Ext = "docx"; Header = "PK";   MinSize = 4096;     MaxSize = 102400   }
    @{ Ext = "txt";  Header = $null; MinSize = 100;       MaxSize = 5000     }
    @{ Ext = "exe";  Header = "MZ";   MinSize = 65536;    MaxSize = 262144   }
)

# Size categories for reporting
function Get-SizeCategory($size) {
    if ($size -lt 900) { return "resident" }
    if ($size -lt 10240) { return "small" }
    if ($size -lt 102400) { return "medium" }
    return "large"
}

Write-Header "Recovery Test - Creating Test Files"
Write-Info "Test Drive: $TestDrive"
Write-Info "Output Dir: $OutputDir"
Write-Info "File Count: $TestFileCount"

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Create test directory on target drive
$testDir = "${TestDrive}:\__recovery_test_tmp__"
if (Test-Path $testDir) {
    Remove-Item $testDir -Recurse -Force
}
New-Item -ItemType Directory -Path $testDir -Force | Out-Null

Write-Info "Test directory: $testDir"

# Initialize test file metadata
$testFiles = @()
$rng = New-Object System.Random

# Create test files
Write-Info "Creating $TestFileCount test files..."

for ($i = 0; $i -lt $TestFileCount; $i++) {
    $fileType = $FileTypes[$i % $FileTypes.Count]

    # Random size within range
    $sizeRange = $fileType.MaxSize - $fileType.MinSize
    $fileSize = $fileType.MinSize + $rng.Next($sizeRange)

    # Generate unique filename with timestamp
    $timestamp = Get-Date -Format "HHmmss_fff"
    $fileName = "test_{0:D4}_{1}.{2}" -f $i, $timestamp, $fileType.Ext
    $filePath = Join-Path $testDir $fileName

    # Create file content
    $buffer = New-Object byte[] $fileSize
    $rng.NextBytes($buffer)

    # Write file header if specified
    if ($fileType.Header) {
        $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($fileType.Header)
        [Array]::Copy($headerBytes, $buffer, [Math]::Min($headerBytes.Length, $buffer.Length))

        # Special handling for PNG header
        if ($fileType.Ext -eq "png") {
            $pngHeader = [byte[]]@(0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
            [Array]::Copy($pngHeader, $buffer, 8)
        }
        # Special handling for JPG header
        if ($fileType.Ext -eq "jpg") {
            $buffer[0] = 0xFF; $buffer[1] = 0xD8; $buffer[2] = 0xFF; $buffer[3] = 0xE0
        }
    }

    # Write file
    [System.IO.File]::WriteAllBytes($filePath, $buffer)

    # Record metadata
    $fileInfo = Get-Item $filePath
    $testFiles += [PSCustomObject]@{
        index = $i
        fileName = $fileName
        filePath = $filePath
        size = $fileSize
        sizeCategory = Get-SizeCategory $fileSize
        extension = $fileType.Ext
        createTime = $fileInfo.CreationTime
        md5 = $null  # Will be calculated after creation
    }

    if (($i + 1) % 10 -eq 0) {
        Write-Info "  Created $($i + 1) / $TestFileCount files..."
    }
}

# Calculate MD5 for all files
Write-Info "Calculating MD5 checksums..."
foreach ($tf in $testFiles) {
    try {
        $md5 = [System.Security.Cryptography.MD5]::Create()
        $bytes = [System.IO.File]::ReadAllBytes($tf.filePath)
        $hash = $md5.ComputeHash($bytes)
        $tf.md5 = [BitConverter]::ToString($hash).Replace("-", "").ToLower()
    }
    catch {
        $tf.md5 = "error"
    }
}

# Show summary
Write-Header "Test Files Created"

$byType = $testFiles | Group-Object extension | Sort-Object Name
$bySize = $testFiles | Group-Object sizeCategory | Sort-Object Name

Write-Host "`n--- By Type ---" -ForegroundColor White
foreach ($g in $byType) {
    Write-Host ("  {0,-8} {1,4} files" -f $g.Name, $g.Count)
}

Write-Host "`n--- By Size ---" -ForegroundColor White
foreach ($g in $bySize) {
    Write-Host ("  {0,-10} {1,4} files" -f $g.Name, $g.Count)
}

Write-Host "`n--- Sample Files ---" -ForegroundColor White
$testFiles | Select-Object -First 5 | ForEach-Object {
    Write-Host ("  {0,-25} {1,10:N0} bytes [{2}]" -f $_.fileName, $_.size, $_.sizeCategory)
}

# Save metadata before deletion
$metadataPath = Join-Path $OutputDir "test_files_metadata.json"
$metadata = @{
    createTime = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    testDrive = $TestDrive
    testDirectory = $testDir
    totalFiles = $testFiles.Count
    byType = @{}
    bySize = @{}
    files = $testFiles | ForEach-Object {
        @{
            index = $_.index
            fileName = $_.fileName
            size = $_.size
            sizeCategory = $_.sizeCategory
            extension = $_.extension
            createTime = $_.createTime.ToString("yyyy-MM-dd HH:mm:ss")
            md5 = $_.md5
        }
    }
}

# Populate summary stats
foreach ($g in $byType) { $metadata.byType[$g.Name] = $g.Count }
foreach ($g in $bySize) { $metadata.bySize[$g.Name] = $g.Count }

$metadata | ConvertTo-Json -Depth 5 | Out-File $metadataPath -Encoding UTF8
Write-OK "Metadata saved: $metadataPath"

# Now delete all files
Write-Header "Deleting Test Files"

Start-Sleep -Milliseconds 500  # Small delay to ensure filesystem sync

$deleteStartTime = Get-Date

foreach ($tf in $testFiles) {
    Remove-Item $tf.filePath -Force -ErrorAction SilentlyContinue
    $tf | Add-Member -MemberType NoteProperty -Name "deleteTime" -Value $deleteStartTime -Force
}

# Remove directory
Remove-Item $testDir -Force -ErrorAction SilentlyContinue

$deleteEndTime = Get-Date
Write-OK "All $($testFiles.Count) files deleted"
Write-Info "Delete time: $($deleteStartTime.ToString('HH:mm:ss'))"
Write-Info "Duration: $(($deleteEndTime - $deleteStartTime).TotalSeconds.ToString('F2'))s"

# Update metadata with delete times
$metadata.deleteTime = $deleteStartTime.ToString("yyyy-MM-dd HH:mm:ss")
$metadata.deleteDurationSeconds = ($deleteEndTime - $deleteStartTime).TotalSeconds

# Re-save metadata
$metadata | ConvertTo-Json -Depth 5 | Out-File $metadataPath -Encoding UTF8

Write-Host "`n--- Next Steps ---" -ForegroundColor White
Write-Host "  1. Run recovery_test_run.ps1 to execute recovery tests"
Write-Host "  2. Run recovery_test_analyze.ps1 to analyze results"
Write-Host ""

exit 0
