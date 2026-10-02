#!/usr/bin/env powershell
# restructure_workspace.ps1
# Reorganizes the Amazon ML Challenge workspace into a clean canonical layout.
# Moves raw data into data/, preserves research_scripts and docs.
# Does NOT delete any source files.

$root = "z:\Amazon ML"
Set-Location $root

Write-Host "=== Amazon ML Challenge: Workspace Restructuring ===" -ForegroundColor Cyan

# -------------------------------------------------------------------------
# 1. Create canonical top-level directory structure
# -------------------------------------------------------------------------
$dirs = @(
    "data\raw\train",
    "data\raw\test",
    "data\original_archive\student_resource\dataset\train",
    "data\original_archive\student_resource\dataset\test",
    "data\original_archive\student_resource\utils",
    "data\macos_metadata",
    "research_scripts",
    "docs",
    "src",
    "output",
    "experiments",
    "venv"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path "$root\$d" | Out-Null
    Write-Host "  [OK] Directory: $root\$d"
}

# -------------------------------------------------------------------------
# 2. Copy (not move) data files to data/raw/ (preserve originals in-place)
# -------------------------------------------------------------------------
Write-Host "`nCopying train data to data\raw\train ..." -ForegroundColor Yellow
$trainFiles = @(
    "student_resource\dataset\train\train_source1.tsv",
    "student_resource\dataset\train\train_source2.tsv",
    "student_resource\dataset\train\train_source3.tsv",
    "student_resource\dataset\train\train_ground_truth.tsv"
)
foreach ($f in $trainFiles) {
    $src = "$root\$f"
    $dst = "$root\data\raw\train\$(Split-Path $f -Leaf)"
    if (!(Test-Path $dst)) {
        Copy-Item $src $dst
        Write-Host "  Copied: $f -> data\raw\train\$(Split-Path $f -Leaf)"
    } else {
        Write-Host "  Already exists (skip): $dst"
    }
}

Write-Host "`nCopying test data to data\raw\test ..." -ForegroundColor Yellow
$testFiles = @(
    "student_resource\dataset\test\test_source1.tsv",
    "student_resource\dataset\test\test_source2.tsv",
    "student_resource\dataset\test\test_source3.tsv"
)
foreach ($f in $testFiles) {
    $src = "$root\$f"
    $dst = "$root\data\raw\test\$(Split-Path $f -Leaf)"
    if (!(Test-Path $dst)) {
        Copy-Item $src $dst
        Write-Host "  Copied: $f -> data\raw\test\$(Split-Path $f -Leaf)"
    } else {
        Write-Host "  Already exists (skip): $dst"
    }
}

# -------------------------------------------------------------------------
# 3. Copy official documentation files
# -------------------------------------------------------------------------
Write-Host "`nCopying documentation files to docs\ ..." -ForegroundColor Yellow
$docFiles = @(
    "student_resource\README.md",
    "student_resource\Documentation_template.md"
)
foreach ($f in $docFiles) {
    $src = "$root\$f"
    $dst = "$root\docs\$(Split-Path $f -Leaf)"
    if (!(Test-Path $dst)) {
        Copy-Item $src $dst
        Write-Host "  Copied: $f -> docs\$(Split-Path $f -Leaf)"
    } else {
        Write-Host "  Already exists (skip): $dst"
    }
}

# -------------------------------------------------------------------------
# 4. Copy validation utility to src/
# -------------------------------------------------------------------------
Write-Host "`nCopying utils to src\ ..." -ForegroundColor Yellow
$utilsSrc = "$root\student_resource\utils\validate_submission.py"
$utilsDst = "$root\src\validate_submission.py"
if (!(Test-Path $utilsDst)) {
    Copy-Item $utilsSrc $utilsDst
    Write-Host "  Copied validate_submission.py -> src\"
} else {
    Write-Host "  Already exists (skip): $utilsDst"
}

Write-Host "`n=== Restructuring complete. Original files are untouched. ===" -ForegroundColor Green
Write-Host "Final workspace layout:" -ForegroundColor Cyan
Get-ChildItem $root -Depth 2 | Select-Object FullName | Format-Table -AutoSize
