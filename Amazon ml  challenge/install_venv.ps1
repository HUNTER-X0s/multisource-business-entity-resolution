#!/usr/bin/env powershell
# install_venv.ps1
# Installs all project dependencies into the project virtual environment (venv\).
# Run ONCE to set up the project. Never installs anything on the system Python.
# PyTorch is installed with CUDA 12.4 support for the NVIDIA GeForce RTX 3050 (6 GB).

$venv_pip = "z:\Amazon ML\venv\Scripts\pip.exe"
$venv_python = "z:\Amazon ML\venv\Scripts\python.exe"

Write-Host "=== Amazon ML Challenge - Virtual Environment Package Installation ===" -ForegroundColor Cyan
Write-Host "venv Python: $venv_python" -ForegroundColor Green

# Upgrade pip inside venv
Write-Host "`n[1/6] Upgrading pip..." -ForegroundColor Yellow
& $venv_pip install --upgrade pip

# Core data packages
Write-Host "`n[2/6] Installing core data packages..." -ForegroundColor Yellow
& $venv_pip install `
    "pandas==3.0.3" `
    "polars==1.44.2" `
    "pyarrow==25.0.1" `
    "numpy==1.26.4" `
    "scipy==1.17.1" `
    "tqdm==4.70.0" `
    "psutil==7.2.2" `
    "regex==2026.7.19" `
    "joblib==1.5.3"

# String matching
Write-Host "`n[3/6] Installing string matching packages..." -ForegroundColor Yellow
& $venv_pip install `
    "rapidfuzz==3.14.6" `
    "python-Levenshtein==0.28.0" `
    "jellyfish==1.2.2"

# ML packages
Write-Host "`n[4/6] Installing ML packages (scikit-learn, XGBoost, LightGBM)..." -ForegroundColor Yellow
& $venv_pip install `
    "scikit-learn==1.9.0" `
    "xgboost==3.2.0" `
    "lightgbm==4.7.0"

# Visualization
Write-Host "`n[5/6] Installing visualization packages..." -ForegroundColor Yellow
& $venv_pip install `
    "matplotlib==3.11.1" `
    "plotly==6.9.0"

# PyTorch with CUDA 12.4 support (compatible with CUDA 13.2 driver on RTX 3050)
Write-Host "`n[6/6] Installing PyTorch with CUDA 12.4 support (GPU-enabled)..." -ForegroundColor Yellow
Write-Host "  This installs torch+cu124 for NVIDIA GeForce RTX 3050 6GB" -ForegroundColor DarkYellow
& $venv_pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host "`n=== Installation complete. Verifying key packages ===" -ForegroundColor Cyan
& $venv_python -c "
import sys
print('Python:', sys.version)

pkgs = ['torch', 'polars', 'pandas', 'numpy', 'scipy', 'sklearn', 'xgboost', 'lightgbm', 'rapidfuzz', 'jellyfish', 'psutil', 'matplotlib']
for pkg in pkgs:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'ok')
        print(f'  [OK] {pkg}: {ver}')
    except ImportError:
        print(f'  [MISSING] {pkg}')

import torch
print()
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
"
