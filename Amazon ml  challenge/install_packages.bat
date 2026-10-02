@echo off
:: install_packages.bat
:: Install all project dependencies into the venv
:: Run from the project root: cmd /c install_packages.bat

set VENV_PYTHON=z:\Amazon ML\venv\Scripts\python.exe
set VENV_PIP=%VENV_PYTHON% -m pip

echo === Amazon ML Challenge 2026: Virtual Environment Setup ===
echo.

echo [1/7] Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip

echo.
echo [2/7] Installing core data packages (numpy, pandas, polars, pyarrow, scipy)...
"%VENV_PYTHON%" -m pip install "numpy==1.26.4" "pandas==3.0.3" "polars==1.44.2" "pyarrow==25.0.1" "scipy==1.17.1"

echo.
echo [3/7] Installing utilities (tqdm, psutil, regex, joblib)...
"%VENV_PYTHON%" -m pip install "tqdm==4.70.0" "psutil==7.2.2" "regex==2026.7.19" "joblib==1.5.3"

echo.
echo [4/7] Installing string matching (rapidfuzz, python-Levenshtein, jellyfish)...
"%VENV_PYTHON%" -m pip install "rapidfuzz==3.14.6" "python-Levenshtein==0.28.0" "jellyfish==1.2.2"

echo.
echo [5/7] Installing ML frameworks (scikit-learn, XGBoost, LightGBM)...
"%VENV_PYTHON%" -m pip install "scikit-learn==1.9.0" "xgboost==3.2.0" "lightgbm==4.7.0"

echo.
echo [6/7] Installing visualization packages (matplotlib, plotly)...
"%VENV_PYTHON%" -m pip install "matplotlib==3.11.1" "plotly==6.9.0"

echo.
echo [7/7] Installing PyTorch with CUDA 12.4 support (GPU-enabled for RTX 3050)...
echo  This may take several minutes - downloading ~2.5GB CUDA-enabled PyTorch...
"%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo === Installation complete. Running verification... ===
"%VENV_PYTHON%" -c "import sys; print('Python:', sys.version)"
"%VENV_PYTHON%" -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
"%VENV_PYTHON%" -c "import polars, pandas, numpy, rapidfuzz, sklearn, xgboost, lightgbm; print('All core packages OK')"

echo.
echo === Setup complete. Use venv\Scripts\activate to activate. ===
