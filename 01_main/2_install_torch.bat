@echo off
cd /d "%~dp0"

echo ============================================================
echo  STEP 2 : torch + Model B
echo ============================================================
echo.
echo  torch is about 127 MB.
echo  If the download breaks, just run this file again -
echo  pip keeps a cache, so retries add up.
echo.

rem ---------------------------------------------------------
rem  Find a real Python that has pip (same logic as step 1).
rem ---------------------------------------------------------
echo [0/3] Locating a Python that has pip

set PYCMD=
set "CAND=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not exist "%CAND%" goto try_launcher
"%CAND%" -m pip --version >nul 2>&1
if errorlevel 1 goto try_launcher
set PYCMD="%CAND%"
goto found

:try_launcher
py -3 -m pip --version >nul 2>&1
if errorlevel 1 goto try_plain
set PYCMD=py -3
goto found

:try_plain
python -m pip --version >nul 2>&1
if errorlevel 1 goto no_python
set PYCMD=python
goto found

:found
echo       Using: %PYCMD%
echo.

echo [1/3] Installing torch
%PYCMD% -m pip install torch
if errorlevel 1 goto install_failed
echo.

echo [2/3] Installing transformers
%PYCMD% -m pip install transformers
if errorlevel 1 goto install_failed
echo.

echo [3/3] Running Model A + B
echo       ESM-2 weights (about 30 MB) via Hugging Face mirror.
set HF_ENDPOINT=https://hf-mirror.com
echo       HF_ENDPOINT=%HF_ENDPOINT%
echo.
%PYCMD% run_prelim.py
if errorlevel 1 goto run_failed

echo.
echo ============================================================
echo  ALL DONE.
echo  Tell Claude: "all done"
echo ============================================================
goto end

:no_python
echo.
echo ------------------------------------------------------------
echo  NO PYTHON WITH PIP FOUND. Run 1_setup.bat first.
echo ------------------------------------------------------------
goto end

:install_failed
echo.
echo ------------------------------------------------------------
echo  INSTALL FAILED. Try these in order:
echo.
echo   (1) Run this file again, two or three times.
echo       pip caches partial downloads, so progress accumulates.
echo.
echo   (2) Switch to the Aliyun mirror, then run this file again:
echo       %PYCMD% -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
echo.
echo   (3) Still failing - tell Claude.
echo ------------------------------------------------------------
goto end

:run_failed
echo.
echo ------------------------------------------------------------
echo  SCRIPT ERROR. A file run_log.txt was written.
echo  Just tell Claude: "error" - it will read the log.
echo ------------------------------------------------------------

:end
echo.
pause
