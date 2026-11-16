@echo off
cd /d "%~dp0"

echo ============================================================
echo  MAIN EXPERIMENT : 1000 sequences per class + cluster splits
echo ============================================================
echo.
echo  Rough timing on CPU:
echo    fetching sequences    5 - 15 min  (cached afterwards)
echo    ESM-2 8M embedding   15 - 25 min  (cached afterwards)
echo    clustering + eval     2 -  5 min
echo.
echo  Safe to interrupt with Ctrl+C - caches are kept and the
echo  next run continues from where it stopped.
echo.

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
echo Using: %PYCMD%
echo.

echo [0/1] Making sure scipy is installed (needed for clustering)
%PYCMD% -m pip install scipy
if errorlevel 1 goto install_failed
echo.

set HF_ENDPOINT=https://hf-mirror.com
%PYCMD% run_main.py --n 1000
if errorlevel 1 goto run_failed

echo.
echo ============================================================
echo  DONE. Tell Claude: "main done"
echo ============================================================
goto end

:no_python
echo.
echo  NO PYTHON WITH PIP FOUND. Run 1_setup.bat first.
goto end

:install_failed
echo.
echo  scipy install failed. Tell Claude.
goto end

:run_failed
echo.
echo ------------------------------------------------------------
echo  SCRIPT ERROR. A file main_log.txt was written.
echo  Just tell Claude: "error" - it will read the log.
echo ------------------------------------------------------------

:end
echo.
pause
