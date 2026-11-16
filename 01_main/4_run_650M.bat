@echo off
cd /d "%~dp0"

echo ============================================================
echo  MODEL C : ESM-2 650M  (compared against 8M in one figure)
echo ============================================================
echo.
echo  WARNING - this one is slow.
echo    model download   ~2.6 GB   (one time, via HF mirror)
echo    embedding 7000   1-3 hours on CPU
echo.
echo  The 8M embeddings are already cached, so they are reused.
echo  Safe to interrupt with Ctrl+C; sequences stay cached, but
echo  650M embedding restarts from the beginning if stopped.
echo.
pause

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

set HF_ENDPOINT=https://hf-mirror.com
echo HF_ENDPOINT=%HF_ENDPOINT%
echo.

%PYCMD% run_main.py --n 1000 --esm 8M,650M
if errorlevel 1 goto run_failed

echo.
echo ============================================================
echo  DONE. Tell Claude: "650M done"
echo ============================================================
goto end

:no_python
echo.
echo  NO PYTHON WITH PIP FOUND. Run 1_setup.bat first.
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
