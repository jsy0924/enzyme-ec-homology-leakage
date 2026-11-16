@echo off
cd /d "%~dp0"

echo ============================================================
echo  BLAST nearest-neighbour prediction (model D)
echo ============================================================
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
%PYCMD% 03_blast_predict.py
if errorlevel 1 goto failed
goto end

:no_python
echo  NO PYTHON WITH PIP FOUND.
goto end

:failed
echo.
echo  Script error. Tell Claude: "blast error"

:end
echo.
pause
