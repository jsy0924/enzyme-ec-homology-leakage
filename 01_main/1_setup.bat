@echo off
cd /d "%~dp0"

echo ============================================================
echo  STEP 1 : pip mirror + light packages + Model A
echo ============================================================
echo.

rem ---------------------------------------------------------
rem  Find a real Python that has pip.
rem  Plain "python" may resolve to Inkscape's bundled Python,
rem  which has no pip, so try known locations first.
rem ---------------------------------------------------------
echo [0/4] Locating a Python that has pip

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
%PYCMD% -c "import sys; print('      ' + sys.version.split()[0] + '  ' + sys.executable)"
%PYCMD% -m pip --version
echo.

echo [1/4] Setting pip mirror to Tsinghua (permanent)
%PYCMD% -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
%PYCMD% -m pip config set global.timeout 120
%PYCMD% -m pip config set global.retries 10
echo.

echo [2/4] Current pip config
%PYCMD% -m pip config list
echo.

echo [3/4] Installing: requests pandas scikit-learn matplotlib
echo       About 50 MB total.
echo       Watch the host name in the download lines:
echo         pypi.tuna.tsinghua.edu.cn  = mirror active  (good)
echo         files.pythonhosted.org     = mirror NOT used (bad)
echo.
%PYCMD% -m pip install requests pandas scikit-learn matplotlib
if errorlevel 1 goto install_failed
echo.

echo [4/4] Running Model A  (no torch needed)
echo.
%PYCMD% run_prelim.py --only-a
if errorlevel 1 goto run_failed

echo.
echo ============================================================
echo  STEP 1 DONE.
echo  Tell Claude: "1 done"
echo  Then run 2_install_torch.bat
echo ============================================================
goto end

:no_python
echo.
echo ------------------------------------------------------------
echo  NO PYTHON WITH PIP FOUND.
echo.
echo  Your "python" command points at Inkscape's bundled Python,
echo  which has no pip. The real Python was expected at:
echo    %LOCALAPPDATA%\Programs\Python\Python311\python.exe
echo.
echo  To find it yourself, run:
echo    where python
echo    dir /b "%LOCALAPPDATA%\Programs\Python"
echo.
echo  Then send that output to Claude.
echo ------------------------------------------------------------
goto end

:install_failed
echo.
echo ------------------------------------------------------------
echo  PACKAGE INSTALL FAILED.
echo  Check the host name shown above, then tell Claude.
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
