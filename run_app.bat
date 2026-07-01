@echo off
setlocal
cd /d "%~dp0"

set "APP=%~dp0app.py"
set "URL=http://127.0.0.1:8501"
set "STREAMLIT_EXE=D:\ProgramData\Anaconda\Scripts\streamlit.exe"

if not exist "%APP%" (
    echo ERROR: app.py was not found.
    echo Current folder: %CD%
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%URL%' -TimeoutSec 2; if ($r.StatusCode -ge 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 (
    echo Streamlit is already running.
    echo Opening %URL%
    start "" "%URL%"
    pause
    exit /b 0
)

if exist "%STREAMLIT_EXE%" (
    set "RUNNER=%STREAMLIT_EXE%"
) else (
    where streamlit >nul 2>nul
    if not errorlevel 1 (
        set "RUNNER=streamlit"
    ) else (
        where python >nul 2>nul
        if not errorlevel 1 (
            set "RUNNER=python -m streamlit"
        ) else (
            echo ERROR: Streamlit runner was not found.
            echo Install dependencies with: pip install -r requirements.txt
            pause
            exit /b 1
        )
    )
)

echo Starting Streamlit app...
echo App: %APP%
echo URL: %URL%
echo Keep this window open. Close it to stop the app.
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process '%URL%'"
%RUNNER% run "%APP%" --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false

echo.
echo Streamlit stopped.
pause
