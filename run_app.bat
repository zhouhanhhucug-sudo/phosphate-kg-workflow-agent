@echo off
cd /d "%~dp0"
"D:\ProgramData\Anaconda\Scripts\streamlit.exe" run "%~dp0app.py" --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
pause
