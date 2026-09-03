@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Environnement Python introuvable : venv\Scripts\python.exe
    exit /b 1
)

"venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000
