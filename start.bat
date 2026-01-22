@echo off
REM Forensic Face Description System - Windows Launcher
REM Start both backend and frontend with one click

setlocal enabledelayedexpansion

cls
echo ============================================================
echo   FORENSIC FACE DESCRIPTION SYSTEM - Launcher
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [1] Python found: 
python --version
echo.

REM Check dependencies
echo [2] Checking dependencies...
python -c "import fastapi, streamlit, requests, pydantic" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Some dependencies are missing.
    echo Installing required packages...
    echo.
    
    pip install -r requirements.txt
    pip install -r backend\requirements.txt
    pip install -r frontend\requirements.txt
    
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Dependencies OK
echo.

REM Ask user which mode to run
echo [3] Select startup mode:
echo.
echo 1. Run Everything (Backend + Frontend)
echo 2. Backend Only
echo 3. Frontend Only
echo 4. Custom Ports
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting Forensic Face Description System...
    python app.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Backend API only...
    python app.py --backend-only
) else if "%choice%"=="3" (
    echo.
    echo Starting Frontend only...
    echo NOTE: Make sure backend is running first!
    python app.py --frontend-only
) else if "%choice%"=="4" (
    echo.
    set /p backend_port="Enter backend port (default 8000): "
    set /p frontend_port="Enter frontend port (default 8501): "
    
    if "!backend_port!"=="" set backend_port=8000
    if "!frontend_port!"=="" set frontend_port=8501
    
    echo Starting with Backend: !backend_port!, Frontend: !frontend_port!
    python app.py --backend-port !backend_port! --frontend-port !frontend_port!
) else (
    echo Invalid choice
    pause
    exit /b 1
)

pause
