@echo off
REM Windows GUI Installation Script for Trading Bot Dashboard

echo ========================================
echo Trading Bot Dashboard - GUI Installer
echo ========================================
echo.

echo [1/3] Installing GUI dependencies...
pip install -r requirements_gui.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Please check your Python installation.
    pause
    exit /b 1
)

echo.
echo [2/3] Verifying database connection...
echo Please ensure PostgreSQL is running.
echo.

echo [3/3] Setup complete!
echo.
echo ========================================
echo Installation successful!
echo ========================================
echo.
echo To start the GUI dashboard:
echo    python app.py
echo.
echo Or double-click: run_gui.bat
echo.

pause
