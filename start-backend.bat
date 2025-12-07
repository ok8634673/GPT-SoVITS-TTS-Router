@echo off
REM 后端服务启动脚本
echo =======================================
echo 🚀 Backend Server Launcher
echo =======================================
echo.

REM 设置脚本所在目录为工作目录
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Current directory: %CD%
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ERROR: Python is not installed!
    echo Please download and install Python 3.7+ from https://python.org/
    echo.
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM 检查main.py是否存在
if not exist "main.py" (
    echo ❌ ERROR: main.py not found!
    echo Please ensure this script is in the backend directory:
    echo E:\python project\GPT-SoVITS-v4-20250422fix\tts_router\backend
    echo.
    pause
    exit /b 1
)

REM 启动后端服务
echo =======================================
echo Starting Backend Server
echo =======================================
echo.
echo ⚠️  IMPORTANT: This window will remain open to keep the server running!
echo ⚠️  Do NOT close this window while using the backend!
echo.
echo Server will be available at: http://localhost:8888
echo Health check: http://localhost:8888/health
echo.
echo Press Ctrl+C to stop the server
echo.
echo =======================================
echo Server Output:
echo =======================================
echo.

REM 直接启动后端服务
python main.py

REM 如果服务器意外退出，显示错误信息
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ ERROR: Backend server exited unexpectedly!
    echo Exit code: %ERRORLEVEL%
    echo Please check the output above for errors.
    echo.
    pause
)
