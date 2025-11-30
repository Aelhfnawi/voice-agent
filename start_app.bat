@echo off
echo ========================================
echo Starting Voice Agent System
echo ========================================
echo.

REM Get the directory where the batch file is located
set "PROJECT_DIR=%~dp0project"
set "FRONTEND_DIR=%~dp0frontend"

echo Starting Backend API...
start "Backend API" cmd /k "cd /d %PROJECT_DIR% && uvicorn api.server:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

echo Starting Voice Agent...
start "Voice Agent" cmd /k "cd /d %PROJECT_DIR% && python agent/livekit_agent.py dev"

timeout /t 3 /nobreak >nul

echo Starting Frontend...
start "Frontend" cmd /k "cd /d %FRONTEND_DIR% && npm start"

echo.
echo ========================================
echo All services starting...
echo ========================================
echo.
echo Backend API:     http://localhost:8000
echo Frontend:        http://localhost:3000
echo Voice Agent:     Running in dev mode
echo.
echo Press any key to close this window...
pause >nul
