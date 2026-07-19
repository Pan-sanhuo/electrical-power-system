@echo off
chcp 65001 >nul
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo 请先运行“一键安装环境.bat”
  pause
  exit /b 1
)
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" "%~dp0demo_vscode.py"
pause
