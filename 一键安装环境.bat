@echo off
chcp 65001 >nul
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
pause
