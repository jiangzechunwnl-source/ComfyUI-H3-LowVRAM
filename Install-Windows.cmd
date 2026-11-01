@echo off
setlocal
title H3 LowVRAM - Windows Setup
powershell.exe -NoProfile -File "%~dp0Install-Windows.ps1"
if errorlevel 1 (
  echo.
  echo Setup did not complete. Read the message above.
  pause
)
