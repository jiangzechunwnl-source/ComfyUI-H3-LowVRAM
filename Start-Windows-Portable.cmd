@echo off
setlocal
rem Place this folder beside ComfyUI and python_embeded inside a Windows portable install.
set "H3_SHARE_ROOT=%~dp0.."
if not exist "%H3_SHARE_ROOT%\python_embeded\python.exe" (
  echo Existing portable Python not found. See README.md.
  pause
  exit /b 2
)
"%H3_SHARE_ROOT%\python_embeded\python.exe" -s -B "%~dp0run.py" --comfy-root "%H3_SHARE_ROOT%\ComfyUI"
pause
