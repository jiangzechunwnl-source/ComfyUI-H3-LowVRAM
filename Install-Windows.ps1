# SPDX-License-Identifier: GPL-3.0-or-later
[CmdletBinding()]
param([string]$ComfyFolder, [switch]$NoDialogs)
$ErrorActionPreference='Stop'
$payload=[IO.Path]::GetFullPath($PSScriptRoot)
if([string]::IsNullOrWhiteSpace($ComfyFolder)) {
    if($NoDialogs){throw 'Specify -ComfyFolder.'}
    Add-Type -AssemblyName System.Windows.Forms
    $picker=New-Object System.Windows.Forms.OpenFileDialog
    $picker.Title='Select main.py inside your existing ComfyUI folder'
    $picker.Filter='ComfyUI main.py|main.py'
    if($picker.ShowDialog() -ne [Windows.Forms.DialogResult]::OK){Write-Host 'Setup cancelled.';exit 0}
    $ComfyFolder=Split-Path -Parent $picker.FileName
    $picker.Dispose()
}
if($ComfyFolder.StartsWith('\\') -or $ComfyFolder.StartsWith('//')){throw 'Network folders are not supported.'}
$comfy=[IO.Path]::GetFullPath($ComfyFolder)
if(Test-Path -LiteralPath (Join-Path $comfy 'ComfyUI\main.py')){$comfy=Join-Path $comfy 'ComfyUI'}
if(-not (Test-Path -LiteralPath (Join-Path $comfy 'main.py') -PathType Leaf) -or
   -not (Test-Path -LiteralPath (Join-Path $comfy 'comfy\ldm\minimax\model.py') -PathType Leaf)) {
    throw 'This is not a ComfyUI folder with native MiniMax H3 support.'
}
$parent=Split-Path -Parent $comfy
$candidates=@(
    (Join-Path $parent 'python_embeded\python.exe'),
    (Join-Path $parent 'python_embedded\python.exe'),
    (Join-Path $comfy '.venv\Scripts\python.exe'),
    (Join-Path $comfy 'venv\Scripts\python.exe'),
    (Join-Path $parent '.venv\Scripts\python.exe')
)
$python=$candidates | Where-Object {Test-Path -LiteralPath $_ -PathType Leaf} | Select-Object -First 1
if(-not $python){throw 'Existing ComfyUI Python not found. No packages were installed. See README.md for manual launch.'}
$destination=Join-Path $parent 'H3-LowVRAM'
if([IO.Path]::GetFullPath($destination).Equals($payload,[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Extract the download into a separate folder, then run Setup again.'
}
# Only explicit package files are copied; original ComfyUI code is never modified.
$files=@('run.py','h3_low_vram.py','README.md','LICENSE','NOTICE.md','VALIDATION.md')
foreach($file in $files){if(-not (Test-Path -LiteralPath (Join-Path $payload $file) -PathType Leaf)){throw "Package file missing: $file"}}
$launcher=Join-Path $parent 'Start-H3-LowVRAM.cmd'
$backup=Join-Path $parent ('H3-LowVRAM-backup-'+(Get-Date -Format 'yyyyMMdd-HHmmss-ffff'))
if((Test-Path -LiteralPath $destination) -or (Test-Path -LiteralPath $launcher)) {
    New-Item -ItemType Directory -Path $backup | Out-Null
    if(Test-Path -LiteralPath $destination){Copy-Item -LiteralPath $destination -Destination (Join-Path $backup 'H3-LowVRAM') -Recurse}
    if(Test-Path -LiteralPath $launcher){Copy-Item -LiteralPath $launcher -Destination $backup}
}
New-Item -ItemType Directory -Path $destination -Force | Out-Null
foreach($file in $files) {
    $source=Join-Path $payload $file
    $target=Join-Path $destination $file
    Copy-Item -LiteralPath $source -Destination $target -Force
    if((Get-FileHash -LiteralPath $source).Hash -cne (Get-FileHash -LiteralPath $target).Hash){throw "Copy verification failed: $file"}
}
# Use paths relative to the launcher's folder so an entire portable install can move.
$pythonRelative=$python.Substring($parent.TrimEnd('\').Length+1)
$comfyRelative=$comfy.Substring($parent.TrimEnd('\').Length+1)
if($pythonRelative.Contains('%') -or $comfyRelative.Contains('%')){throw 'Folder names containing a percent sign are not supported.'}
$start=@'
@echo off
@chcp 65001 >nul
setlocal
title MiniMax H3 - LowVRAM
echo Checking H3 compatibility. Nothing will be downloaded.
"%~dp0__PYTHON__" -s -B "%~dp0H3-LowVRAM\run.py" --comfy-root "%~dp0__COMFY__" --check
if errorlevel 1 goto failed
echo.
echo Keep this window open. Open http://127.0.0.1:8188 in your browser when ready.
"%~dp0__PYTHON__" -s -B "%~dp0H3-LowVRAM\run.py" --comfy-root "%~dp0__COMFY__"
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo H3 LowVRAM stopped. Read the error above. Your normal ComfyUI launcher is unchanged.
pause
exit /b 1
'@
$start=$start.Replace('__PYTHON__',$pythonRelative).Replace('__COMFY__',$comfyRelative)
[IO.File]::WriteAllText($launcher,$start.Replace("`r`n","`n").Replace("`n","`r`n"),[Text.UTF8Encoding]::new($false))
Write-Host "Installed successfully. Double-click: $launcher"
Write-Host 'Use your existing H3 workflow. Close your normal ComfyUI before launching this one.'
if(-not $NoDialogs){[Windows.Forms.MessageBox]::Show("Setup complete.`r`n`r`nDouble-click Start-H3-LowVRAM.cmd in your ComfyUI installation folder.`r`n`r`nClose your normal ComfyUI first.",'H3 LowVRAM') | Out-Null}
