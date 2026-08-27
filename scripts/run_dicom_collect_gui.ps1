$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. "$PSScriptRoot\_env.ps1"

Write-Host "Launching DICOM collection GUI..."
& $Python gui/dicom_collection_gui.py
