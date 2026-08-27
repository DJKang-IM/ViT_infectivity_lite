$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. "$PSScriptRoot\_env.ps1"

Write-Host "Preparing gangnam (1xxxx) DICOM -> D:\[260626] ViT_Infectivity ..."
& $Python src/data/prepare_dicom.py --mode hardlink

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done."
