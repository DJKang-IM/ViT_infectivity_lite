$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. "$PSScriptRoot\_env.ps1"

& "$PSScriptRoot/prepare_dicom_gangnam.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& "$PSScriptRoot/run_build_labels.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Smoke train (CPU-friendly: vit_small, 64 studies, 1 epoch)..."
& $Python src/train.py --config configs/smoke.yaml --tag v1_smoke --fold 0

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Smoke OK. See artifacts/v1_v1_smoke/"
