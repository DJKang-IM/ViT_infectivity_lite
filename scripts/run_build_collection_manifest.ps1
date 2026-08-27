$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. "$PSScriptRoot\_env.ps1"

Write-Host "Building DICOM collection manifest..."
& $Python src/data/collection_manifest.py --out artifacts/dicom_collection_manifest.csv

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done."
