# One-time / repair: Python 3.11 venv + CUDA PyTorch (cu124, RTX 4070).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py311 = $env:PYTHON311
if (-not $Py311) {
    $Py311 = "C:\Users\SEJONG_ENDO_3\AppData\Local\Programs\Python\Python311\python.exe"
}
if (-not (Test-Path $Py311)) {
    $Py311 = (& py -3.11 -c "import sys; print(sys.executable)").Trim()
}
if (-not (Test-Path $Py311)) {
    Write-Error "Python 3.11 required for CUDA PyTorch (system 3.14 has CPU-only wheels)."
}

Write-Host "Using Python: $Py311"
Write-Host "Creating venv at $Root\.venv ..."
& $Py311 -m venv ".venv"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPy -m pip install --upgrade pip

Write-Host "Installing CUDA PyTorch (cu124)..."
& $VenvPy -m pip install torch torchvision --index-url http<REDACTED_PATH>
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing project dependencies..."
& $VenvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
& $VenvPy -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host ""
Write-Host "Done. Scripts use .venv automatically; or: . .\activate.ps1"
