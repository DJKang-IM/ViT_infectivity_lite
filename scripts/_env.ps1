# Shared Python resolver: project .venv (Python 3.11 + CUDA PyTorch).
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$script:Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Ensure-ProjectVenv {
    if (-not (Test-Path $script:Python)) {
        Write-Host "Project venv not found; running setup..."
        & (Join-Path $PSScriptRoot "setup_venv.ps1")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        return
    }

    $cudaOk = & $script:Python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaOk -ne "True") {
        Write-Host "Venv PyTorch is not CUDA-enabled; re-running setup..."
        & (Join-Path $PSScriptRoot "setup_venv.ps1")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

Ensure-ProjectVenv
