# =============================================================================
#  setup.ps1 - One-shot environment setup for the Video Summarizer project
# -----------------------------------------------------------------------------
#  What this does (mirrors Setup_Guide.docx exactly):
#    1. Cleans up stray .crdownload files
#    2. Verifies Python 3.11 is installed (prompts to install via winget if not)
#    3. Verifies ffmpeg is on PATH (installs via winget if not)
#    4. Removes any existing backend\venv (was Python 3.8 - we rebuild clean)
#    5. Creates a fresh venv with Python 3.11
#    6. Activates it, upgrades pip
#    7. Installs requirements.txt (CPU-only PyTorch)
#    8. Verifies the checkpoint file (best.pt) is in place
#
#  Run from the vsum-web folder (the one containing backend/ and frontend/):
#      .\setup.ps1
#
#  Safe to re-run: skips steps that are already done.
# =============================================================================

$ErrorActionPreference = "Stop"

function Write-Section($msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $msg -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Write-Ok($msg)   { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Yellow }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERR]  $msg" -ForegroundColor Red }

# Make sure we are in the project root (the one with backend\ and frontend\)
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path "backend") -or -not (Test-Path "frontend")) {
    Write-Err "This script must live alongside backend\ and frontend\ folders."
    Write-Err "Current location: $projectRoot"
    exit 1
}

# -----------------------------------------------------------------------------
# 1. Clean up stray partial downloads
# -----------------------------------------------------------------------------
Write-Section "Step 1/7  Clean up stray partial downloads"

$stray = Get-ChildItem -Path . -Recurse -Filter "Unconfirmed *.crdownload" -ErrorAction SilentlyContinue
if ($stray) {
    foreach ($f in $stray) {
        Write-Info "Removing $($f.FullName)"
        Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
    }
    Write-Ok "Cleaned $($stray.Count) partial download file(s)."
} else {
    Write-Ok "No stray .crdownload files found."
}

# -----------------------------------------------------------------------------
# 2. Python 3.11 check
# -----------------------------------------------------------------------------
Write-Section "Step 2/7  Verify Python 3.11"

$python311 = $null

# Try the "py" launcher first (standard on Windows Python installs)
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    try {
        $v = & py -3.11 --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $v -match "3\.11") {
            $python311 = "py -3.11"
            Write-Ok "Found Python 3.11 via py launcher: $v"
        }
    } catch {}
}

# Fallback: look for python.exe directly
if (-not $python311) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "C:\Python311\python.exe",
        "C:\Program Files\Python311\python.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $v = & $c --version 2>&1
            if ($v -match "3\.11") {
                $python311 = "`"$c`""
                Write-Ok "Found Python 3.11 at $c"
                break
            }
        }
    }
}

if (-not $python311) {
    Write-Warn "Python 3.11 not found on PATH."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Info "Installing Python 3.11 via winget..."
        winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Err "winget install failed. Please install Python 3.11 manually:"
            Write-Err "  https://www.python.org/downloads/release/python-3119/"
            exit 1
        }
        # Refresh PATH in this session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        # Re-test
        $v = & py -3.11 --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $v -match "3\.11") {
            $python311 = "py -3.11"
            Write-Ok "Installed Python 3.11 successfully: $v"
        } else {
            Write-Err "Python 3.11 installed but not detected. Close this terminal, open a new one, and re-run setup.ps1."
            exit 1
        }
    } else {
        Write-Err "winget is not available. Please install Python 3.11 manually:"
        Write-Err "  https://www.python.org/downloads/release/python-3119/"
        Write-Err "  IMPORTANT: tick 'Add python.exe to PATH' on the first installer screen."
        exit 1
    }
}

# -----------------------------------------------------------------------------
# 3. ffmpeg check
# -----------------------------------------------------------------------------
Write-Section "Step 3/7  Verify ffmpeg"

$ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffmpegCmd) {
    $ver = (& ffmpeg -version 2>&1 | Select-Object -First 1)
    Write-Ok "ffmpeg already on PATH: $ver"
} else {
    Write-Warn "ffmpeg not found on PATH."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Info "Installing ffmpeg via winget (Gyan.FFmpeg)..."
        winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Err "winget install failed."
            Write-Err "Manual option: unzip ffmpeg-release-essentials.zip (in the parent folder)"
            Write-Err "to C:\ffmpeg, then add C:\ffmpeg\bin to your PATH and reopen the terminal."
            exit 1
        }
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
        if ($ffmpegCmd) {
            Write-Ok "ffmpeg installed successfully."
        } else {
            Write-Warn "ffmpeg installed but not yet on PATH in this session."
            Write-Warn "Close this terminal, open a new one, and re-run setup.ps1."
            exit 1
        }
    } else {
        Write-Err "winget is not available. Unzip ffmpeg-release-essentials.zip to C:\ffmpeg,"
        Write-Err "then add C:\ffmpeg\bin to your PATH environment variable, reopen the terminal,"
        Write-Err "and re-run setup.ps1."
        exit 1
    }
}

# -----------------------------------------------------------------------------
# 4. Remove old venv
# -----------------------------------------------------------------------------
Write-Section "Step 4/7  Remove old venv (if any)"

$venvPath = Join-Path $projectRoot "backend\venv"
if (Test-Path $venvPath) {
    $cfg = Join-Path $venvPath "pyvenv.cfg"
    if (Test-Path $cfg) {
        $line = Get-Content $cfg | Where-Object { $_ -match "^version\s*=" } | Select-Object -First 1
        Write-Info "Existing venv detected ($line). Removing for clean Python 3.11 rebuild..."
    } else {
        Write-Info "Existing venv folder found. Removing..."
    }
    Remove-Item -LiteralPath $venvPath -Recurse -Force
    Write-Ok "Old venv removed."
} else {
    Write-Ok "No existing venv to remove."
}

# -----------------------------------------------------------------------------
# 5. Create fresh venv
# -----------------------------------------------------------------------------
Write-Section "Step 5/7  Create new venv with Python 3.11"

Set-Location (Join-Path $projectRoot "backend")
Write-Info "Running: $python311 -m venv venv"
Invoke-Expression "$python311 -m venv venv"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path "venv\Scripts\python.exe")) {
    Write-Err "Failed to create venv. Check Python 3.11 install."
    exit 1
}
Write-Ok "Created venv at backend\venv"

# Use the venv's python directly - no need to activate the shell
$venvPy  = Join-Path $projectRoot "backend\venv\Scripts\python.exe"
$venvPip = Join-Path $projectRoot "backend\venv\Scripts\pip.exe"

$venvVer = & $venvPy --version 2>&1
Write-Ok "venv Python: $venvVer"

# -----------------------------------------------------------------------------
# 6. Install dependencies
# -----------------------------------------------------------------------------
Write-Section "Step 6/7  Install Python dependencies (this can take 5-15 minutes)"

Write-Info "Upgrading pip..."
& $venvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Err "pip upgrade failed."; exit 1 }

Write-Info "Installing CPU-only PyTorch 2.4.1..."
& $venvPy -m pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { Write-Err "PyTorch install failed."; exit 1 }

Write-Info "Installing project requirements..."
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Err "requirements.txt install failed."; exit 1 }

Write-Ok "All Python dependencies installed."

# -----------------------------------------------------------------------------
# 7. Verify checkpoint
# -----------------------------------------------------------------------------
Write-Section "Step 7/7  Verify model checkpoint"

$ckpt = Join-Path $projectRoot "backend\models\best.pt"
if (Test-Path $ckpt) {
    $sizeMB = [math]::Round((Get-Item $ckpt).Length / 1MB, 1)
    Write-Ok "Found checkpoint: backend\models\best.pt ($sizeMB MB)"
} else {
    Write-Warn "No backend\models\best.pt found."
    Write-Warn "Copy your trained .pt checkpoint into backend\models\ and rename it best.pt"
    Write-Warn "(or set \$env:CHECKPOINT_PATH when starting the server)."
}

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
Write-Section "Setup complete!"

Write-Host ""
Write-Host "To start the web server, run these commands in the VS Code terminal:" -ForegroundColor Green
Write-Host ""
Write-Host "    cd backend"                                        -ForegroundColor White
Write-Host "    .\venv\Scripts\Activate.ps1"                        -ForegroundColor White
Write-Host "    uvicorn main:app --host 0.0.0.0 --port 8000"        -ForegroundColor White
Write-Host ""
Write-Host "Or just run:  .\run.ps1"                                -ForegroundColor Green
Write-Host ""
Write-Host "Then open http://localhost:8000 in your browser."       -ForegroundColor Green
Write-Host ""
