<#
.SYNOPSIS
    One-shot setup on Windows: virtualenv, dependencies, ffmpeg and models.

.DESCRIPTION
    The Windows counterpart of setup.sh. Safe to re-run: every step is skipped
    when it is already satisfied.

    Unlike Linux there is no package manager to lean on for ffmpeg, so a static
    build is downloaded into bin\ and used from there. Nothing is installed
    system-wide and nothing needs administrator rights.

.PARAMETER Cpu
    Install the CPU-only build of PyTorch even if an NVIDIA GPU is present.

.PARAMETER NoGui
    Command line only; skip the desktop interface.

.PARAMETER WithDemucs
    Also install Demucs (keeps the original music under the dub).

.PARAMETER WithKokoro
    Also install the Kokoro voice (more natural, heavier).

.PARAMETER SkipModels
    Install packages only; models download on first use.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1
#>
[CmdletBinding()]
param(
    [string]$Python   = "",
    [string]$AsrModel = "",
    [string]$Voice    = "",
    [switch]$Cpu,
    [switch]$NoGui,
    [switch]$WithDemucs,
    [switch]$WithKokoro,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"   # a visible progress bar makes downloads far slower
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$TorchVersion = "2.5.1"
$CudaTag      = "cu121"
$FfmpegUrl    = "https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip"

function Info($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  ok $m"  -ForegroundColor Green }
function Warn($m) { Write-Host "warn $m"  -ForegroundColor Yellow }
function Die($m)  { Write-Host " err $m"  -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------------ Python
# A candidate counts only if it runs and can build a venv. The Microsoft Store
# stub named python.exe is on PATH by default and exits without doing anything,
# so "does python.exe exist" is not a usable test.
function Test-PythonCandidate($exe) {
    try {
        $v = & $exe -c "import ensurepip, sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $v) { return $false }
        $parts = $v.Trim().Split('.')
        return ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 9 -and [int]$parts[1] -le 11)
    } catch { return $false }
}

function Find-Python {
    if ($Python) {
        if (Test-PythonCandidate $Python) { return $Python }
        Die "$Python is not a usable Python 3.9-3.11 (it must also provide venv/ensurepip)."
    }
    # The py launcher can name an exact version, which is the reliable route.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.11", "3.10", "3.9")) {
            try {
                $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $exe -and (Test-PythonCandidate $exe.Trim())) {
                    return $exe.Trim()
                }
            } catch { }
        }
    }
    foreach ($name in @("python3.11", "python3.10", "python3.9", "python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and (Test-PythonCandidate $cmd.Source)) { return $cmd.Source }
    }
    return $null
}

Info "Locating a suitable Python (3.9-3.11)"
$py = Find-Python
if (-not $py) {
    Die @"
No usable Python 3.9-3.11 found.

  Install it from https://www.python.org/downloads/  (tick "Add python.exe to PATH"),
  or from the Microsoft Store, then re-run this script.

  If you have one already, point at it directly:
    powershell -ExecutionPolicy Bypass -File setup.ps1 -Python C:\Path\To\python.exe
"@
}
Ok "$py  $(& $py -c 'import sys; print(sys.version.split()[0])')"

# ------------------------------------------------------------------- ffmpeg
$binDir = Join-Path $root "bin"
if ((Test-Path (Join-Path $binDir "ffmpeg.exe")) -and (Test-Path (Join-Path $binDir "ffprobe.exe"))) {
    Ok "ffmpeg already in bin\"
} elseif (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Ok "ffmpeg found on PATH"
} else {
    Info "Downloading a static ffmpeg into bin\ (about 80 MB)"
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    $zip = Join-Path $env:TEMP "vtrans-ffmpeg.zip"
    try {
        Invoke-WebRequest -Uri $FfmpegUrl -OutFile $zip -UseBasicParsing
    } catch {
        Die "Could not download ffmpeg: $_`nDownload it yourself from https://www.gyan.dev/ffmpeg/builds/ and put ffmpeg.exe and ffprobe.exe in $binDir"
    }
    $stage = Join-Path $env:TEMP "vtrans-ffmpeg"
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    Expand-Archive -Path $zip -DestinationPath $stage -Force
    foreach ($name in @("ffmpeg.exe", "ffprobe.exe")) {
        $found = Get-ChildItem -Path $stage -Recurse -Filter $name -File | Select-Object -First 1
        if ($found) { Copy-Item $found.FullName -Destination $binDir -Force }
    }
    Remove-Item -Recurse -Force $stage, $zip -ErrorAction SilentlyContinue
    if (-not (Test-Path (Join-Path $binDir "ffmpeg.exe"))) { Die "The ffmpeg archive did not contain ffmpeg.exe" }
    Ok "ffmpeg -> bin\"
}

# -------------------------------------------------------------------- venv
$venv = Join-Path $root ".venv"
$vpy  = Join-Path $venv "Scripts\python.exe"
if (Test-Path $vpy) {
    Info "Reusing existing virtualenv at .venv"
} else {
    Info "Creating virtualenv at .venv"
    & $py -m venv $venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $vpy)) { Die "venv creation failed" }
}
& $vpy -m pip install --quiet --upgrade pip setuptools wheel
Ok "pip $(& $vpy -m pip --version | ForEach-Object { $_.Split(' ')[1] })"

# ------------------------------------------------------------------- torch
function Test-NvidiaGpu {
    if ($Cpu) { return $false }
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return $false }
    & $smi.Source -L *> $null
    return ($LASTEXITCODE -eq 0)
}

& $vpy -c "import torch" *> $null
if ($LASTEXITCODE -eq 0) {
    Info "PyTorch already installed ($(& $vpy -c 'import torch; print(torch.__version__)'))"
} else {
    if (Test-NvidiaGpu) {
        Info "NVIDIA GPU detected - installing PyTorch $TorchVersion ($CudaTag). This is about 2.5 GB."
        & $vpy -m pip install "torch==$TorchVersion" "torchaudio==$TorchVersion" --index-url "https://download.pytorch.org/whl/$CudaTag"
        if ($LASTEXITCODE -ne 0) { Die "PyTorch install failed. Retry with -Cpu for the CPU-only build." }
    } else {
        Info "No NVIDIA GPU detected - installing the CPU build of PyTorch"
        & $vpy -m pip install "torch==$TorchVersion" "torchaudio==$TorchVersion" --index-url "https://download.pytorch.org/whl/cpu"
        if ($LASTEXITCODE -ne 0) { Die "PyTorch install failed." }
    }
}

# -------------------------------------------------------------- python deps
Info "Installing pipeline dependencies"
& $vpy -m pip install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { Die "Dependency install failed" }

if (-not $NoGui) {
    Info "Installing the desktop interface (PySide6)"
    & $vpy -m pip install "PySide6-Essentials==6.8.1"
    if ($LASTEXITCODE -ne 0) { Die "PySide6 install failed" }
}

if ($WithDemucs) {
    Info "Installing Demucs"
    & $vpy -m pip install "demucs==4.0.1"
}
if ($WithKokoro) {
    Info "Installing Kokoro"
    & $vpy -m pip install "kokoro>=0.9.2" "misaki[en]>=0.9.3"
    if (-not (Get-Command espeak-ng -ErrorAction SilentlyContinue)) {
        Warn "espeak-ng is not installed. Kokoro works without it but mispronounces unusual words.
       Get it from https://github.com/espeak-ng/espeak-ng/releases"
    }
}

Info "Installing the vtrans package"
& $vpy -m pip install --no-deps -e $root
if ($LASTEXITCODE -ne 0) { Die "Editable install failed" }

# ------------------------------------------------------------- sanity check
Info "Verifying the install"
$checkGui = if ($NoGui) { "0" } else { "1" }
$env:VTRANS_CHECK_GUI = $checkGui
$check = @'
import os, sys
problems = []
try:
    import torch
    print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu   {torch.cuda.get_device_name(0)}")
    else:
        print("  gpu   none - conversions will run on the CPU and be much slower")
except Exception as exc:
    problems.append(f"torch: {exc}")

checks = [("faster_whisper", "faster-whisper"), ("transformers", "transformers"),
          ("piper.voice", "piper-tts"), ("soundfile", "soundfile"),
          ("yaml", "PyYAML"), ("vtrans", "vtrans")]
if os.environ.get("VTRANS_CHECK_GUI") == "1":
    checks += [("PySide6.QtWidgets", "PySide6"), ("vtrans_gui", "vtrans_gui")]
for module, label in checks:
    try:
        __import__(module)
        print(f"  ok    {label}")
    except Exception as exc:
        problems.append(f"{label}: {exc}")

try:
    from vtrans import binaries
    print(f"  ok    ffmpeg at {binaries.ffmpeg()}")
except Exception as exc:
    problems.append(f"ffmpeg: {exc}")

if problems:
    print("\nFAILED:", file=sys.stderr)
    for p in problems:
        print("  " + p, file=sys.stderr)
    sys.exit(1)
'@
$checkFile = Join-Path $env:TEMP "vtrans-verify.py"
# UTF8 without BOM: python reads source as UTF-8, and a BOM is a syntax error.
[System.IO.File]::WriteAllText($checkFile, $check, (New-Object System.Text.UTF8Encoding $false))
& $vpy $checkFile
$verifyCode = $LASTEXITCODE
Remove-Item $checkFile -ErrorAction SilentlyContinue
if ($verifyCode -ne 0) { Die "Verification failed - see the errors above." }

# ------------------------------------------------------------------ models
if ($SkipModels) {
    Warn "Skipping model download; the first conversion will fetch them."
} else {
    Info "Downloading models into models\ (a few GB, only happens once)"
    $dl = @()
    if ($AsrModel) { $dl += @("--asr-model", $AsrModel) }
    if ($Voice)    { $dl += @("--piper-voice", $Voice) }
    if ($WithKokoro) { $dl += "--kokoro" }
    if ($WithDemucs) { $dl += "--demucs" }
    & $vpy -m vtrans.download @dl
    if ($LASTEXITCODE -ne 0) { Die "Model download failed. Re-run this script to resume it." }
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
if (-not $NoGui) {
    Write-Host "  Open the desktop app:"
    Write-Host "    .\run_gui.bat"
    Write-Host ""
}
Write-Host "  Convert from the command line:"
Write-Host "    .\convert_video.bat input.mp4 -out output.mp4"
Write-Host ""
Write-Host "  All options:"
Write-Host "    .\convert_video.bat --help"
Write-Host ""
