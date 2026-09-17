<#
.SYNOPSIS
    Builds the Miko Video Translator installer. Run this on Windows.

.DESCRIPTION
    Downloads a standalone CPython, stages it into windows\runtime, then calls
    the Inno Setup compiler to produce windows\Output\MikoVideoTranslator-1.0.0-Setup.exe.

    A standalone CPython is used rather than the official "embeddable" zip
    because the embeddable build omits tkinter, ensurepip and the full standard
    library, and it cannot reliably pip-install packages with compiled
    extensions - which is most of this project's dependencies.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File windows\build.ps1
#>
[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11.9",
    [string]$ReleaseTag    = "20240415",
    [switch]$SkipRuntime,
    [switch]$SkipInstaller
)

# Deliberately NOT "Stop". With Stop, PowerShell 5.1 turns anything a native
# command writes to stderr into a terminating error, which breaks this script
# in two ways: probes like `python -c "import torch"` are *meant* to fail when
# the package is absent, and pip and huggingface_hub write ordinary progress
# and warnings to stderr. Every external call below checks $LASTEXITCODE
# explicitly instead, and the cmdlets that must throw say -ErrorAction Stop.
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here

function Info($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  ok $m" -ForegroundColor Green }
function Die($m)  { Write-Host " err $m" -ForegroundColor Red; exit 1 }

# ----------------------------------------------------------- standalone Python
$runtime = Join-Path $here "runtime"
if (-not $SkipRuntime) {
    if (Test-Path (Join-Path $runtime "python.exe")) {
        Ok "runtime already staged"
    } else {
        Info "Downloading standalone CPython $PythonVersion"
        $file = "cpython-$PythonVersion+$ReleaseTag-x86_64-pc-windows-msvc-install_only.tar.gz"
        $url  = "https://github.com/indygreg/python-build-standalone/releases/download/$ReleaseTag/$file"
        $vendor = Join-Path $here "vendor"
        New-Item -ItemType Directory -Force -Path $vendor | Out-Null
        $archive = Join-Path $vendor $file

        if (-not (Test-Path $archive)) {
            try {
                Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing -ErrorAction Stop
            } catch {
                Die "Could not download CPython: $_`nCheck the version/tag, or place $file in $vendor manually."
            }
        }

        Info "Extracting"
        New-Item -ItemType Directory -Force -Path $vendor | Out-Null
        tar -xzf $archive -C $vendor
        if ($LASTEXITCODE -ne 0) { Die "tar failed to extract $archive" }

        $extracted = Join-Path $vendor "python"
        if (-not (Test-Path $extracted)) { Die "Expected $extracted after extraction" }
        if (Test-Path $runtime) { Remove-Item -Recurse -Force $runtime }
        Move-Item $extracted $runtime

        # Trim what the app will never use; it is all re-downloadable anyway.
        foreach ($d in @("Lib\test", "Lib\idlelib", "Lib\tkinter", "tcl")) {
            $p = Join-Path $runtime $d
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
        Ok "runtime staged at $runtime"
    }

    $py = Join-Path $runtime "python.exe"
    & $py -c "import sys; print('runtime python', sys.version.split()[0])"
    if ($LASTEXITCODE -ne 0) { Die "The staged runtime does not run" }
}

# ------------------------------------------------------------------- icon
$assets = Join-Path $root "src\vtrans_gui\assets"
if (-not (Test-Path (Join-Path $assets "app.ico"))) {
    Write-Host "  note: no app.ico in $assets; the installer will use the default icon" -ForegroundColor Yellow
}

# --------------------------------------------------------------- Inno Setup
if (-not $SkipInstaller) {
    $iscc = $null
    foreach ($candidate in @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe")) {
        if (Test-Path $candidate) { $iscc = $candidate; break }
    }
    if (-not $iscc) {
        $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
        if ($cmd) { $iscc = $cmd.Source }
    }
    if (-not $iscc) {
        Die "Inno Setup 6 not found. Install it from https://jrsoftware.org/isdl.php and re-run."
    }

    Info "Compiling the installer"
    & $iscc (Join-Path $here "installer.iss")
    if ($LASTEXITCODE -ne 0) { Die "Inno Setup failed" }

    $out = Get-ChildItem (Join-Path $here "Output") -Filter *.exe | Select-Object -First 1
    Ok "installer built: $($out.FullName)  ($([math]::Round($out.Length/1MB,1)) MB)"
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
