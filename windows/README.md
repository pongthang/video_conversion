# Windows

Two ways to get this running on Windows. **Set up from a clone** is the one to
use on your own machine. **Build the installer** is for handing a single `.exe`
to someone else.

---

# Set up from a clone

Clones the source only — about 500 KB. Everything heavy (PyTorch, ffmpeg, the
models: roughly 3–8 GB depending on options) is downloaded by the setup script.

## Prerequisites

1. **Python 3.9–3.11** from https://www.python.org/downloads/ — tick
   **"Add python.exe to PATH"** during install. 3.11 is the safest choice;
   3.12+ does not have wheels for every dependency yet.
2. **Git** from https://git-scm.com/download/win
3. An **NVIDIA GPU with a current driver**, if you want it to be fast. Nothing
   else to install: the CUDA runtime arrives with PyTorch as pip wheels.

ffmpeg is *not* a prerequisite — the setup script downloads a static build into
`bin\` by itself.

## Steps

```powershell
git clone <repo-url> vtrans
cd vtrans
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`-ExecutionPolicy Bypass` is needed because the script is unsigned; it applies
to that one command only and changes nothing on the machine.

Setup takes 15–40 minutes, almost all of it downloading. It:

- finds a usable Python and builds a virtualenv in `.venv`
- downloads a static **ffmpeg** into `bin\`
- installs **PyTorch**, with CUDA if `nvidia-smi` reports a GPU, otherwise the
  CPU build (2.5 GB vs 200 MB)
- installs the pipeline dependencies and the desktop interface
- verifies every import and prints whether **CUDA is available**
- downloads the models

Then:

```powershell
.\run_gui.bat                                   # desktop app
.\convert_video.bat input.mp4 -out output.mp4   # command line
```

Re-running `setup.ps1` is safe and cheap: every step is skipped when already
satisfied, so it doubles as a repair and as the way to resume an interrupted
model download.

## Options

| Flag | Effect |
|---|---|
| `-Cpu` | CPU-only PyTorch even if a GPU is present |
| `-NoGui` | command line only, skip PySide6 |
| `-WithKokoro` | install the higher-quality Kokoro voice |
| `-WithDemucs` | keep the original music under the dub |
| `-SkipModels` | packages only; models download on first use |
| `-AsrModel small` | pre-download a specific Whisper size |
| `-Python C:\path\to\python.exe` | use a specific interpreter |

## If something goes wrong

**"No usable Python 3.9-3.11 found"** — the Microsoft Store ships a stub
`python.exe` on PATH that does nothing, which is why the script tests whether a
candidate actually runs. Install real Python from python.org, or pass
`-Python` explicitly.

**`cuda=False` in the verification** — the app will still work but every
conversion runs on the CPU and takes many times longer. Update the NVIDIA
driver and re-run `setup.ps1`.

**The app window does not appear** — run `.venv\Scripts\python.exe -m vtrans_gui`
from a terminal to see the error. `run_gui.bat` uses `pythonw.exe`, which has
no console.

**A model download stopped partway** — re-run `setup.ps1`; it resumes.

**`faster-whisper: DLL load failed ... An Application Control policy has blocked
this file`** — Windows **Smart App Control** is blocking CTranslate2, the engine
behind faster-whisper. It ships unsigned compiled DLLs, and Smart App Control
blocks binaries it has neither a signature nor a reputation for. PyTorch passes
the same check because it is common enough to be known, which is why the
verification reports `cuda=True` and then fails only on this one import.

Check what is enforcing it:

```powershell
Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -Name VerifiedAndReputablePolicyState
```

`1` is Smart App Control enforcing, `2` evaluation mode, `0` off. If the key is
absent, an enterprise WDAC policy is responsible and your administrator has to
make the exception.

There is no way to allowlist a single file under consumer Smart App Control, so
the options are:

- **Turn Smart App Control off** — Windows Security → App & browser control →
  Smart App Control settings → Off, then restart. Fastest, and keeps
  faster-whisper's speed and low VRAM use. **Microsoft does not allow Smart App
  Control to be switched back on afterwards without reinstalling Windows**, so
  treat it as permanent.
- **Use a machine without it.** Smart App Control only ever activates on clean
  installs of Windows 11; an upgraded machine will not have it.

Nothing needs reinstalling afterwards: the DLL is already on disk, it was only
prevented from loading. Re-run `setup.ps1` and it continues from the
verification step.

---

# Build the installer

For distributing a single `.exe` that installs everything. **The build has to
run on Windows** — Inno Setup is a Windows compiler.

## Build prerequisites

1. **Inno Setup 6** — https://jrsoftware.org/isdl.php (default install location)
2. **PowerShell 5+** — already on Windows 10/11
3. `tar` and `curl` — already on Windows 10 1803+

Nothing else. Python is *not* required on the build machine: `build.ps1`
downloads a standalone CPython and stages it into the installer.

## Build

```powershell
git clone <this repo> vtrans
cd vtrans
powershell -ExecutionPolicy Bypass -File windows\build.ps1
```

Takes about two minutes, mostly the ~25 MB CPython download. The result is:

```
windows\Output\VideoTranslator-1.0.0-Setup.exe     (~120 MB)
```

Useful switches:

| Switch | Effect |
|---|---|
| `-SkipRuntime` | reuse the already-staged `windows\runtime` |
| `-SkipInstaller` | stage the runtime only, do not compile |
| `-PythonVersion 3.11.9` | pin a different CPython |

## What the installer does

The installer is deliberately small. Bundling PyTorch with CUDA (~2.5 GB) plus
every model (1–6 GB) would mean an 8 GB download for everyone regardless of
what their machine can use. Instead it ships the app and a private CPython,
then fetches exactly what that machine needs:

1. Two wizard pages: **model tier** (Fast / Balanced / Best) and **GPU or CPU**
2. Installs to `%ProgramFiles%\VideoTranslator` (or per-user, no admin needed)
3. Runs `bootstrap.py`, which:
   - installs the CUDA or CPU build of PyTorch, chosen by probing `nvidia-smi`
   - downloads a **static ffmpeg** into `bin\` — nothing is installed
     system-wide, and the uninstaller removes it
   - installs the pipeline dependencies and PySide6
   - downloads the models for the chosen tier
   - verifies every import and reports `CUDA: True/False`
4. Creates Start-menu and optional desktop shortcuts

Every step is skipped when already satisfied, so **Repair Video
Translator** in the Start menu is a safe, cheap way to restore a broken install.

The uninstaller removes `models\`, `work\`, `bin\` and the installed packages,
so nothing is left behind.

## Files

| File | Role |
|---|---|
| `build.ps1` | stages CPython, compiles the installer |
| `installer.iss` | Inno Setup script: wizard pages, files, shortcuts |
| `bootstrap.py` | downloads PyTorch, ffmpeg and the models, with progress |
| `launcher.py` | what the shortcut runs; reports early failures in a message box |

`launcher.py` exists because shortcuts run `pythonw.exe`, which has no console:
without it, a missing dependency would make the app fail with no visible error
at all.

## Test checklist

Nothing below has been exercised — this machine has no Windows and no Wine.
Worth walking through on first install:

- [ ] installer runs without admin rights (per-user install)
- [ ] both wizard pages appear and the choice is honoured
- [ ] on an NVIDIA machine, verification prints `CUDA: True`
- [ ] on a machine with no GPU, the CPU build installs and the app still runs
- [ ] Start-menu and desktop shortcuts launch the app
- [ ] the app window opens with no console window behind it
- [ ] choosing a video shows the caption preview within a second or two
- [ ] moving the size and height sliders updates the preview
- [ ] a conversion completes and the output plays with English audio
- [ ] Cancel stops a conversion promptly and leaves no partial output
- [ ] selecting a not-yet-downloaded voice downloads it before converting
- [ ] "Repair" completes on an already-working install without re-downloading
- [ ] uninstall leaves no files behind in the install directory

## Known gaps

- **Unsigned.** SmartScreen will warn on first run. Signing needs a code
  signing certificate; without one, "More info → Run anyway" is the path.
- **Paths with non-ASCII characters** in the install directory or video
  filenames are handled (UTF-8 is forced for the child process) but untested
  on a non-English Windows.
- `en_GB-northern_english_male-medium` and the Kokoro voices are offered in the
  UI but only fetched on demand, so their first use needs a network connection.
