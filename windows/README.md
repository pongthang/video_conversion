# Building the Windows installer

The application code is finished and tested. This directory turns it into
`MikoVideoTranslator-1.0.0-Setup.exe`. **The build has to run on Windows** —
Inno Setup is a Windows compiler, so the `.exe` cannot be produced on the Linux
development machine.

## One-time prerequisites

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
windows\Output\MikoVideoTranslator-1.0.0-Setup.exe     (~120 MB)
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
2. Installs to `%ProgramFiles%\MikoVideoTranslator` (or per-user, no admin needed)
3. Runs `bootstrap.py`, which:
   - installs the CUDA or CPU build of PyTorch, chosen by probing `nvidia-smi`
   - downloads a **static ffmpeg** into `bin\` — nothing is installed
     system-wide, and the uninstaller removes it
   - installs the pipeline dependencies and PySide6
   - downloads the models for the chosen tier
   - verifies every import and reports `CUDA: True/False`
4. Creates Start-menu and optional desktop shortcuts

Every step is skipped when already satisfied, so **Repair Miko Video
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
