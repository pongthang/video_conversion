@echo off
REM Launch the desktop application.
REM
REM   run_gui.bat
REM
REM Runs through windows\launcher.py rather than -m vtrans_gui directly: it
REM reports an import failure in a message box, and pythonw.exe has no console
REM to print one to otherwise.
REM
REM The CUDA libraries need no PATH setup here - vtrans.runtime calls
REM os.add_dll_directory for the nvidia-* wheels at start-up, which is what
REM lets the GPU work when this is started from a shortcut.
setlocal
set "PROJECT_DIR=%~dp0"
set "VPYW=%PROJECT_DIR%.venv\Scripts\pythonw.exe"

if not exist "%VPYW%" (
  echo Virtualenv not found at %PROJECT_DIR%.venv
  echo.
  echo Run setup first:
  echo     powershell -ExecutionPolicy Bypass -File setup.ps1
  echo.
  pause
  exit /b 1
)

cd /d "%PROJECT_DIR%"
start "" "%VPYW%" "%PROJECT_DIR%windows\launcher.py" %*
