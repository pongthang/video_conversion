@echo off
REM Convert a Chinese video into an English-dubbed, English-subtitled video.
REM
REM   convert_video.bat input.mp4 -out output.mp4
REM   convert_video.bat "https://youtu.be/XXXX" -out output.mp4
REM   convert_video.bat --help
setlocal
set "PROJECT_DIR=%~dp0"
set "VPY=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%VPY%" (
  echo Virtualenv not found at %PROJECT_DIR%.venv
  echo.
  echo Run setup first:
  echo     powershell -ExecutionPolicy Bypass -File setup.ps1
  exit /b 1
)

REM Keep model downloads inside the project directory.
if not defined HF_HOME    set "HF_HOME=%PROJECT_DIR%models\hf"
if not defined TORCH_HOME set "TORCH_HOME=%PROJECT_DIR%models\torch"
set "TOKENIZERS_PARALLELISM=false"
set "PATH=%PROJECT_DIR%bin;%PROJECT_DIR%.venv\Scripts;%PATH%"
REM Safety net: the editable install normally makes vtrans importable, but it
REM can fail quietly on paths with spaces. Naming src directly costs nothing.
set "PYTHONPATH=%PROJECT_DIR%src;%PYTHONPATH%"

cd /d "%PROJECT_DIR%"
"%VPY%" -m vtrans %*
