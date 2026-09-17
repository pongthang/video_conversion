; Inno Setup script for Video Translator.
; Build with:  iscc windows\installer.iss   (or run windows\build.ps1)
;
; The installer is intentionally small. It ships the application, a standalone
; CPython and the bootstrap script; PyTorch, ffmpeg and the models are fetched
; during installation, sized to the machine and to the tier the user picks.

#define AppName        "Video Translator"
#define AppShortName   "VideoTranslator"
#define AppVersion     "1.0.0"
#define AppPublisher   "Video Translator"
#define AppExeName     "VideoTranslator.exe"

[Setup]
AppId={{8E2A6C41-7B3D-4F58-9A12-5C7E0D4B9F31}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppShortName}
DefaultGroupName={#AppName}
OutputDir=Output
OutputBaseFilename={#AppShortName}-{#AppVersion}-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The models need room; refuse early rather than failing halfway through.
ExtraDiskSpaceRequired=1073741824
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
SetupLogging=yes
UninstallDisplayName={#AppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
; The application itself.
Source: "..\src\*";            DestDir: "{app}\src";    Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\config\*";         DestDir: "{app}\config"; Flags: ignoreversion recursesubdirs
Source: "..\requirements.txt"; DestDir: "{app}";        Flags: ignoreversion
Source: "..\README.md";        DestDir: "{app}";        Flags: ignoreversion
Source: "bootstrap.py";        DestDir: "{app}\windows"; Flags: ignoreversion
Source: "launcher.py";         DestDir: "{app}\windows"; Flags: ignoreversion
; The standalone CPython, unpacked by build.ps1 into windows\runtime.
Source: "runtime\*";           DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";          Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\windows\launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\src\vtrans_gui\assets\app.ico"
Name: "{group}\Repair {#AppName}";   Filename: "{app}\runtime\python.exe";  Parameters: """{app}\windows\bootstrap.py"""; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}";    Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\windows\launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\src\vtrans_gui\assets\app.ico"; Tasks: desktopicon

[Run]
; The long part: fetch PyTorch, ffmpeg and the chosen models.
Filename: "{app}\runtime\python.exe"; \
  Parameters: """{app}\windows\bootstrap.py"" --preset {code:GetPreset}{code:GetCpuFlag}"; \
  WorkingDir: "{app}"; \
  StatusMsg: "Downloading components and models - this can take a while..."; \
  Flags: runhidden waituntilterminated
Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\windows\launcher.py"""; \
  Description: "Start {#AppName}"; WorkingDir: "{app}"; \
  Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Everything the app created after installation, so nothing is left behind.
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\work"
Type: filesandordirs; Name: "{app}\bin"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages"
Type: filesandordirs; Name: "{app}\src\vtrans\__pycache__"
Type: filesandordirs; Name: "{app}\src\vtrans_gui\__pycache__"

[Code]
var
  ModelPage: TInputOptionWizardPage;
  DevicePage: TInputOptionWizardPage;

procedure InitializeWizard;
begin
  ModelPage := CreateInputOptionPage(wpSelectTasks,
    'Choose the models', 'How accurate should the translation be?',
    'Bigger models transcribe and translate noticeably better but take longer to ' +
    'download and to run. You can change this later inside the application; ' +
    'additional models are downloaded on demand.',
    True, False);
  ModelPage.Add('Fast'          + #13#10 + '    about 0.9 GB download - small Whisper, light translation. Fine on any machine.');
  ModelPage.Add('Balanced'      + #13#10 + '    about 4.1 GB download - near-best quality at a fraction of the cost. Recommended.');
  ModelPage.Add('Best quality'  + #13#10 + '    about 5.9 GB download - largest models. Needs a 6 GB graphics card to run well.');
  ModelPage.SelectedValueIndex := 1;

  DevicePage := CreateInputOptionPage(ModelPage.ID,
    'Graphics card', 'Should the app use your NVIDIA graphics card?',
    'Using the GPU is many times faster. If you do not have an NVIDIA card, or ' +
    'you would rather not use it, choose CPU only: the download is about 2.3 GB ' +
    'smaller, but conversions take considerably longer.',
    True, False);
  DevicePage.Add('Use the graphics card when one is available (recommended)');
  DevicePage.Add('CPU only');
  DevicePage.SelectedValueIndex := 0;
end;

function GetPreset(Param: String): String;
begin
  case ModelPage.SelectedValueIndex of
    0: Result := 'fast';
    2: Result := 'best';
  else
    Result := 'balanced';
  end;
end;

function GetCpuFlag(Param: String): String;
begin
  if DevicePage.SelectedValueIndex = 1 then
    Result := ' --cpu-only'
  else
    Result := '';
end;
