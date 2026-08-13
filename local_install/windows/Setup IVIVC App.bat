@echo off
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..\..") do set "APP_DIR=%%~fI"
set "INSTALL_ROOT=%USERPROFILE%\.ivivc-app"
set "MINIFORGE_DIR=%INSTALL_ROOT%\mf"
set "ENV_DIR=%INSTALL_ROOT%\env"
set "CONDA=%MINIFORGE_DIR%\condabin\conda.bat"
set "ENV_FILE=%APP_DIR%\environment-windows.yml"
set "CONDARC_FILE=%INSTALL_ROOT%\condarc"
set "MINIFORGE_VERSION=26.1.0-0"
set "ASSET=Miniforge3-Windows-x86_64.exe"
set "DOWNLOAD_URL=https://github.com/conda-forge/miniforge/releases/download/%MINIFORGE_VERSION%/%ASSET%"
set "INSTALLER=%INSTALL_ROOT%\%ASSET%"
set "LOG_FILE=%INSTALL_ROOT%\setup.log"

cls
echo IVIVC App - Windows local setup
echo =================================
echo.
echo This setup installs a private Miniforge copy and IVIVC Python environment
echo under:
echo   %INSTALL_ROOT%
echo.
echo It uses a per-user installation and does not require administrator access,
echo modify PATH, initialize Conda in your shell, or change an existing Python
echo or Conda installation.
echo.
echo An internet connection is required for this one-time setup.
echo.
pause

if /I "%PROCESSOR_ARCHITECTURE%"=="AMD64" goto architecture_ok
if /I "%PROCESSOR_ARCHITEW6432%"=="AMD64" goto architecture_ok

echo.
echo This first Windows prototype supports 64-bit x86 Windows only.
echo Detected processor architecture: %PROCESSOR_ARCHITECTURE%
goto fail

:architecture_ok
if not exist "%INSTALL_ROOT%" mkdir "%INSTALL_ROOT%" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Could not create the private installation folder:
    echo   %INSTALL_ROOT%
    goto fail
)

rem Keep a small persistent setup log for troubleshooting.
> "%LOG_FILE%" echo IVIVC App Windows setup log
>> "%LOG_FILE%" echo Started: %DATE% %TIME%
>> "%LOG_FILE%" echo App directory: %APP_DIR%
>> "%LOG_FILE%" echo Install root: %INSTALL_ROOT%
>> "%LOG_FILE%" echo Miniforge target: %MINIFORGE_DIR%
>> "%LOG_FILE%" echo Miniforge version: %MINIFORGE_VERSION%
>> "%LOG_FILE%" echo Installer path: %INSTALLER%
>> "%LOG_FILE%" echo.

if exist "%CONDA%" (
    echo.
    echo Using the IVIVC App Miniforge installation already present.
    >> "%LOG_FILE%" echo Existing Miniforge installation found.
    goto miniforge_ready
)

echo.
echo Downloading tested Miniforge version %MINIFORGE_VERSION% from the official
echo conda-forge release...
>> "%LOG_FILE%" echo Downloading %DOWNLOAD_URL%
if exist "%INSTALLER%" del /q "%INSTALLER%" >nul 2>&1

powershell.exe -NoLogo -NoProfile -Command ^
  "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri $env:DOWNLOAD_URL -OutFile $env:INSTALLER"

if errorlevel 1 goto download_fallback
if not exist "%INSTALLER%" goto download_fallback
for %%I in ("%INSTALLER%") do if %%~zI LSS 1000000 goto download_fallback

goto install_miniforge

:download_fallback
>> "%LOG_FILE%" echo Command-line download failed or produced an invalid installer.
echo.
echo The secure command-line download did not complete. This can happen on
echo managed networks that inspect HTTPS traffic. Setup will not disable
echo certificate verification.
echo.
echo Your browser will open the same official Miniforge download. Save the file
echo in your Downloads folder with this name:
echo   %ASSET%
echo.
start "" "%DOWNLOAD_URL%"
pause

set "MANUAL_INSTALLER=%USERPROFILE%\Downloads\%ASSET%"
if not exist "%MANUAL_INSTALLER%" (
    echo.
    echo Cannot find the downloaded installer at:
    echo   %MANUAL_INSTALLER%
    echo Move or rename the Miniforge installer to that location and run setup again.
    goto fail
)
copy /y "%MANUAL_INSTALLER%" "%INSTALLER%" >nul
if errorlevel 1 goto fail
>> "%LOG_FILE%" echo Browser-downloaded installer copied from %MANUAL_INSTALLER%

:install_miniforge
echo.
echo Installing tested Miniforge version %MINIFORGE_VERSION% for this user...
echo.
echo The Miniforge installer will open with the IVIVC App destination pre-filled.
echo Keep the per-user / "Just Me" installation and the destination shown below:
echo   %MINIFORGE_DIR%
echo.
echo Setup will continue automatically after the installer window is closed.
echo.
>> "%LOG_FILE%" echo Starting interactive Miniforge installation with pre-filled options.
start "" /wait "%INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /D=%MINIFORGE_DIR%
set "MINIFORGE_EXIT=%ERRORLEVEL%"
>> "%LOG_FILE%" echo Interactive installer exit code: %MINIFORGE_EXIT%

rem Verify the expected conda launcher instead of relying only on the installer
rem exit code, since installer failures may not always return a useful status.
if not exist "%CONDA%" (
    >> "%LOG_FILE%" echo Interactive installation did not create %CONDA%
    echo.
    echo Miniforge %MINIFORGE_VERSION% did not install successfully at:
    echo   %MINIFORGE_DIR%
    echo.
    echo Setup details were written to:
    echo   %LOG_FILE%
    echo.
    echo The downloaded installer has been kept for troubleshooting at:
    echo   %INSTALLER%
    echo.
    echo Windows application-control software can block installation even for a
    echo per-user installer. If your organization manages this computer, you may
    echo need its approval for Miniforge.
    goto fail
)

>> "%LOG_FILE%" echo Miniforge installation verified at %CONDA%
if exist "%INSTALLER%" del /q "%INSTALLER%" >nul 2>&1

:miniforge_ready
if not exist "%ENV_FILE%" (
    echo.
    echo Cannot find the environment definition:
    echo   %ENV_FILE%
    goto fail
)

rem Keep Conda configuration private to this IVIVC App installation. Current
rem Conda releases can use the Windows certificate store through truststore.
> "%CONDARC_FILE%" echo channels:
>> "%CONDARC_FILE%" echo   - conda-forge
>> "%CONDARC_FILE%" echo channel_priority: strict
>> "%CONDARC_FILE%" echo ssl_verify: truststore
set "CONDARC=%CONDARC_FILE%"

echo.
echo Creating/updating the IVIVC App Python environment...
>> "%LOG_FILE%" echo Creating/updating Conda environment at %ENV_DIR%
if exist "%ENV_DIR%\python.exe" (
    call "%CONDA%" env update --prefix "%ENV_DIR%" --file "%ENV_FILE%" --prune -y
) else (
    call "%CONDA%" env create --prefix "%ENV_DIR%" --file "%ENV_FILE%" -y
)
set "CONDA_STATUS=%ERRORLEVEL%"
>> "%LOG_FILE%" echo Conda environment command exit code: %CONDA_STATUS%
if not "%CONDA_STATUS%"=="0" goto conda_fail

if not exist "%ENV_DIR%\python.exe" (
    echo.
    echo The environment command completed but Python was not found at:
    echo   %ENV_DIR%\python.exe
    goto fail
)

echo.
echo Checking the local installation...
pushd "%APP_DIR%" >nul
"%ENV_DIR%\python.exe" -c "import flask,numpy,pandas,scipy,sklearn,uncertainties,yaml,openpyxl,docx,plotly; from app import app; print('Python dependencies and IVIVC App imports: OK')"
set "CHECK_STATUS=%ERRORLEVEL%"
popd >nul
>> "%LOG_FILE%" echo Import check exit code: %CHECK_STATUS%
if not "%CHECK_STATUS%"=="0" goto fail

>> "%LOG_FILE%" echo Setup completed successfully: %DATE% %TIME%
echo.
echo IVIVC App setup completed successfully.
echo You can close this window and double-click "Start IVIVC App.bat".
echo.
pause
exit /b 0

:conda_fail
echo.
echo Conda could not create the IVIVC App environment.
echo If the message above reports an SSL/certificate error, your organization's
echo certificate may not be available to command-line applications. Contact your
echo IT/support group rather than disabling SSL verification.

goto fail

:fail
if exist "%INSTALL_ROOT%" >> "%LOG_FILE%" echo Setup failed: %DATE% %TIME%
echo.
echo IVIVC App setup did not complete.
echo Review the message above or share it with the app support contact.
if exist "%LOG_FILE%" echo Setup log: %LOG_FILE%
echo.
pause
exit /b 1
