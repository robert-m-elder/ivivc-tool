@echo off
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..\..") do set "APP_DIR=%%~fI"
set "INSTALL_ROOT=%USERPROFILE%\.ivivc-app"
set "ENV_DIR=%INSTALL_ROOT%\env"
set "PYTHON=%ENV_DIR%\python.exe"
set "LAUNCHER=%APP_DIR%\local_launcher.py"

cls
echo IVIVC App - local Windows launcher
echo ====================================
echo.

if not exist "%PYTHON%" (
    echo The IVIVC App local environment has not been set up yet.
    echo First double-click "Setup IVIVC App.bat".
    echo.
    pause
    exit /b 1
)

if not exist "%LAUNCHER%" (
    echo Cannot find the IVIVC App local launcher:
    echo   %LAUNCHER%
    echo Keep the complete IVIVC App folder together and try again.
    echo.
    pause
    exit /b 1
)

pushd "%APP_DIR%" >nul
"%PYTHON%" "%LAUNCHER%"
set "STATUS=%ERRORLEVEL%"
popd >nul

echo.
if not "%STATUS%"=="0" (
    echo The IVIVC App stopped with an error ^(status %STATUS%^).
) else (
    echo IVIVC App stopped.
)
echo.
pause
exit /b %STATUS%
