# Run the IVIVC App locally on Windows

This package can run the IVIVC App on your own Windows computer rather than on an external IVIVC web server. The local edition uses the same Python analysis code as the web app and opens the interface in your normal web browser.

This first Windows prototype is intended for 64-bit x86 Windows 10 or later.

## First-time setup

1. Keep the full IVIVC App folder together; do not move the `.bat` files out of `local_install/windows/`.
2. Double-click **Setup IVIVC App.bat**.
3. Windows may show a security warning for files downloaded from the internet. Follow your organization's normal procedure for approving downloaded software. Institutional application-control settings may prevent Miniforge or the setup script from running even though administrator rights are not required by the installer itself.
4. Follow the prompts in the Command Prompt window. The first setup requires an internet connection to download Miniforge and the Python packages.
5. When setup reports success, close the Command Prompt window.

The setup uses Miniforge's per-user (`JustMe`) installation mode and creates a private installation under `%USERPROFILE%\.ivivc-app\`. It does not require administrator/root access, does not modify the system or user PATH, does not initialize Conda in PowerShell/Command Prompt, and does not modify another Python or Conda installation.

## Start the app

Double-click **Start IVIVC App.bat**. A Command Prompt window remains open while the app is running and the app opens in your default browser. Keep that window open during use. Press **Control-C** in the Command Prompt window, or close the window, to stop the local app.

The local server listens only on `127.0.0.1` (localhost), so it is not exposed to other computers on the network.

## Data privacy and network access

Uploaded datasets are sent by your browser only to the IVIVC App process running on your own computer; they are not uploaded to an external IVIVC server. The current app interface still references some third-party JavaScript/CSS libraries from public content-delivery networks, so normal use is not yet fully offline. This local prototype should therefore not be interpreted as a no-network build. Those front-end assets can be packaged locally in a later step if fully offline operation is required.

## Updating this copy of the app

After replacing the app source with a newer compatible version, rerun **Setup IVIVC App.bat**. The script updates the dedicated environment from `environment-windows.yml` and runs a basic import check.

## Remove the local environment

Stop the app and delete this folder from your user profile:

`%USERPROFILE%\.ivivc-app`

This removes the private Miniforge installation and IVIVC Python environment but does not remove the IVIVC App source folder itself.

## If setup reports an SSL certificate error

Some managed or institutional networks inspect HTTPS traffic using an organization-specific certificate. The setup uses Windows HTTPS certificate verification and does not disable TLS checks. If the command-line Miniforge download cannot be verified, setup opens the same official Miniforge download in the default browser and asks you to save it in your Downloads folder before continuing.

The private Conda environment is configured to use the operating-system certificate trust store when downloading packages. If Conda still reports a certificate-verification error, contact your IT/support group rather than disabling SSL verification.

## If Miniforge installation is blocked

The setup uses the official Miniforge silent per-user installation mode and verifies that Conda was actually installed before continuing. Some managed Windows computers use application-control policies that can block downloaded executables or files unpacked by installers. If the setup reports that Miniforge did not install, follow your organization's normal software-approval process or share the setup message with IT/support.
