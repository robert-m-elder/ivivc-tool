# Run the IVIVC App locally on macOS

This package can run the IVIVC App on your own Mac rather than on an external IVIVC web server. The local edition uses the same Python analysis code as the web app and opens the interface in your normal web browser.

## First-time setup

1. Keep the full IVIVC App folder together; do not move the `.command` files out of `local_install/macos/`.
2. Double-click **Setup IVIVC App.command**.
3. macOS may ask whether you want to open the downloaded script. If Finder blocks it, Control-click the file, choose **Open**, and confirm. Institutional security settings may prevent unapproved scripts from running.
4. Follow the prompts in the Terminal window. The first setup requires an internet connection to download Miniforge and the Python packages.
5. When setup reports success, close the Terminal window.

The setup creates a private installation under `~/.ivivc-app/`. It does not require `sudo`, does not install into system directories, does not initialize Conda in your shell, and does not modify another Python or Conda installation.

## Start the app

Double-click **Start IVIVC App.command**. A Terminal window remains open while the app is running and the app opens in your default browser. Keep that Terminal window open during use. Press **Control-C** in the Terminal window, or close the window, to stop the local app.

The local server listens only on `127.0.0.1` (localhost), so it is not exposed to other computers on the network.

## Data privacy and network access

Uploaded datasets are sent by your browser only to the IVIVC App process running on your own Mac; they are not uploaded to an external IVIVC server. The current app interface still references some third-party JavaScript/CSS libraries from public content-delivery networks, so normal use is not yet fully offline. This local prototype should therefore not be interpreted as a no-network build. Those front-end assets can be packaged locally in a later step if fully offline operation is required.

## Updating this copy of the app

After replacing the app source with a newer compatible version, rerun **Setup IVIVC App.command**. The script updates the dedicated environment from `environment-macos.yml` and runs a basic import check.

## Remove the local environment

To remove the private Miniforge installation and IVIVC environment, stop the app and delete the folder:

`~/.ivivc-app`

This does not remove the IVIVC App source folder itself.

### If setup reports an SSL certificate error

Some managed or institutional networks inspect HTTPS traffic using an organization-specific certificate. The setup script first uses macOS certificate verification and does not disable TLS checks. If the Miniforge command-line download still cannot be verified, setup opens the same official Miniforge download in the default browser and asks you to save it in Downloads before continuing.

The private Conda environment is configured to use the macOS system certificate trust store when downloading packages. If Conda still reports a certificate-verification error, your organization's certificate may not be available to command-line applications; contact your IT/support group rather than disabling SSL verification.
