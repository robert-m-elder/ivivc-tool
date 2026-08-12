#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_ROOT="$HOME/.ivivc-app"
MINIFORGE_DIR="$INSTALL_ROOT/miniforge3"
ENV_DIR="$INSTALL_ROOT/env"
CONDA="$MINIFORGE_DIR/bin/conda"
ENV_FILE="$APP_DIR/environment-macos.yml"
CONDARC_FILE="$INSTALL_ROOT/condarc"

finish() {
    status=$1
    echo
    if [ "$status" -eq 0 ]; then
        echo "IVIVC App setup completed successfully."
        echo "You can close this window and double-click 'Start IVIVC App.command'."
    else
        echo "IVIVC App setup did not complete."
        echo "Review the message above or share it with the app support contact."
    fi
    echo
    read -r -p "Press Return to close this window..." _
    exit "$status"
}

trap 'finish 1' ERR

clear
echo "IVIVC App - macOS local setup"
echo "================================"
echo
echo "This setup installs a private Miniforge copy and IVIVC Python environment"
echo "under: $INSTALL_ROOT"
echo "It does not require administrator/root access and does not modify your"
echo "existing Python or Conda installations."
echo
echo "An internet connection is required for this one-time setup."
echo
read -r -p "Press Return to continue, or close this window to cancel..." _

mkdir -p "$INSTALL_ROOT"

if [ ! -x "$CONDA" ]; then
    machine="$(uname -m)"
    case "$machine" in
        arm64)
            asset="Miniforge3-MacOSX-arm64.sh"
            ;;
        x86_64)
            asset="Miniforge3-MacOSX-x86_64.sh"
            ;;
        *)
            echo "Unsupported Mac processor architecture: $machine"
            finish 1
            ;;
    esac

    tmp_dir="$(mktemp -d)"
    installer="$tmp_dir/$asset"
    download_url="https://github.com/conda-forge/miniforge/releases/latest/download/$asset"
    echo
    echo "Downloading Miniforge for $machine..."

    # Prefer Apple's system curl so the setup does not inherit certificate
    # settings from Homebrew/MacPorts or another user-installed curl.
    CURL_BIN="/usr/bin/curl"
    if [ ! -x "$CURL_BIN" ]; then
        CURL_BIN="$(command -v curl || true)"
    fi

    curl_ok=0
    if [ -n "$CURL_BIN" ]; then
        # Newer curl builds can explicitly consult the native macOS trust
        # store. Older Apple curl builds do so through their normal backend
        # and do not recognize --ca-native, so use it only when available.
        if "$CURL_BIN" --help all 2>/dev/null | grep -q -- '--ca-native'; then
            if "$CURL_BIN" --fail --location --proto '=https' --tlsv1.2 --ca-native \
                "$download_url" --output "$installer"; then
                curl_ok=1
            fi
        else
            if "$CURL_BIN" --fail --location --proto '=https' --tlsv1.2 \
                "$download_url" --output "$installer"; then
                curl_ok=1
            fi
        fi
    fi

    if [ "$curl_ok" -ne 1 ]; then
        echo
        echo "The secure command-line download could not verify the HTTPS certificate."
        echo "This can occur on institutional networks that inspect HTTPS traffic."
        echo "The setup will not bypass certificate verification."
        echo
        echo "A browser window will open to the same official Miniforge download."
        echo "Save the downloaded file as:"
        echo "  $HOME/Downloads/$asset"
        echo
        open "$download_url" || true
        read -r -p "After the download finishes, press Return to continue..." _

        manual_installer="$HOME/Downloads/$asset"
        if [ ! -f "$manual_installer" ]; then
            echo
            echo "Cannot find the downloaded installer at:"
            echo "  $manual_installer"
            echo "Move or rename the Miniforge installer to that location and run setup again."
            rm -rf "$tmp_dir"
            finish 1
        fi
        cp "$manual_installer" "$installer"
    fi

    echo "Installing Miniforge for this user..."
    bash "$installer" -b -p "$MINIFORGE_DIR"
    rm -rf "$tmp_dir"
else
    echo
    echo "Using the IVIVC App Miniforge installation already present."
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Cannot find environment definition: $ENV_FILE"
    finish 1
fi

# Keep Conda configuration private to the IVIVC App installation. Using
# 'truststore' tells current Conda releases to consult the operating system
# certificate store, which helps on managed Macs whose organization CA is
# installed in the macOS Keychain.
cat > "$CONDARC_FILE" <<'EOF_CONDARC'
channels:
  - conda-forge
channel_priority: strict
ssl_verify: truststore
EOF_CONDARC

echo
echo "Creating/updating the IVIVC App Python environment..."
if [ -x "$ENV_DIR/bin/python" ]; then
    CONDARC="$CONDARC_FILE" "$CONDA" env update --prefix "$ENV_DIR" --file "$ENV_FILE" --prune -y
else
    CONDARC="$CONDARC_FILE" "$CONDA" env create --prefix "$ENV_DIR" --file "$ENV_FILE" -y
fi

echo
echo "Checking the local installation..."
cd "$APP_DIR" || finish 1
CONDARC="$CONDARC_FILE" "$CONDA" run --prefix "$ENV_DIR" --no-capture-output python - <<'PY'
import flask
import numpy
import pandas
import scipy
import sklearn
import uncertainties
import yaml
import openpyxl
import docx
import plotly
from app import app
print("Python dependencies and IVIVC App imports: OK")
PY

finish 0
