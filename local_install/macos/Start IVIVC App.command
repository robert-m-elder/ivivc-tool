#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_ROOT="$HOME/.ivivc-app"
CONDA="$INSTALL_ROOT/miniforge3/bin/conda"
ENV_DIR="$INSTALL_ROOT/env"
LAUNCHER="$APP_DIR/local_launcher.py"

clear
echo "IVIVC App - local macOS launcher"
echo "================================="
echo

if [ ! -x "$CONDA" ] || [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "The IVIVC App local environment has not been set up yet."
    echo "First double-click 'Setup IVIVC App.command'."
    echo
    read -r -p "Press Return to close this window..." _
    exit 1
fi

cd "$APP_DIR" || exit 1
"$CONDA" run --prefix "$ENV_DIR" --no-capture-output python "$LAUNCHER"
status=$?

echo
if [ "$status" -ne 0 ]; then
    echo "The IVIVC App stopped with an error (status $status)."
else
    echo "IVIVC App stopped."
fi
read -r -p "Press Return to close this window..." _
exit "$status"
