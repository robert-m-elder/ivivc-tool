"""Run the IVIVC App locally and open it in the user's default browser."""

from __future__ import annotations

import socket
import threading
import webbrowser

from werkzeug.serving import make_server

from app import app


HOST = "127.0.0.1"


def _available_port() -> int:
    """Ask the operating system for an unused localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def main() -> None:
    port = _available_port()
    url = f"http://{HOST}:{port}/"
    server = make_server(HOST, port, app, threaded=True)

    print("IVIVC App is running locally on this computer.")
    print(f"Opening {url}")
    print("No app data are sent to an external IVIVC server.")
    print("Keep this window open while using the app.")
    print("Press Control-C or close this window to stop the app.\n")

    threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping IVIVC App.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
