"""Serve the zcmir web app locally and open it in the browser.

    zcmir                        # or: python -m zcmir
    zcmir --port 8000 --no-browser

The app runs entirely in the browser (zcmir.wasm on WebGPU): this only serves its static files on
127.0.0.1, which browsers treat as a secure context, as WebGPU requires. Needs a browser with
WebGPU and JavaScript Promise Integration (Chrome / Edge 137+). Images never leave the machine.

The files come from the installed package (``zcmir/web``, put there when the wheel is built) or,
in a source checkout, from the repository (``web/app``, ``web/zcmir.js`` and
``zig-out/web/zcmir.wasm`` after ``zig build wasm``).
"""
import argparse
import functools
import http.server
import socket
import sys
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
    ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm",
    ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
}


def site():
    """(root directory, {url path: file}) of the app: the packaged copy, else the source checkout."""
    packaged = HERE / "web"
    if (packaged / "index.html").is_file():
        return packaged, {}
    app = REPO / "web" / "app"
    wasm = REPO / "zig-out" / "web" / "zcmir.wasm"
    if not (app / "index.html").is_file():
        raise SystemExit("zcmir: the web app is not in this installation (a wheel built by scripts/build_wheels.py includes it)")
    if not wasm.is_file():
        raise SystemExit(f"zcmir: {wasm} missing; build it with `zig build wasm`")
    return app, {"/zcmir/zcmir.js": REPO / "web" / "zcmir.js", "/zcmir/zcmir.wasm": wasm}


class _Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **TYPES}

    def __init__(self, *args, files=None, **kwargs):
        self._files = files or {}
        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        p = path.split("?", 1)[0].split("#", 1)[0]
        if p in self._files:
            return str(self._files[p])
        return super().translate_path(path)

    def setup(self):
        super().setup()
        # One write per response, unbuffered (some Windows loopback filters stall on many small sends).
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def copyfile(self, source, outputfile):
        outputfile.write(source.read())

    def log_message(self, fmt, *args):
        pass


def _free_port(preferred):
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise SystemExit("zcmir: no free port")


def serve(port=4180, open_browser=True, block=True):
    """Serve the app on http://127.0.0.1:<port>/ (the next free port if taken); returns the URL.
    With block=False the server runs on a daemon thread (e.g. from a notebook)."""
    root, files = site()
    port = _free_port(port)
    handler = functools.partial(_Handler, directory=str(root), files=files)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"zcmir app: {url}  (Ctrl+C to stop)" if block else f"zcmir app: {url}")
    if open_browser:
        threading.Timer(0.3, webbrowser.open, (url,)).start()
    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return url
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return url


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zcmir", description="Serve the zcmir web app locally and open it in the browser.")
    ap.add_argument("--port", type=int, default=4180, help="port on 127.0.0.1 (default 4180; the next free one if taken)")
    ap.add_argument("--no-browser", action="store_true", help="only print the URL")
    a = ap.parse_args(argv)
    serve(a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    sys.exit(main())
