"""Local dashboard for the Prediction Markets Arbitrage Finder.

Usage:
    python3 dashboard.py            # serves http://localhost:8400
    python3 dashboard.py --port N

Serves dashboard.html, exposes the latest results.json, and can kick off
a full scan (matcher.main) in a background thread while streaming its
console output to the browser. Stdlib only — no extra dependencies.
"""
import argparse
import io
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import matcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_state = {
    "running": False,
    "log": [],
    "error": None,
}
_state_lock = threading.Lock()


class _LogCapture(io.TextIOBase):
    """Mirrors matcher's print() output into the shared log buffer."""

    def __init__(self, real):
        self._real = real
        self._buf = ""

    def write(self, s):
        self._real.write(s)
        self._real.flush()
        self._buf += s
        # Progress lines use \r; treat both as line breaks.
        while True:
            cut = min(
                (i for i in (self._buf.find("\n"), self._buf.find("\r")) if i >= 0),
                default=-1,
            )
            if cut < 0:
                break
            line, self._buf = self._buf[:cut], self._buf[cut + 1:]
            if line.strip():
                with _state_lock:
                    _state["log"].append(line.rstrip())
        return len(s)

    def flush(self):
        self._real.flush()


def _run_scan():
    old_stdout = sys.stdout
    sys.stdout = _LogCapture(old_stdout)
    try:
        matcher.main()
    except Exception as exc:  # surface the failure in the UI
        with _state_lock:
            _state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sys.stdout = old_stdout
        with _state_lock:
            _state["running"] = False


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(BASE_DIR, "dashboard.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/results":
            path = os.path.join(BASE_DIR, "results.json")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self._send(200, f.read())
            else:
                self._send(404, {"error": "no results yet"})
        elif self.path == "/api/status":
            with _state_lock:
                self._send(200, {
                    "running": _state["running"],
                    "log": _state["log"][-400:],
                    "error": _state["error"],
                    "has_results": os.path.exists(os.path.join(BASE_DIR, "results.json")),
                    "llm_enabled": bool(os.environ.get("ANTHROPIC_API_KEY")),
                })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/run":
            with _state_lock:
                if _state["running"]:
                    self._send(409, {"error": "scan already running"})
                    return
                _state["running"] = True
                _state["log"] = []
                _state["error"] = None
            threading.Thread(target=_run_scan, daemon=True).start()
            self._send(202, {"started": True})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):  # silence per-request noise
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Dashboard running at {url}  (Ctrl-C to stop)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Note: ANTHROPIC_API_KEY is not set — scans will skip LLM "
              "verification and use TF-IDF matches only.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
