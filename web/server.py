"""Lightweight HTTP server for the AmazonHelp agent web preview.

Serves the static web UI and exposes two JSON API endpoints:

  POST /api/classify    { "message": "..." }
    -> { intent, confidence, triggered, action, reason, reply, evidence }

  GET  /api/metrics
    -> contents of results/metrics.json

  GET  /api/golden
    -> first 100 rows of data/golden/golden_set.csv as JSON

  GET  /api/baselines
    -> trivial + simple baseline metrics as JSON

Usage (from project root):
    python web/server.py
    # Open http://127.0.0.1:8000
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import run as agent_run

PORT   = 8000
WEB    = Path(__file__).parent
STATIC = {
    "/":            (WEB / "index.html",  "text/html; charset=utf-8"),
    "/styles.css":  (WEB / "styles.css",  "text/css; charset=utf-8"),
    "/app.js":      (WEB / "app.js",      "application/javascript; charset=utf-8"),
}


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_as_json(path: Path, limit: int = 100) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))[:limit]


class AgentHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # reduce noise
        pass

    def _send_json(self, data: dict | list, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        # Static files
        if path in STATIC:
            file_path, ct = STATIC[path]
            self._send_file(file_path, ct)
            return

        # API: metrics
        if path == "/api/metrics":
            self._send_json(_load_json(ROOT / "results" / "metrics.json"))
            return

        # API: golden set
        if path == "/api/golden":
            rows = _load_csv_as_json(ROOT / "data" / "golden" / "golden_set.csv")
            self._send_json(rows)
            return

        # API: baselines comparison
        if path == "/api/baselines":
            trivial = _load_json(ROOT / "results" / "trivial_baseline_metrics.json")
            simple  = _load_json(ROOT / "results" / "simple_baseline_metrics.json")
            agent   = _load_json(ROOT / "results" / "metrics.json")
            self._send_json({
                "trivial": trivial,
                "simple":  simple,
                "agent":   agent,
            })
            return

        # API: failure analysis
        if path == "/api/failures":
            rows = _load_csv_as_json(ROOT / "results" / "failure_analysis.csv", limit=10)
            self._send_json(rows)
            return

        self.send_error(404, "Not found")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/classify":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length).decode("utf-8")
            try:
                data    = json.loads(body)
                message = data.get("message", "").strip()
                if not message:
                    self._send_json({"error": "message field is required"}, 400)
                    return
                result = agent_run(message)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
            return

        self.send_error(404, "Not found")


def main() -> None:
    server = HTTPServer(("127.0.0.1", PORT), AgentHandler)
    print(f"AmazonHelp Agent Preview")
    print(f"  Open: http://127.0.0.1:{PORT}")
    print(f"  Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
