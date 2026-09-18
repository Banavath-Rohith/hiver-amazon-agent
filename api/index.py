import json
import csv
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler

ROOT = Path(__file__).resolve().parent.parent

# Make src importable
sys.path.insert(0, str(ROOT / "src"))

from agent import run as agent_run


def load_json(path):
    if not path.exists():
        return {}

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_csv(path, limit=100):
    if not path.exists():
        return []

    with path.open(
        encoding="utf-8",
        newline=""
    ) as f:
        return list(csv.DictReader(f))[:limit]


def make_response(data):
    return json.dumps(
        data,
        ensure_ascii=False
    ).encode("utf-8")


class handler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = make_response(data)

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(body)

    def send_html(self, html):
        body = html.encode("utf-8")

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(body)

    def do_OPTIONS(self):

        self.send_response(204)

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.end_headers()

    def do_GET(self):

        try:
            path = self.path.split("?", 1)[0]

            # ---------------------------------
            # FRONTEND
            # ---------------------------------

            if path == "/":
                index_file = ROOT / "index.html"

                if not index_file.exists():
                    self.send_json(
                        {
                            "error": "index.html not found"
                        },
                        500
                    )
                    return

                html = index_file.read_text(
                    encoding="utf-8"
                )

                self.send_html(html)
                return

            # ---------------------------------
            # FRONTEND CSS
            # ---------------------------------

            if path == "/styles.css":
                css_file = ROOT / "styles.css"

                if not css_file.exists():
                    self.send_json(
                        {
                            "error": "styles.css not found"
                        },
                        404
                    )
                    return

                body = css_file.read_bytes()

                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    "text/css; charset=utf-8"
                )

                self.send_header(
                    "Content-Length",
                    str(len(body))
                )

                self.end_headers()

                self.wfile.write(body)
                return

            # ---------------------------------
            # FRONTEND JAVASCRIPT
            # ---------------------------------

            if path == "/app.js":
                js_file = ROOT / "app.js"

                if not js_file.exists():
                    self.send_json(
                        {
                            "error": "app.js not found"
                        },
                        404
                    )
                    return

                body = js_file.read_bytes()

                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    "application/javascript; charset=utf-8"
                )

                self.send_header(
                    "Content-Length",
                    str(len(body))
                )

                self.end_headers()

                self.wfile.write(body)
                return

            # ---------------------------------
            # API: METRICS
            # ---------------------------------

            if path == "/api/metrics":
                self.send_json(
                    load_json(
                        ROOT
                        / "results"
                        / "metrics.json"
                    )
                )
                return

            # ---------------------------------
            # API: GOLDEN SET
            # ---------------------------------

            if path == "/api/golden":
                self.send_json(
                    load_csv(
                        ROOT
                        / "data"
                        / "golden"
                        / "golden_set.csv",
                        100
                    )
                )
                return

            # ---------------------------------
            # API: BASELINES
            # ---------------------------------

            if path == "/api/baselines":
                self.send_json(
                    {
                        "trivial": load_json(
                            ROOT
                            / "results"
                            / "trivial_baseline_metrics.json"
                        ),

                        "simple": load_json(
                            ROOT
                            / "results"
                            / "simple_baseline_metrics.json"
                        ),

                        "agent": load_json(
                            ROOT
                            / "results"
                            / "metrics.json"
                        )
                    }
                )
                return

            # ---------------------------------
            # API: FAILURES
            # ---------------------------------

            if path == "/api/failures":
                self.send_json(
                    load_csv(
                        ROOT
                        / "results"
                        / "failure_analysis.csv",
                        10
                    )
                )
                return

            # ---------------------------------
            # NOT FOUND
            # ---------------------------------

            self.send_json(
                {
                    "error": "Not found"
                },
                404
            )

        except Exception as exc:

            self.send_json(
                {
                    "error": str(exc)
                },
                500
            )

    def do_POST(self):

        try:
            path = self.path.split("?", 1)[0]

            # ---------------------------------
            # API: CLASSIFY
            # ---------------------------------

            if path == "/api/classify":

                content_length = int(
                    self.headers.get(
                        "Content-Length",
                        0
                    )
                )

                body = self.rfile.read(
                    content_length
                )

                data = json.loads(
                    body.decode("utf-8")
                )

                message = str(
                    data.get(
                        "message",
                        ""
                    )
                ).strip()

                if not message:

                    self.send_json(
                        {
                            "error":
                            "message field is required"
                        },
                        400
                    )

                    return

                result = agent_run(message)

                self.send_json(result)
                return

            # ---------------------------------
            # NOT FOUND
            # ---------------------------------

            self.send_json(
                {
                    "error": "Not found"
                },
                404
            )

        except Exception as exc:

            self.send_json(
                {
                    "error": str(exc)
                },
                500
            )