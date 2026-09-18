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

    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path, limit=100):
    if not path.exists():
        return []

    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))[:limit]


def make_response(data, status=200):
    return json.dumps(
        data,
        ensure_ascii=False
    ).encode("utf-8")


class handler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = make_response(data, status)

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

            # GET /api/metrics
            if path == "/api/metrics":
                self.send_json(
                    load_json(
                        ROOT / "results" / "metrics.json"
                    )
                )
                return

            # GET /api/golden
            if path == "/api/golden":
                self.send_json(
                    load_csv(
                        ROOT / "data" / "golden" / "golden_set.csv",
                        100
                    )
                )
                return

            # GET /api/baselines
            if path == "/api/baselines":
                self.send_json({
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
                    ),
                })
                return

            # GET /api/failures
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

            self.send_json(
                {"error": "Not found"},
                404
            )

        except Exception as exc:
            self.send_json(
                {"error": str(exc)},
                500
            )

    def do_POST(self):

        try:
            path = self.path.split("?", 1)[0]

            # POST /api/classify
            if path == "/api/classify":

                content_length = int(
                    self.headers.get("Content-Length", 0)
                )

                body = self.rfile.read(content_length)

                data = json.loads(
                    body.decode("utf-8")
                )

                message = str(
                    data.get("message", "")
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

            self.send_json(
                {"error": "Not found"},
                404
            )

        except Exception as exc:
            self.send_json(
                {"error": str(exc)},
                500
            )