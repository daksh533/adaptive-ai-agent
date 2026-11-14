"""
web_ui/app.py

Phase 12 — Interactive Visual Web UI Dashboard

Flask backend with Server-Sent Events (SSE) for real-time streaming
of agent execution logs to the browser.

Usage:
    python web_ui/app.py
    Then open http://localhost:5000
"""

from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import time
import uuid

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["SECRET_KEY"] = "adaptive-agent-ui-2026"

# ---------------------------------------------------------------------------
# In-memory run store: run_id -> {queue, state, done}
# ---------------------------------------------------------------------------
_runs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Stdout interceptor — captures agent print() calls and routes to SSE queue
# ---------------------------------------------------------------------------

class _StreamCapture(io.TextIOBase):
    """Wraps sys.stdout so every print() line also goes into the SSE queue."""

    def __init__(self, run_id: str, orig_stdout):
        self._run_id = run_id
        self._orig = orig_stdout
        self._buf = ""

    def write(self, text: str) -> int:
        self._orig.write(text)
        self._orig.flush()
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                _emit(self._run_id, "log", {"text": line})
        return len(text)

    def flush(self):
        self._orig.flush()

    @property
    def encoding(self):
        return getattr(self._orig, "encoding", "utf-8")


def _emit(run_id: str, event: str, data: dict) -> None:
    run = _runs.get(run_id)
    if run:
        run["queue"].put({"event": event, "data": data})


# ---------------------------------------------------------------------------
# Background worker — runs the agent in a thread
# ---------------------------------------------------------------------------

def _run_agent_thread(run_id: str, goal: str, mode: str) -> None:
    old_stdout = sys.stdout
    sys.stdout = _StreamCapture(run_id, old_stdout)

    try:
        from agent.loop import run_agent
        _emit(run_id, "status", {"phase": "running", "text": "Agent starting…"})
        final_state = run_agent(goal, mode=mode)

        # Build structured step results for the UI
        results = []
        for entry in final_state.get("results_so_far", []):
            results.append({
                "step": entry.get("step", ""),
                "result": entry.get("result", ""),
                "verification_status": entry.get("verification_status", ""),
                "verification_reason": entry.get("verification_reason", ""),
                "error_diagnosis": entry.get("error_diagnosis", ""),
                "correction_attempts": entry.get("correction_attempts", 0),
            })

        _emit(run_id, "complete", {
            "agent_status": final_state.get("agent_status", "unknown"),
            "risk_level": final_state.get("risk_level", ""),
            "verification_mode": final_state.get("verification_mode", mode),
            "plan": final_state.get("plan", []),
            "results": results,
        })

    except Exception as exc:
        _emit(run_id, "error", {"text": str(exc)})
    finally:
        sys.stdout = old_stdout
        _runs[run_id]["done"] = True
        _runs[run_id]["queue"].put(None)   # sentinel


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def start_run():
    body = request.get_json(force=True)
    goal = (body.get("goal") or "").strip()
    mode = body.get("mode", "adaptive")
    if mode not in ("baseline", "fixed", "adaptive"):
        mode = "adaptive"
    if not goal:
        return jsonify({"error": "goal is required"}), 400

    run_id = str(uuid.uuid4())
    _runs[run_id] = {"queue": queue.Queue(), "done": False}

    thread = threading.Thread(target=_run_agent_thread, args=(run_id, goal, mode), daemon=True)
    thread.start()

    return jsonify({"run_id": run_id})


@app.route("/api/stream/<run_id>")
def stream(run_id: str):
    if run_id not in _runs:
        return jsonify({"error": "run not found"}), 404

    def generate():
        q = _runs[run_id]["queue"]
        while True:
            item = q.get()
            if item is None:
                yield "event: done\ndata: {}\n\n"
                break
            payload = json.dumps(item["data"], ensure_ascii=False)
            yield f"event: {item['event']}\ndata: {payload}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    print("=" * 55)
    print("  Adaptive AI Agent — Web UI")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
