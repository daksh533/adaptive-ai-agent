"""
web_ui/app.py

Phase 12 — Interactive Visual Web UI Dashboard

Flask backend with Server-Sent Events (SSE) for real-time streaming
of agent execution logs to the browser.

New in this revision:
  • /api/upload  — ingest uploaded PDF / TXT files into ChromaDB
  • /api/input/<run_id> — receive user-provided input() values
  • input_required SSE event — tells the browser to show an input modal

Usage:
    python web_ui/app.py
    Then open http://localhost:5000
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
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
# In-memory run store: run_id -> {queue, state, done, input_event, input_value}
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
# Input-interception patch for python_tools.run_python
# ---------------------------------------------------------------------------

def _patch_run_python(run_id: str):
    """
    Monkey-patch tools.python_tools.run_python so that any code containing
    input() pauses execution, asks the browser for the value via SSE, waits
    for the user to POST it, then re-runs the code with the value substituted.
    """
    import tools.python_tools as pt

    if not hasattr(pt, "_orig_run_python"):
        pt._orig_run_python = pt.run_python
    _base_run_python = pt._orig_run_python

    def _patched_run_python(code: str, timeout: int = pt.DEFAULT_TIMEOUT) -> str:
        # Detect all input() calls with optional prompt strings
        input_pattern = re.compile(
            r"""input\s*\(\s*(?:["'](.*?)["']|f["'](.*?)["']|)?\s*\)""",
            re.DOTALL,
        )
        matches = list(input_pattern.finditer(code))

        if not matches:
            return _base_run_python(code, timeout)

        # Collect prompts & ask user for each
        collected = {}
        for idx, m in enumerate(matches):
            prompt_text = m.group(1) or m.group(2) or f"Input required (#{idx + 1})"
            _emit(run_id, "input_required", {"prompt": prompt_text, "idx": idx})

            # Wait for user response (up to 5 minutes)
            run = _runs.get(run_id, {})
            evt = run.get("input_event")
            if evt:
                evt.clear()
                evt.wait(timeout=300)
                val = run.get("input_value", "")
            else:
                val = ""
            collected[idx] = val

        # Replace each input(...) call with its collected value (as a string literal)
        # We replace in reverse order so positions stay valid
        patched_code = code
        for idx, m in reversed(list(enumerate(matches))):
            replacement = repr(str(collected.get(idx, "")))
            patched_code = patched_code[:m.start()] + replacement + patched_code[m.end():]

        return _base_run_python(patched_code, timeout)

    pt.run_python = _patched_run_python


# ---------------------------------------------------------------------------
# Background worker — runs the agent in a thread
# ---------------------------------------------------------------------------

def _run_agent_thread(run_id: str, goal: str, mode: str) -> None:
    old_stdout = sys.stdout
    sys.stdout = _StreamCapture(run_id, old_stdout)

    # Patch run_python for this run so input() is intercepted
    _patch_run_python(run_id)

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
    _runs[run_id] = {
        "queue": queue.Queue(),
        "done": False,
        "input_event": threading.Event(),
        "input_value": "",
    }

    thread = threading.Thread(target=_run_agent_thread, args=(run_id, goal, mode), daemon=True)
    thread.start()

    return jsonify({"run_id": run_id})


@app.route("/api/input/<run_id>", methods=["POST"])
def receive_input(run_id: str):
    """Browser POSTs the user's typed value here when input() is waiting."""
    run = _runs.get(run_id)
    if not run:
        return jsonify({"error": "run not found"}), 404
    body = request.get_json(force=True)
    run["input_value"] = body.get("value", "")
    run["input_event"].set()
    return jsonify({"status": "ok"})


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


# ---------------------------------------------------------------------------
# RAG Upload Route
# ---------------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Accept a .pdf or .txt file, extract its text, chunk it, embed it,
    and upsert into the ChromaDB 'agent_knowledge' collection.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename.lower()
    raw_bytes = file.read()

    # ---- Text extraction ----
    text = ""
    if filename.endswith(".txt"):
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")

    elif filename.endswith(".pdf"):
        try:
            from pdfminer.high_level import extract_text as pdf_extract_text
            import io as _io
            text = pdf_extract_text(_io.BytesIO(raw_bytes))
        except ImportError:
            # Fallback: try PyPDF2
            try:
                import PyPDF2, _io as _io2
                reader = PyPDF2.PdfReader(_io2.BytesIO(raw_bytes))
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                return jsonify({
                    "error": "PDF support requires pdfminer.six. Run: pip install pdfminer.six"
                }), 500
    else:
        return jsonify({"error": "Only .pdf and .txt files are supported"}), 400

    text = text.strip()
    if not text:
        return jsonify({"error": "Could not extract any text from the file"}), 400

    # ---- Chunk & embed ----
    try:
        from rag.ingest import chunk_text, CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME
        from sentence_transformers import SentenceTransformer
        import chromadb
        from pathlib import Path

        chunks = chunk_text(text)
        if not chunks:
            return jsonify({"error": "File produced no chunks after processing"}), 400

        chroma_dir = Path(CHROMA_DIR)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        model = SentenceTransformer(EMBED_MODEL_NAME)
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        safe_name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", file.filename)
        ids = [f"upload_{safe_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": file.filename, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]

        collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

        return jsonify({
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "message": f"✅ '{file.filename}' ingested — {len(chunks)} chunks added to knowledge base.",
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Adaptive AI Agent — Web UI")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
