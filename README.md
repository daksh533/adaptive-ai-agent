# Adaptive Self-Correcting AI Agent

A multi-step LLM agent built with **LangGraph** that plans, executes, verifies,
diagnoses its own failures, corrects them, and replans when correction is not
enough.

The agent is deliberately small and readable: every phase is a separate module,
and the control flow is a single compiled state graph.

---

## What it does

```
Goal
  │
  ▼
Planner ──► Executor ──► Verifier ──passed──► next step / done
                            │
                            └─failed──► Diagnoser ──► Corrector ──► Verifier
                                             │
                                             └─ attempts exhausted ──► Replanner
                                                                          │
                                                                          ▼
                                                                     new plan / stop
```

* **Planner** breaks the goal into a small, ordered list of subtasks.* **Executor** solves one subtask at a time using tools.
* **Verifier** checks every result with three independent methods.
* **Diagnoser** classifies failures into a six-class taxonomy.
* **Corrector** applies an error-specific recovery strategy.
* **Replanner** rewrites the remaining plan when targeted fixes fail.

---

## Phase map

| Phase | Module | Responsibility |
|---|---|---|
| 1–2 | `llm_client.py`, `tools/` | Groq LLM access + tool calling |
| 3 | `agent/planner.py` | Goal → ordered subtask plan |
| 4 | `agent/executor.py`, `agent/graph.py` | LangGraph execution loop |
| 5 | `rag/`, `tools/rag_lookup.py` | ChromaDB knowledge-base retrieval |
| 6 | `verification/verifier.py` | Numeric, evidence, and logic checks |
| 7 | `verification/diagnoser.py` | 6-class error diagnosis |
| 8 | `correction/corrector.py` | Targeted self-correction |
| 9 | `planner/replanner.py` | Dynamic plan invalidation |

### Available tools

| Tool | Description |
|---|---|
| `calculate` | Safe AST-based arithmetic evaluator (`+ - * / // % **`, `sqrt`, `log`, trig, `factorial`, `pi`, `e`) |
| `web_search` | DuckDuckGo search via `ddgs` |
| `run_python` | Short Python snippets in an isolated subprocess with a 10 s timeout |
| `rag_lookup` | Top-k retrieval from the local ChromaDB knowledge base |

---

## Project layout

```
adaptive agent ai/
├── main.py                  # interactive CLI + one-shot mode
├── llm_client.py            # shared LLM client + retry/JSON helpers
├── .env.example             # template for GROQ_API_KEY
├── agent/
│   ├── graph.py             # LangGraph nodes, routing, retry bounds
│   ├── graph_state.py       # TypedDict channel schema
│   ├── planner.py           # goal → plan
│   ├── executor.py          # one subtask + tools
│   ├── loop.py              # public run_agent() entry point
│   └── state.py             # legacy dataclass (early-phase test only)
├── verification/
│   ├── verifier.py          # 3 verification methods
│   └── diagnoser.py         # 6-class failure taxonomy
├── correction/
│   └── corrector.py         # one strategy per error class
├── planner/
│   └── replanner.py         # revised remaining plan
├── tools/
│   ├── registry.py          # tool schemas + dispatch
│   ├── calculator.py
│   ├── python_tools.py
│   ├── web_search.py
│   └── rag_lookup.py
├── rag/
│   ├── ingest.py            # chunk + embed + upsert
│   ├── retriever.py         # semantic search
│   └── knowledge_base/      # .txt / .md source documents
├── test_offline.py          # fast tests, no API calls
├── test_*.py                # phase checkpoint scripts (API-backed)
└── demo_all_phases.py       # showcase script
```

---

## Quick start (Windows PowerShell)

```powershell
cd "C:\Users\chauh\Desktop\adaptive agent ai"

# 1. Install dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Add your Groq key
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# then edit .env and replace your_key_here

# 3. Build the vector store (first run downloads all-MiniLM-L6-v2)
python -m rag.ingest --reset

# 4. Run the agent
python main.py

# ...or run a single goal non-interactively (handy for demos):
python main.py "Calculate 25% of 800, then add 50."
```

`main.py` also works without activating the venv — it adds
`venv/Lib/site-packages` to `sys.path` automatically.

### Example goals

```
Calculate 25% of 800, then calculate 15% of that result, then add 50.
Look up the project overview in the knowledge base and cite the authors.
Use Python to compute the factorial of 6, then multiply it by 3.
```

---

## Configuration

| Variable | Required | Default |
|---|---|---|
| `GROQ_API_KEY` | yes | — |
| `GROQ_MODEL` | no | `openai/gpt-oss-20b` |
| `GROQ_BASE_URL` | no | `https://api.groq.com/openai/v1` |
| `LLM_MAX_RETRY_WAIT` | no | `120` (seconds; longer rate-limit waits fail fast) |

The `.env` file is loaded automatically by `llm_client.py`.

---

## Tests

Fast, offline (no API key required, 40 tests):

```powershell
.\venv\Scripts\python.exe test_offline.py -v
```

The offline suite covers the calculator sandbox, numeric re-checking (including
LaTeX answers), all six diagnoser classes, graph state transitions, retry
bounds, routing, and a full graph run with mocked LLM/tool collaborators.

If the free-tier quota is exhausted during a demo, `test_offline.py` is a good
fallback: it exercises the complete control flow without making API calls.

API-backed phase checkpoints:

```powershell
.\venv\Scripts\python.exe test_verification.py
.\venv\Scripts\python.exe test_phases_7_8_9.py
.\venv\Scripts\python.exe test_all_phases.py
```

Full showcase (diagnoser, corrector, replanner, end-to-end run):

```powershell
.\venv\Scripts\python.exe demo_all_phases.py
```

---

## Design notes

### Bounded retries

* Each step gets `MAX_CORRECTION_ATTEMPTS = 2` targeted correction attempts.
* After that, the plan is rewritten up to `MAX_REPLANS = 2` times.
* If the plan still fails, the agent stops with `agent_status = "failed"`
  instead of looping forever.

### Correct step targeting

`current_step` advances **only when a step passes verification**. A failed
step keeps its index, so the diagnoser and corrector always see the subtask
that actually failed, and the corrected result is re-verified before the
agent moves on.

After a replan, previously failed attempts stay in the execution history for
auditing but are excluded from the context given to the executor and
verifier, so a stale wrong answer cannot poison the revised plan.

### Verification is independent

1. **Numeric re-check** parses arithmetic from the answer and re-computes it
   with the safe calculator — no LLM involved.
2. **Evidence grounding** asks the LLM whether claims are supported by prior
   results / context.
3. **Logical consistency** asks the LLM whether the result actually answers
   the subtask.

Any failure fails the step. If an LLM-backed check is unavailable, it is
skipped (fail-open) and logged, so the agent does not stall on a transient
API error.

### RAG

Documents in `rag/knowledge_base/` are chunked (400 chars, 80-char overlap),
embedded with `all-MiniLM-L6-v2`, and stored persistently in ChromaDB. The
`rag_lookup` tool returns chunks with source and chunk index for citation.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GROQ_API_KEY is not set` | Add `GROQ_API_KEY=...` to `.env` in the project root. |
| `429` / `tokens per day` | The Groq free tier has a daily token quota. Wait for it to reset, upgrade the Groq plan, or set `GROQ_MODEL` to another supported model in `.env`. The CLI fails fast with a clear message instead of hanging. |
| `ChromaDB store not found` | Run `python -m rag.ingest --reset`. |
| First run is slow / downloads a model | `sentence-transformers` downloads `all-MiniLM-L6-v2` once; later runs use the local cache. |
| Python tool output is truncated | The tool caps output at 4,000 characters by design. |

---

## Demo flow

A short, reliable sequence for a live demo:

```powershell
# 1. Prove the logic without spending API tokens (40 tests, ~0.1 s)
.\venv\Scripts\python.exe test_offline.py

# 2. Show one real end-to-end goal with verification
.\venv\Scripts\python.exe -u main.py "Calculate 25% of 800, then add 50."

# 3. Show the full phase showcase (diagnoser, corrector, replanner)
.\venv\Scripts\python.exe demo_all_phases.py
```

Expected: the agent prints the plan, each tool call, a three-part verification
verdict per step, and a final verified answer (`80` for the example above).

---

## Known limitations

* `run_python` is an isolated subprocess, **not a hardened security sandbox**.
  It runs with the current user's permissions — do not feed it untrusted code.
* Numeric verification is heuristic: it catches common inline arithmetic
  (`25% of 800 = 200`, `15 + 30 = 45`, LaTeX `\boxed{200}` / `\times`) but not
  every possible phrasing. It ignores intermediate expressions such as
  `25% of 800 = 0.25 * 800 = 200`.
* LLM-backed verification checks fail open: an API outage can let an
  unverified result through (it is logged, not silent).
* The agent has no long-term memory between runs; state lives for one
  `run_agent()` call.
* ChromaDB is local and single-process; it is not a production vector store.
