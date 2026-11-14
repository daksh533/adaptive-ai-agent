# 🤖 Adaptive Self-Correcting AI Agent

> **MCA Capstone Research Project** — IILM University, Greater Noida (Academic Year 2026)  
> **Authors:** Daksh · Tarkeshwar Kumar Yadav · Abhinav Sharma  
> **Specialization:** Artificial Intelligence & Machine Learning

---

## 📌 What Is This Project?

This project implements an **autonomous, adaptive, self-correcting AI agent** built on [LangGraph](https://github.com/langchain-ai/langgraph). The agent accepts complex multi-step natural language goals, plans and executes them using real tools (calculator, web search, Python execution, knowledge base), and independently detects, diagnoses, and corrects its own mistakes — all without any human intervention.

The core research novelty is **Adaptive Risk-Based Verification (Phase 10)**: instead of blindly running expensive 3-way LLM verification on every single step, the agent classifies each step into a **LOW / MEDIUM / HIGH** risk tier and applies just the right amount of verification — saving tokens, reducing latency, and maintaining accuracy.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Autonomous Planning** | Breaks any natural language goal into ordered subtasks |
| **RAG-First Retrieval** | Checks local ChromaDB knowledge base before hitting the web |
| **Tool Execution** | Calculator, Python runner, DuckDuckGo web search, RAG lookup |
| **3-Method Verification** | Numeric re-check (AST), Evidence Grounding, Logical Consistency |
| **Adaptive Risk Tiers** | LOW / MEDIUM / HIGH verification depth per step |
| **Structured Error Diagnosis** | 6-class error taxonomy for targeted recovery |
| **Self-Correction** | Targeted fix strategies per error class |
| **Dynamic Replanning** | Rewrites remaining plan on repeated failures |
| **Web UI Dashboard** | Real-time streaming UI at http://localhost:5000 |
| **Benchmark Suite** | Comparative evaluation: Baseline vs Fixed vs Adaptive |

---

## 🗂️ Project Structure

```
adaptive agent ai/
│
├── agent/                      # Core agent modules
│   ├── graph.py                # LangGraph state machine (all nodes & routing)
│   ├── graph_state.py          # Shared TypedDict state schema
│   ├── planner.py              # Goal → ordered subtask plan
│   ├── executor.py             # Subtask execution with tool dispatch
│   └── loop.py                 # run_agent(goal, mode) entry point
│
├── verification/               # Verification & risk system
│   ├── verifier.py             # 3-way verifier + verify_step_adaptive()
│   ├── risk_classifier.py      # Keyword + LLM risk tier classifier
│   └── diagnoser.py            # 6-class error taxonomy diagnoser
│
├── correction/
│   └── corrector.py            # Targeted self-correction strategies
│
├── planner/
│   └── replanner.py            # Dynamic plan revision on failure
│
├── tools/                      # Agent tools
│   ├── calculator.py           # Safe AST-based math evaluator
│   ├── web_search.py           # DuckDuckGo multi-backend search
│   ├── run_python.py           # Sandboxed Python subprocess runner
│   ├── rag_lookup.py           # RAG lookup dispatcher
│   └── knowledge_lookup.py     # ChromaDB + web fallback retrieval
│
├── rag/                        # Knowledge base (RAG)
│   ├── ingest.py               # ChromaDB ingestion pipeline
│   └── knowledge_base/         # Markdown knowledge documents
│       ├── roadmap.md
│       ├── architecture_spec.md
│       ├── error_taxonomy.md
│       └── ...
│
├── benchmark/                  # Phase 11 benchmarking
│   ├── benchmark_tasks.py      # Standardized task suite (5 tasks)
│   ├── metrics.py              # RunResult & BenchmarkMetrics data models
│   └── evaluator.py            # Multi-config comparative evaluator
│
├── web_ui/                     # Phase 12 Web UI
│   ├── app.py                  # Flask server with SSE streaming
│   └── templates/index.html    # Clean dark-theme dashboard
│
├── llm_client.py               # Central Groq LLM client (throttling + retry)
├── main.py                     # CLI entry point
├── run_benchmark.py            # Benchmark CLI runner
├── requirements.txt
└── .env                        # API keys (never commit this)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+ (tested on 3.14)
- A [Groq](https://console.groq.com) API key (free tier available)

### 1. Clone or open the project

```bash
cd "adaptive agent ai"
```

### 2. Create virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
pip install flask flask-cors   # For Web UI
```

### 4. Configure API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

> Get your free API key at: https://console.groq.com

### 5. Ingest knowledge base into ChromaDB

```powershell
.\venv\Scripts\python -m rag.ingest
```

---

## 🚀 How to Run

### Option A — Web UI (Recommended)

```powershell
.\venv\Scripts\python web_ui\app.py
```

Then open **http://localhost:5000** in your browser.

The UI provides:
- 🎯 **Goal input** with preset demo prompts
- ⚙️ **Mode selector** (Adaptive / Fixed / Baseline)
- 📋 **Live execution plan** with step-by-step status badges
- 🖥️ **Real-time terminal log** streamed to the browser (color-coded by module)
- ✅ **Verification results** with LOW/MEDIUM/HIGH risk tier badges

### Option B — Command Line

```powershell
.\venv\Scripts\python main.py
```

Then type any goal at the prompt, e.g.:
```
Goal: Calculate 25% of 500, then add 50
```

### Option C — Python API

```python
from agent.loop import run_agent

state = run_agent("Research Tokyo's population and calculate 20% of it.")
print(state["agent_status"])   # "completed"
print(state["results_so_far"]) # list of all step results
```

---

## 🧠 System Architecture

```
            User Goal
                │
                ▼
         ┌─────────────┐
         │   Planner   │  ← Breaks goal into ordered subtasks
         └──────┬──────┘
                │
         ┌──────▼──────┐
     ┌──▶│  Executor   │  ← Runs tools: calculate / web_search / rag_lookup / run_python
     │   └──────┬──────┘
     │          │
     │   ┌──────▼──────────────────────────────────────────┐
     │   │  Verifier (Phase 10: Adaptive Risk-Based)        │
     │   │                                                   │
     │   │  RiskClassifier ──▶ LOW   ──▶ numeric_recheck    │
     │   │                 ──▶ MEDIUM ──▶ + evidence_check  │
     │   │                 ──▶ HIGH   ──▶ + logic_check     │
     │   └──────┬───────────────────┬──────────────────────┘
     │          │ PASSED            │ FAILED
     │          ▼                   ▼
     │   Next Step / END     ┌─────────────┐
     │                       │  Diagnoser  │  ← Classifies failure (6 error types)
     │                       └──────┬──────┘
     │                              │
     │                       ┌──────▼──────┐
     └───────────────────────│  Corrector  │  ← Targeted fix strategy
                   retry     └──────┬──────┘
                                    │ too many failures
                             ┌──────▼──────┐
                             │  Replanner  │  ← Rewrites remaining plan
                             └─────────────┘
```

---

## 🔬 Phase-by-Phase Implementation

| Phase | Component | Status | Key Files |
| :---: | :--- | :---: | :--- |
| 1 | Foundation & Groq API | ✅ Complete | `llm_client.py` |
| 2 | Deterministic Tool Integration | ✅ Complete | `tools/` |
| 3 | Basic Agent Control Loop | ✅ Complete | `agent/planner.py`, `agent/executor.py` |
| 4 | LangGraph State Machine | ✅ Complete | `agent/graph.py`, `agent/graph_state.py` |
| 5 | RAG Knowledge Base Grounding | ✅ Complete | `rag/`, `tools/knowledge_lookup.py` |
| 6 | Comprehensive Verification | ✅ Complete | `verification/verifier.py` |
| 7 | Structured Error Diagnosis | ✅ Complete | `verification/diagnoser.py` |
| 8 | Targeted Self-Correction | ✅ Complete | `correction/corrector.py` |
| 9 | Dynamic Replanning | ✅ Complete | `planner/replanner.py` |
| 10 | **Adaptive Risk-Based Verification** | ✅ Complete | `verification/risk_classifier.py` |
| 11 | Empirical Benchmarking | ✅ Complete | `benchmark/`, `run_benchmark.py` |
| 12 | Interactive Web UI Dashboard | ✅ Complete | `web_ui/` |
| 13 | Research Paper & Thesis | 🔄 In Progress | — |

---

## ⚡ Adaptive Risk Tiers (Phase 10)

The core research novelty. Each step is classified before verification runs:

| Tier | Trigger Signals | Checks Run | LLM Calls |
| :---: | :--- | :--- | :---: |
| 🟢 **LOW** | Text formatting, summarisation, no math or external facts | `numeric_recheck` only | **0** |
| 🟡 **MEDIUM** | Factual claims, web / RAG citations, knowledge retrieval | `numeric_recheck` + `evidence_grounding` | **1** |
| 🔴 **HIGH** | Arithmetic, financial calculations, multi-step numeric reasoning | Full 3-way verification | **2** |

**Classification is done first by keyword rules (zero LLM calls), and falls back to an LLM call only if the keywords are ambiguous.** This means most steps are classified instantly at zero cost.

---

## 📊 Phase 11 Benchmarks

### Benchmark Task Suite

| Task ID | Category | Goal |
| :--- | :--- | :--- |
| `MATH_01` | Arithmetic / High Risk | Multi-step chained calculation (15% → +50 → ×2) |
| `RAG_01` | Knowledge Base / Medium Risk | Retrieve error taxonomy classes from ChromaDB |
| `FORMAT_01` | Formatting / Low Risk | Alphabetical name list — no math, no lookups |
| `RESEARCH_01` | Web Research + Math / Hybrid | Population research + percentage calculation |
| `RECOVERY_01` | Error Recovery | Multi-step math with compound dependencies |

### Configuration Comparison

| Configuration | Description | Verification Calls | Accuracy | Speed |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | No verification, no correction | 0 per step | Lower (errors undetected) | Fastest |
| **Fixed** | Full 3-way check every step | 2 per step | High | Slowest |
| **Adaptive** ⭐ | Risk-tiered dynamic verification | 0–2 per step | High | Faster than Fixed |

> **Key Finding:** The Adaptive configuration achieves the same accuracy as the Fixed configuration with significantly fewer LLM verification calls, lower API cost, and reduced latency — especially for text-heavy tasks classified as LOW risk.

### How to Run the Benchmark

```powershell
# Run all 5 tasks across all 3 configurations
.\venv\Scripts\python run_benchmark.py

# Run only the math task, compare adaptive vs baseline
.\venv\Scripts\python run_benchmark.py --configs baseline adaptive --task MATH_01

# Custom throttle delay between API calls (seconds)
.\venv\Scripts\python run_benchmark.py --delay 3.0
```

**Reports are saved to:**
- `benchmark/results/benchmark_report.json` — machine-readable data
- `benchmark/results/benchmark_report.md` — human-readable evaluation table

---

## 🧪 Running Tests

```powershell
# Phase 10 — Risk-Based Verification (7 tests)
.\venv\Scripts\python test_phase_10.py

# Phase 11 — Benchmarking & Evaluation (5 tests)
.\venv\Scripts\python test_phase_11.py

# Phases 7, 8, 9 — Diagnosis, Correction, Replanning (9 tests)
.\venv\Scripts\python test_phases_7_8_9.py

# Full comprehensive test suite
.\venv\Scripts\python test_all_phases_comprehensive.py

# Offline (no API key needed — tests logic only)
.\venv\Scripts\python test_offline.py
```

---

## 🌐 Web UI Guide

```powershell
.\venv\Scripts\python web_ui\app.py
# → Open http://localhost:5000
```

| UI Element | What it shows |
| :--- | :--- |
| **Quick Presets** | One-click load of demo goals (Math, Research, RAG, Format) |
| **Mode Selector** | Switch between Adaptive / Fixed / Baseline configurations |
| **Execution Plan** | Each subtask as a card — updates to ✅ / ❌ as steps complete |
| **Live Log** | Every `[Planner]`, `[Executor]`, `[Verifier]` message in real time, color-coded |
| **Verification Results** | Per-step verdict with LOW / MEDIUM / HIGH risk badge |

**Log color coding:**

| Color | Module |
| :--- | :--- |
| 🟣 Purple | `[Planner]` |
| 🟢 Green | `[Executor]` |
| 🟡 Yellow | `[Verifier]` |
| 🟣 Violet | `[RiskClassifier]` |
| 🟠 Orange | `[Corrector]` |
| 🩷 Pink | `[Replanner]` |
| 🔴 Red | Errors & failures |

---

## 🛠️ Tools Available to the Agent

| Tool | Function | Phase |
| :--- | :--- | :---: |
| `calculate` | Safe AST-based math expression evaluator | 2 |
| `web_search` | DuckDuckGo search with multi-backend fallover | 2 |
| `run_python` | Sandboxed Python code execution (10s timeout) | 2 |
| `rag_lookup` | Semantic search over local ChromaDB knowledge base | 5 |
| `knowledge_lookup` | RAG with automatic web fallback if KB score < 0.75 | 5 |

---

## 🔧 Configuration & Environment

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | *(required)* | Your Groq Cloud API key |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | LLM model to use |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | API base URL |
| `LLM_CALL_INTERVAL` | `2.0` | Minimum seconds between LLM calls (rate-limit throttle) |
| `LLM_MAX_RETRY_WAIT` | `120` | Maximum seconds to wait on a rate-limit retry |

---

## ❓ Common Issues & Fixes

| Issue | Fix |
| :--- | :--- |
| `GROQ_API_KEY not set` | Add `GROQ_API_KEY=...` to your `.env` file |
| `Rate limit hit` | Built-in throttle handles it automatically; use `LLM_CALL_INTERVAL=3.0` for slower calls |
| `ChromaDB empty / RAG not working` | Run `.\venv\Scripts\python -m rag.ingest` |
| `UnicodeEncodeError` on Windows | Already handled with `sys.stdout.reconfigure(encoding='utf-8')` in all test files |
| Web UI shows blank plan | The agent is still planning — wait for the `[Planner] Plan created` log line |

---

## 📄 License & Academic Use

This project is developed as an **MCA capstone thesis** at IILM University, Greater Noida (2026). It is intended for academic research and demonstration purposes.

---

*Built with ❤️ using LangGraph · Groq · ChromaDB · Flask · Tailwind CSS*
