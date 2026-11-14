# Complete Project Context & Architecture Specification
**Project**: Adaptive Self-Correcting AI Agent  
**Environment**: Python 3.14 on Windows, Groq Cloud API (`openai/gpt-oss-20b` via OpenAI SDK), LangGraph, ChromaDB (`sentence-transformers/all-MiniLM-L6-v2`), DuckDuckGo Search (`ddgs`).

---

## 1. Executive Summary & Purpose
This project implements an autonomous, adaptive, self-correcting AI agent designed around **LangGraph**. The agent accepts high-level, multi-step natural language goals, breaks them down into an ordered execution plan, executes each step using programmatic tools, inspects the result through an independent 3-method verification engine, diagnoses any detected failure into a structured 6-class error taxonomy, and applies targeted self-correction or dynamic replanning.

A central engineering principle of the system is **non-redundancy**:
1. **Never Re-Do Completed Work**: When an intermediate step fails verification and enters correction or dynamic replanning, the agent strictly preserves already completed and verified steps. It never re-executes previous steps or restarts from Step 1.
2. **RAG-First with Seamless Web Fallback**: When looking up facts, documents, or external data, the agent checks its local vector knowledge base (ChromaDB) first. If relevant chunks are present (distance $\le 0.75$), it answers with local document citations. If the knowledge base lacks the information (distance $> 0.75$ or empty), it automatically queries live web search (DuckDuckGo), cites web URLs, takes the information, and proceeds immediately to the next step without staying stuck in search loops.

---

## 2. Architecture & Control Flow Graph

The state machine is compiled using **LangGraph** (`agent/graph.py`) over a shared `GraphState` TypedDict (`agent/graph_state.py`).

### Visual Flow Diagram
```
                      +-------------------+
                      |   START / Input   |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   planner_node    |
                      +---------+---------+
                                |
                                v
           +--------->+-------------------+
           |          |   executor_node   |<-------------------------+
           |          +---------+---------+                          |
           |                    |                                    |
           |                    v                                    |
           |          +-------------------+                          |
           |          |   verifier_node   |                          |
           |          +---------+---------+                          |
           |                    |                                    |
   (verdict == "passed")        | (verdict == "failed")              |
           |                    v                                    |
           |          +-------------------+                          |
           |          |   route_verifier  |                          |
           |          +----+---------+----+                          |
           |               |         |                               |
    [more steps?]          |         |                               |
     /         \           |         |                               |
   yes          no         |         |                               |
   /             \         |         |                               |
  v               v        |         |                               |
continue         END       |         |                               |
                           |         |                               |
              attempts < 2 |         | attempts >= 2                 |
                           v         v                               |
             +---------------+     +----------------+                |
             | diagnoser_node|     | replanner_node |----------------+
             +-------+-------+     +----------------+ (replan_attempts < 2)
                     |                      |
                     v                      | replan_attempts >= 2
             +---------------+              v
             | corrector_node|      +---------------+
             +-------+-------+      | give_up_node  |
                     |              +-------+-------+
                     +                      |
          (re-verify corrected result)      v
                     |                     END
                     v
             [back to verifier]
```

---

## 3. Shared State Contract (`agent/graph_state.py`)

Every node receives and updates a typed dictionary with the following schema:

```python
class GraphState(TypedDict):
    goal: str                             # The original natural-language user task
    plan: list[str]                       # Full ordered list of subtasks (completed + pending)
    current_step: int                     # Zero-based index of the subtask currently active
    results_so_far: list[dict[str, Any]]  # Execution history entries:
                                          # {
                                          #   "step": str,
                                          #   "result": str,
                                          #   "verification_status": "passed" | "failed" | "pending",
                                          #   "verification_reason": str,
                                          #   "error_diagnosis": str,
                                          #   "correction_attempts": int
                                          # }
    verification_status: str              # "passed" | "failed" | "pending"
    verification_reason: str              # Diagnostic message explaining pass/fail
    error_diagnosis: str                  # Classified taxonomy category (e.g., "calculation_error")
    correction_attempts: int              # Counter for corrections on current step (max: 2)
    replan_attempts: int                  # Counter for full plan rewrites (max: 2)
    is_done: bool                         # Boolean indicating overall execution completion
    agent_status: str                     # "running" | "completed" | "failed"
```

---

## 4. Detailed Component Walkthrough

### Phase 1: Shared LLM Gateway & Rate Limiting (`llm_client.py`)
* Connects to Groq using the official `OpenAI` client pointing to `https://api.groq.com/openai/v1`.
* Model: `openai/gpt-oss-20b` (or active model in Groq configuration).
* **Exponential Backoff**: Wrapped with `call_with_retry`. When Groq returns HTTP 429 (`rate_limit_exceeded` on 8,000 TPM limit) or HTTP 503, it parses the retry duration, pauses cleanly, and retries up to 5 times.
* **Resilience**: JSON extractors (`extract_json_dict`, `extract_plan`) use regex fallback to extract JSON objects or arrays even if the LLM wraps them in conversational text or markdown blocks.

### Phase 2: Tool Registry & Deterministic Tools (`tools/`)
The agent has access to 5 registered tools via `tools/registry.py`:
1. `calculate` (`tools/calculator.py`): Deterministic arithmetic engine using Python's `ast` module. Safe against code injection (supports `+`, `-`, `*`, `/`, `^`, `%`, `**`).
2. `run_python` (`tools/python_tools.py`): Executes Python scripts in an isolated subprocess (`python -c ...`) with timeout and captured `stdout`/`stderr`. Schema accepts parameters `code`, `script`, or `command`.
3. `web_search` (`tools/web_search.py`): Live internet search using DuckDuckGo (`ddgs`). Features automated backend failover (`auto` $\to$ `html` $\to$ `lite`) with pause and retry to eliminate network rate glitches.
4. `rag_lookup` (`tools/rag_lookup.py`): Agent tool interface that searches the local ChromaDB store first and automatically falls back to web search if missing.
5. `knowledge_lookup` (`tools/knowledge_lookup.py`): Unified RAG-first tool with explicit source tagging: `[SOURCE: KNOWLEDGE BASE]`, `[SOURCE: WEB FALLBACK]`, or `[SOURCE: UNAVAILABLE]`.

### Phase 3: Task Planner (`agent/planner.py`)
* Receives the user's high-level goal and prompts the LLM to output a JSON list of subtasks: `{"plan": ["step 1", "step 2", ...]}`.
* Explicit prompt instructions prevent the planner from generating redundant retrieval steps (e.g., instead of creating separate "Search KB" and "Search Web" steps, it creates a single unified lookup step).

### Phase 4: Executor Node (`agent/executor.py`)
* Executes a single subtask (`plan[current_step]`) at a time.
* Context passing: Uses `_passed_results(results_so_far)` so only successfully verified earlier steps are passed as context (condensed to prevent token explosion).
* Tool Turn Loop: Supports up to `MAX_TOOL_TURNS = 3`. Tool calls are dispatched, output is appended as `role: "tool"`, and the model is prompted to synthesize its final text answer.
* Tool Formatting Guard: If the model generates malformed arguments or invalid parameters, it catches `BadRequestError` and falls back to plain text recovery without crashing.

### Phase 5: RAG Pipeline & Calibrated Web Fallback (`rag/`, `tools/knowledge_lookup.py`)
* Ingestion (`rag/ingest.py`): Markdown/text documents from `rag/knowledge_base/` are chunked and embedded using `sentence-transformers/all-MiniLM-L6-v2`. Stored locally in `rag/chroma_store`.
* Retrieval (`rag/retriever.py`): Queries ChromaDB for top-$k$ nearest neighbors by cosine distance.
* **Calibrated Relevance Threshold**: The relevance distance threshold is set to **`0.75`** (`RAG_MAX_DISTANCE`).
  * If nearest chunk distance $\le 0.75$: Valid local hit $\to$ returns text tagged with `[SOURCE: KNOWLEDGE BASE]` and document chunk metadata.
  * If nearest chunk distance $> 0.75$ (or collection empty/unavailable): Irrelevant match $\to$ immediately triggers `web_search(query)` and returns results tagged with `[SOURCE: WEB FALLBACK]` and source URLs.
  * The executor system prompt instructs the agent to accept the retrieved fallback data immediately and answer the subtask, preventing infinite search loops on the same step.

### Phase 6: 3-Way Verification Engine (`verification/verifier.py`)
Every step result is inspected before state commit using three independent methods:
1. **Numeric Re-check**: Normalizes LaTeX/Markdown, extracts math expressions (e.g. `25% of 800 = 200` or `15 * 10 = 150`) using regex, independently calculates them using AST `calculate`, and asserts relative tolerance ($< 1\%$).
2. **Evidence Grounding**: LLM check cross-referencing claims against earlier step results and cited sources (requires valid KB citations or live web URLs).
3. **Logical Consistency**: LLM check confirming the answer directly satisfies the subtask requirements without internal contradictions.
* Result: A `VerificationResult(status="passed"|"failed", method=..., reason=...)`.

### Phase 7: Error Diagnoser (`verification/diagnoser.py`)
When a step fails verification, the diagnoser analyzes the subtask, the bad result, the verifier failure reason, and past context to classify the failure into one of 6 categories:
* `calculation_error`: Arithmetic mismatch or incorrect numerical operation.
* `tool_error`: Tool returned an error, timed out, or parameter format failed.
* `grounding_error`: Unsupported factual claims or missing citations/URLs.
* `consistency_error`: Contradictory statements or internal logic mismatch.
* `incomplete_error`: Answer omitted required parts of the subtask.
* `unknown_error`: Unclassified or ambiguous failure.

### Phase 8: Targeted Self-Correction Engine (`correction/corrector.py`)
Applies a specific recovery strategy based on the diagnosed error class:
* `_correct_calculation`: Uses `CALCULATOR_TOOL_SCHEMA` with `tool_choice="auto"`.
* `_correct_tool`: Retries with reformulated tool calls using all schemas.
* `_correct_grounding`: Searches for supporting evidence using `knowledge_lookup` or `web_search`.
* `_correct_consistency` / `_correct_incomplete` / `_correct_unknown`: LLM-guided prompt refinement.
* **Anti-Redundancy Fix**: Every correction function receives `_previous_context(previous_results)`. Prompts explicitly instruct the LLM **not to re-do earlier completed steps**. After tool execution, the final answer turn sets `tools=None` so the model is forced to output plain text rather than entering tool loops.
* Maximum attempts per step: `MAX_CORRECTION_ATTEMPTS = 2`.

### Phase 9: Dynamic Replanner (`planner/replanner.py`, `agent/graph.py`)
Triggered when a subtask exhausts its 2 correction attempts without passing:
* Inputs: `goal`, `completed_results` (only passed steps), `remaining_subtasks`, and `failure_reason`.
* Deduplication & Filter: The prompt explicitly forbids repeating completed work. A deduplication filter removes any generated subtask that matches an already completed step.
* State Reconstruction in `replanner_node`:
  ```python
  completed_subtasks = [r["step"] for r in completed]
  full_plan = completed_subtasks + revised_remaining
  next_step_index = len(completed_subtasks)
  ```
  * Instead of resetting `current_step = 0`, `current_step` is set to `len(completed_subtasks)`. The agent executes only the newly revised steps (e.g., Step 2/3 becomes Step 2/3 revised, never repeating Step 1).
* Bound: Maximum 2 replans per goal (`MAX_REPLANS = 2`). If still failing, graph cleanly terminates at `give_up_node`.

---

## 5. File & Directory Reference Map

| Path | Purpose |
| :--- | :--- |
| `main.py` | CLI entry point with interactive prompt, execution trace, and summary display. |
| `llm_client.py` | Central Groq LLM client, retry wrapper, and JSON parsing helpers. |
| `agent/graph.py` | Main LangGraph state graph definition, node functions, and routing logic. |
| `agent/graph_state.py` | TypedDict defining the state schema shared across all nodes. |
| `agent/planner.py` | Initial goal decomposition into ordered subtasks. |
| `agent/executor.py` | Subtask executor, tool dispatch loop, and schema recovery. |
| `agent/loop.py` | Reusable graph invocation entry point (`run_agent(goal)`). |
| `tools/registry.py` | Schema aggregator (`TOOL_SCHEMAS`) and tool `dispatch()` router. |
| `tools/calculator.py` | AST-based deterministic arithmetic evaluator. |
| `tools/python_tools.py` | Isolated subprocess Python execution engine. |
| `tools/web_search.py` | Multi-backend resilient DuckDuckGo web search. |
| `tools/knowledge_lookup.py` | Unified RAG search with calibrated 0.75 distance threshold and live web fallback. |
| `tools/rag_lookup.py` | Tool schema and router delegating to `knowledge_lookup`. |
| `verification/verifier.py` | 3-way verifier (Numeric, Evidence Grounding, Logical Consistency). |
| `verification/diagnoser.py` | 6-class error taxonomy classifier. |
| `correction/corrector.py` | Error-specific correction strategies with context preservation. |
| `planner/replanner.py` | Dynamic remaining-subtask replanning and deduplication. |
| `rag/ingest.py` | Document reader, chunker, and vector store populator. |
| `rag/retriever.py` | Cosine similarity vector search over ChromaDB. |
| `rag/knowledge_base/` | Source markdown knowledge documents (`roadmap.md`, `synopsis.md`, etc.). |
| `test_all_phases_comprehensive.py` | Full integration test suite testing Phases 1 through 9 and end-to-end runs. |
| `test_workflow_and_correction.py` | Targeted test suite validating RAG fallback and replan anti-redundancy. |
| `test_phases_7_8_9.py` | Unit tests for error diagnosis, targeted correction, and plan revision. |

---

## 6. How to Run & Verify the Codebase

All dependencies are installed in the workspace Python virtual environment (`venv/`).

### Run the Interactive CLI
```powershell
.\venv\Scripts\python main.py
```

### Run Comprehensive Test Suite (All Phases 1–9)
```powershell
.\venv\Scripts\python test_all_phases_comprehensive.py
```
*Tests executed:*
1. Phase 2 Deterministic Tools (`calculate` and `run_python`).
2. Phase 5 RAG-First Retrieval (Local KB hit vs. Live Web Fallback).
3. Phase 7 & 8 Error Diagnosis & Targeted Self-Correction.
4. Phase 9 Dynamic Replanner Subtask Preservation.
5. End-to-End Multi-Phase Goal Execution.

### Run Targeted Workflow & Fallback Tests
```powershell
.\venv\Scripts\python test_workflow_and_correction.py
```
*Validates that:*
1. Queries present in `rag/knowledge_base/` hit ChromaDB with distance $\le 0.75$.
2. Queries missing from the knowledge base trigger DuckDuckGo web search with URLs.
3. Steps completed prior to replanning are never re-executed.
