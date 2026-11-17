# IEEE Research Paper Fact Extraction Report: Adaptive Self-Correcting AI Agent

> **Document Purpose**: Factual reference sheet for technical/academic paper drafting (IEEE Conference/Journal format).  
> **Extraction Policy**: Strictly reports assets, metrics, and parameters verified to exist in the codebase. Any unrecorded element is explicitly marked **NOT FOUND**.

---

## 1. Overview
* **Project Title**: Adaptive Self-Correcting AI Agent (also referenced as *Adaptive Agent Framework*)
* **Domain**: Autonomous Agentic AI, Large Language Models (LLM), Automated Reasoning, Self-Correction, Retrieval-Augmented Generation (RAG).
* **What the Project Does**:
  * An autonomous agent system that accepts high-level natural language goals, plans execution sequences, invokes tools (calculator, sandboxed Python, web search, local vector database), evaluates intermediate outputs through a 3-way verification pipeline, diagnoses failures into a structured error taxonomy, and applies targeted self-correction or dynamic replanning without human intervention.
* **Problem It Solves**:
  * Traditional LLM agents execute subtasks linearly without intermediate verification, leading to compounding tool errors, mathematical hallucination drift, cascading reasoning failures, and restarting entire execution loops upon single-step failures.
  * Naive self-correcting systems that unconditionally run full 3-way verification on every step incur excessive latency, token consumption, and rate-limit exhaustion on trivial steps (e.g., text formatting).
* **Target Users**:
  * AI researchers, software engineers building autonomous multi-step reasoning systems, and academic evaluators studying self-correcting agent architectures.

---

## 2. Objectives and Contributions
* **Main Goals**:
  1. Prevent error propagation in multi-step LLM task execution through intermediate verification.
  2. Implement an autonomous closed-loop diagnosis and self-correction mechanism that repairs failures locally without restarting completed subtasks.
  3. Introduce dynamic risk-tiered verification to balance correctness with execution latency and token efficiency.
  4. Provide empirical benchmarking comparing Baseline, Fixed Self-Correcting, and Adaptive Self-Correcting configurations.
* **What Is New Compared to Existing Solutions (Contributions)**:
  1. **Non-Redundant Execution Preservation**: Verified subtasks are committed to state; correction and dynamic replanning strictly preserve completed steps and re-plan only remaining subtasks.
  2. **Adaptive Risk-Based Verification (Phase 10 Novelty)**: Employs a fast regex rule-based classifier with LLM fallback to assign subtasks to LOW, MEDIUM, or HIGH risk tiers, dynamically adjusting verification depth (skipping heavy LLM calls for formatting while enforcing strict AST/evidence checks for calculations).
  3. **RAG-First with Seamless Web Fallback**: Unified knowledge lookup querying local ChromaDB vector store first (distance threshold $\le 0.75$) and falling back to live DuckDuckGo web search only when local knowledge is absent or low-confidence.
  4. **Integrated 6-Class Error Taxonomy**: Explicit root-cause diagnosis mapping failures to specialized correction strategies (re-computation, query reformulation, citation enforcement).

---

## 3. Background & Literature References
* **Existing Work Mentioned in Text**:
  * ReAct (Reasoning and Acting) paradigm (referenced as baseline agent architecture in `benchmark_specs.md` and `PROJECT_CONTEXT.md`).
  * Plan-and-Solve agent architecture (referenced in `benchmark_specs.md`).
* **Specific Literature References & Authors**: **NOT FOUND** (Note: A project roadmap document refers to *"17 references already compiled"* from a synopsis, but the citation list, author names, and publication venues are not present in the repository).
* **DOIs**: **NOT FOUND**
* **External Links & URLs Found in Codebase**:
  * LangGraph Repository: `https://github.com/langchain-ai/langgraph`
  * Groq Cloud Console: `https://console.groq.com`
  * Groq API Base Endpoint: `https://api.groq.com/openai/v1`
  * HuggingFace Embedding Model Hub: `https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2`
  * Wikipedia Demo/Benchmark Grounding Citations:
    * `https://en.wikipedia.org/wiki/Paris`
    * `https://en.wikipedia.org/wiki/Demographics_of_Tokyo`
    * `https://en.wikipedia.org/wiki/Demographics_of_New_York_City`

---

## 4. Methodology
* **Architecture Type**: State Machine Graph orchestrated with LangGraph (`langgraph.graph.StateGraph`).
* **Graph Control Flow**:
  * Entry Point: `planner_node`
  * Execution Cycle: `planner` $\to$ `executor` $\to$ `verifier`
  * Routing After Verifier (`route_after_verifier`):
    * If passed and steps remain $\to$ `executor`
    * If passed and plan complete $\to$ `END`
    * If failed and `correction_attempts < 2` $\to$ `diagnoser` $\to$ `corrector` $\to$ `verifier`
    * If failed and `correction_attempts >= 2` and `replan_attempts < 2` $\to$ `replanner` $\to$ `executor`
    * If failed and replan budget exhausted $\to$ `give_up` $\to$ `END`
* **Core Modules**:
  1. `Planner` (`agent/planner.py`): Decomposes natural language goals into ordered JSON list of subtasks.
  2. `Executor` (`agent/executor.py`): Multi-turn tool-calling loop dispatching to tools.
  3. `Verifier` (`verification/verifier.py`): Multi-method inspector:
     - Method 1: Numeric Re-Check (deterministic regex normalization + AST calculator).
     - Method 2: Evidence Grounding Check (LLM prompt cross-checking claims against context and source citations).
     - Method 3: Logical Consistency Check (LLM prompt checking subtask satisfaction and contextual contradiction).
  4. `RiskClassifier` (`verification/risk_classifier.py`): Evaluates subtask and result against keyword patterns (`_HIGH_KEYWORDS`, `_MEDIUM_KEYWORDS`, `_LOW_KEYWORDS`) with zero-temperature LLM fallback.
  5. `Diagnoser` (`verification/diagnoser.py`): Maps failures to 6 discrete error categories:
     - `calculation_error`, `tool_error`, `grounding_error`, `consistency_error`, `incomplete_result`, `unknown_error`.
  6. `Corrector` (`correction/corrector.py`): Executes targeted recovery strategies based on diagnosis with anti-redundancy context injection.
  7. `Replanner` (`planner/replanner.py`): Invalidation engine rewriting remaining subtasks while preserving completed subtasks.
* **Key Algorithms & Models Used**:
  * **Primary LLM**: `openai/gpt-oss-20b` via Groq Cloud API.
  * **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, local PyTorch execution).
  * **Vector Database**: `ChromaDB` (persistent local vector store with HNSW indexing).
  * **AST Arithmetic Parser**: Python `ast.parse` with `ast.NodeVisitor` safely evaluating mathematical expressions without `eval()`.
* **Key Equations & Quantitative Formulations**:
  1. **Numeric Verification Tolerance**:
     $$\text{Tolerance} = \max(0.01 \times |\text{stated}|, 0.001)$$
  2. **Percentage Calculation Formula**:
     $$\text{Computed} = \left(\frac{\text{pct}}{100.0}\right) \times \text{base}$$
  3. **RAG Retrieval Relevancy Threshold**:
     $$\text{Relevance Condition}: \text{distance} \le 0.75$$
  4. **Task Success Rate**:
     $$\text{Success Rate (\%)} = \left(\frac{N_{\text{successful}}}{N_{\text{total}}}\right) \times 100$$
  5. **Mean Latency**:
     $$\bar{L} = \frac{\sum_{i=1}^{N} \text{latency}_i}{N_{\text{total}}}$$
  6. **Error Recovery Rate**:
     $$\text{Recovery Rate (\%)} = \left(\frac{N_{\text{recoveries}}}{N_{\text{correction\_attempts}}}\right) \times 100$$
  7. **Efficiency Ratio**:
     $$\text{Efficiency} = \frac{\text{Success Rate (\%)}}{\bar{L} \text{ (seconds)}}$$
  8. **API Backoff Throttle Formula**:
     $$\text{Wait} = 3.0 \times (1.5^{\text{attempt}})$$

---

## 5. Implementation
* **Hardware Environment (from execution environment probe)**:
  * CPU: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel (4 logical cores)
  * Architecture: AMD64 (x86_64)
  * System Memory (RAM): **NOT FOUND** (probe restricted by OS/sandbox permissions; `psutil` uninstalled)
  * GPU / Hardware Accelerators: **NOT FOUND** (PyTorch running on CPU)
* **Operating System**:
  * Windows 11 (`Windows-11-10.0.26300-SP0`)
* **Software Tools & Exact Library Versions**:
  * `python`: 3.14.3
  * `langgraph`: 1.2.19
  * `chromadb`: 1.5.9
  * `sentence-transformers`: 6.2.0
  * `torch`: 2.15.0
  * `openai`: 3.16.2
  * `ddgs`: 9.16.0
  * `flask`: 3.1.3
  * `flask-cors`: 6.0.5
  * `python-dotenv`: 1.2.3
* **Dataset Specifications**:
  * **Internal RAG Knowledge Base (`rag/knowledge_base/`)**:
    * Total Documents: 8 Markdown specification files (`adaptive_risk_tiers.md`, `architecture_spec.md`, `benchmark_specs.md`, `error_taxonomy.md`, `project_overview.md`, `roadmap.md`, `synopsis.md`, `verification_methods.md`).
    * Preprocessing: Character chunking with `chunk_size = 400`, `chunk_overlap = 80`.
    * Total Ingested Chunks: 41 chunks.
    * Embedding Model: `all-MiniLM-L6-v2`.
    * Train / Validation / Test Split: **NOT FOUND** (Knowledge base is an unlabelled inference retrieval corpus, not a model training set).
  * **Evaluation Benchmark Dataset (`benchmark/benchmark_tasks.py`)**:
    * Total Standardized Tasks: 5 tasks.
    * Split Breakdown:
      1. `MATH_01` (Arithmetic / High Risk): Multi-step chained arithmetic.
      2. `RAG_01` (Knowledge Base / Medium Risk): Error taxonomy retrieval.
      3. `FORMAT_01` (Formatting / Low Risk): Text sorting & bulleting.
      4. `RESEARCH_01` (Hybrid Web + Math): Population research and percentage calculation.
      5. `RECOVERY_01` (Error Recovery): Chained percentage and multiplication.
* **Hyperparameters & Operational Limits**:
  * Planner Temperature: 0.2
  * Executor Temperature: 0.2
  * Verifier Temperature: 0.0
  * Risk Classifier Temperature: 0.0
  * Diagnoser Temperature: 0.0
  * Corrector Temperature: 0.2
  * `MAX_CORRECTION_ATTEMPTS`: 2 per step
  * `MAX_REPLANS`: 2 per goal
  * `LangGraph Recursion Limit`: 100
  * `LLM_CALL_INTERVAL`: 2.0 seconds minimum inter-call spacing
  * `LLM_MAX_RETRY_WAIT`: 120.0 seconds
  * `RAG_MAX_DISTANCE`: 0.75
  * `DEFAULT_TOP_K` (RAG): 4
  * `DEFAULT_MAX_WEB_RESULTS`: 3
  * `run_python` Execution Timeout: 10.0 seconds

---

## 6. Results & Benchmark Telemetry
* **Experiments Conducted**:
  * Comparative evaluation across configurations defined in `benchmark/evaluator.py`:
    1. *Baseline*: Zero verification, zero correction.
    2. *Fixed*: Unconditional 3-way verification on every step.
    3. *Adaptive*: Risk-tiered verification (LOW $\to$ 0 LLM calls, MEDIUM $\to$ 1 LLM call, HIGH $\to$ 2 LLM calls).
* **Exact Empirical Values on Record (`benchmark/test_results/benchmark_report.json`)**:

| Configuration | Total Tasks | Success Rate (%) | Mean Latency (s) | Total Verification Calls | Error Recovery Rate (%) | Efficiency Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 1 (`TEST_01`) | **100.0%** | **4.33s** | **0** | 100.0% | **23.09** |
| **Adaptive** | 1 (`TEST_01`) | **100.0%** | **11.28s** | **2** | 100.0% | **8.87** |
| **Fixed** | **NOT FOUND** | **NOT FOUND** | **NOT FOUND** | **NOT FOUND** | **NOT FOUND** | **NOT FOUND** |

* **Verification Checks Detail for Recorded Run**:
  * Baseline: `baseline_skips: 1`, `low_risk: 0`, `medium_risk: 0`, `high_risk: 0`.
  * Adaptive: `baseline_skips: 0`, `low_risk: 0`, `medium_risk: 0`, `high_risk: 1`.
* **Full 5-Task Benchmark Aggregated Run Matrix**: **NOT FOUND** (The automated test verified the harness with task `TEST_01`; the 5-task suite `MATH_01`–`RECOVERY_01` has not been committed to persistent result JSON files).
* **Graph/Chart Data**: **NOT FOUND** (Raw plot files or image charts have not been generated in the workspace).
* **Demonstrated Architectural Limitations**:
  1. Single-agent architecture only (no multi-agent consensus or debate).
  2. Reliance on third-party cloud API rate limits (Groq free tier requires artificial 2.0s delays).
  3. No neural model fine-tuning (in-context prompting and retrieval only).

---

## 7. Figures & Diagrams Available in Project
* **Diagram 1 (ASCII Architecture Diagram)**:
  * Location: `PROJECT_CONTEXT.md` (lines 21–82) and `README.md` (lines 150–185).
  * Description: ASCII schematic showing the complete state machine topology: Planner Node $\to$ Executor Node $\to$ Verifier Node $\to$ Router $\to$ Diagnoser Node $\to$ Corrector Node $\to$ Replanner Node.
  * Suggested Caption: *"Figure 1: State machine architecture of the Adaptive Self-Correcting AI Agent implemented in LangGraph, highlighting the closed-loop verification, diagnosis, correction, and replanning cycles."*
* **Diagram 2 (Web UI Dashboard Screenshot)**:
  * Location: User session upload `media_1794695117232_36b08cb0.png`.
  * Description: Web interface screenshot displaying Goal Input, Quick Presets panel, and real-time streaming execution log.
  * Suggested Caption: *"Figure 2: Interactive Web UI Dashboard featuring real-time Server-Sent Events (SSE) telemetry and execution trace streaming."*
* **Diagram 3 (Server Terminal Initialization Screenshot)**:
  * Location: User session upload `media_1794695131014_55c2f31f.png`.
  * Description: Console output demonstrating Flask development server serving HTTP and SSE stream requests on port 5000.
  * Suggested Caption: *"Figure 3: Backend terminal console showing Flask HTTP and SSE client connection logging during agent runs."*
* **Vector Graphics / High-Resolution Plots (.png, .jpg, .svg, .pdf)**: **NOT FOUND** in workspace repository.

---

## 8. Conclusion Inputs
* **Key Achievements Verified**:
  * Complete operational implementation of all Phases 1 through 12.
  * 100% test pass rate across unit and integration suites (`test_phase_10.py` 7/7 passed, `test_phase_11.py` 5/5 passed, `test_phases_7_8_9.py` 9/9 passed).
  * Successful mitigation of Groq free-tier rate limits via rule-based fast risk classification and an adaptive throttle mechanism.
  * Functional Web UI streaming real-time agent thoughts, tool outputs, and verification badges via SSE.
* **Documented Limitations**:
  * Evaluation corpus currently restricted to 5 synthetic benchmark tasks.
  * Dependency on external web search APIs (DuckDuckGo rate limits / HTML structure changes).
  * Single-threaded LangGraph worker loop without distributed execution.
* **Documented Future Work**:
  * Multi-agent collaborative consensus verification.
  * Domain-specific LLM fine-tuning for error diagnosis and risk classification.
  * Expansion of the benchmark task suite to 50+ real-world industry workflows.

---

## 9. Author & Institutional Details
* **Author Names**:
  1. Daksh
  2. Tarkeshwar Kumar Yadav
  3. Abhinav Sharma
* **Degree / Academic Program**: Master of Computer Applications (MCA in Artificial Intelligence & Machine Learning)
* **Department / School**: **NOT FOUND**
* **Institution**: IILM University, Greater Noida
* **Academic Year**: 2026
* **Author Email Addresses**: **NOT FOUND**
* **Faculty Guide / Project Mentor**: **NOT FOUND**
* **Target Conference / Target Journal Name**: **NOT FOUND** (Referred to generically as *"IEEE paper"*).

---

## 10. Gaps & Missing Information (Checklist for Writers)
The following mandatory academic paper items are marked **NOT FOUND** in the project and must be provided by the authors/supervisors before drafting:

1. [ ] **Author Institutional Emails**: Official academic or personal email addresses for Daksh, Tarkeshwar Kumar Yadav, and Abhinav Sharma.
2. [ ] **Faculty Advisor / Guide Details**: Name, academic designation, department, and email of the faculty mentor.
3. [ ] **Target Publication Venue**: Exact IEEE Conference (e.g., IEEE Access, IEEE ICASSP, IEEE Confluence, etc.) with specific submission track and formatting guidelines.
4. [ ] **17 Literature References**: Complete formal citations (Author names, Title, Conference/Journal, Year, DOI/URL) referenced in the roadmap notes.
5. [ ] **Full 5-Task Comparative Benchmark Data**: Execution of all 5 benchmark tasks across all 3 configurations (`Baseline`, `Fixed`, `Adaptive`) to generate multi-task accuracy, latency, and call reduction tables.
6. [ ] **System Hardware RAM**: Exact physical RAM capacity of the experimental testbed.
7. [ ] **Publication-Ready Figures**: High-resolution vector diagrams (PDF/SVG/PNG) for the architecture pipeline and result bar/line charts.
