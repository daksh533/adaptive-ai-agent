# Project Roadmap — Adaptive Self-Correcting AI Agent

## Overview
The Adaptive Self-Correcting AI Agent is an autonomous LLM agent framework that detects, diagnoses, and corrects execution mistakes across complex multi-step reasoning workflows. It prevents cumulative reasoning failures, compounding tool errors, and unverified hallucinations.

## Authors
- Daksh, Tarkeshwar Kumar Yadav, Abhinav Sharma
- MCA in Artificial Intelligence & Machine Learning
- IILM University, Greater Noida
- Academic Year: 2026

## Phase Roadmap (13 Phases)

| Phase | Component | Status | Description |
|-------|-----------|--------|-------------|
| 1 | Foundation & LLM API setup | Complete | Groq API integration with GPT-OSS-20B model |
| 2 | Deterministic & sandboxed tool integration | Complete | AST Calculator, Python Subprocess, Web Search |
| 3 | Basic Agent Control Loop | Complete | Planner-Executor architecture |
| 4 | State Graph Migration using LangGraph | Complete | Compiled LangGraph state machine |
| 5 | Knowledge Base Grounding & RAG | Complete | ChromaDB vector retrieval with sentence-transformers |
| 6 | Comprehensive Verification Module | Complete | Numeric re-check, evidence grounding, logical consistency |
| 7 | Structured Error Diagnosis | Complete | 6-class error taxonomy (calculation, tool, grounding, consistency, incomplete, unknown) |
| 8 | Targeted Self-Correction Engine | Complete | Targeted recovery routines per error class |
| 9 | Dynamic Plan Invalidation & Replanning | Complete | Recomputes remaining subtasks after repeated failures |
| 10 | Adaptive Risk-Based Verification | Complete | Risk classifier (LOW/MEDIUM/HIGH) dynamically scales verification depth per subtask |
| 11 | Empirical Benchmarking & Evaluation | Complete | Standardized task suite, comparative matrix (Baseline vs. Fixed vs. Adaptive), metrics & reports |
| 12 | Interactive Visual Web UI Dashboard | Planned | Modern web UI to visualize live execution, verification, and correction traces |
| 13 | Research Paper & Thesis Defense | Planned | Academic write-up and defense |

## Core Tools
- `calculate`: Safe mathematical expression evaluation (AST-based)
- `web_search`: Internet search via DuckDuckGo
- `run_python`: Sandboxed Python code execution (10s timeout)
- `rag_lookup`: Semantic search over local ChromaDB knowledge base

## Graph Flow (Phases 6-9)
START → Planner → Executor → Verifier
                    ↓
             passed → Executor (next step) | END
             failed → Diagnoser → Corrector → Verifier (re-verify)
                                    ↓
                      max attempts → Replanner → Executor | END

## Correction Limit
Maximum correction attempts per failed step: 2
After 2 failed correction attempts, the Replanner revises the remaining plan.