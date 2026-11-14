# Project Synopsis — Adaptive Self-Correcting AI Agent

## Executive Summary
Autonomous Large Language Model (LLM) agents frequently suffer from cumulative reasoning failures, compounding tool errors, and unverified hallucinations during multi-step planning tasks. The Adaptive Self-Correcting AI Agent project introduces an intelligent agent framework that autonomously detects, diagnoses, and corrects execution mistakes across complex multi-step reasoning workflows.

## Problem Statement
Traditional LLM agents execute subtasks linearly without verifying intermediate results. When a step fails, the error propagates to subsequent steps, causing cascading failures and incorrect final outputs. There is no mechanism to detect, classify, or self-correct errors mid-execution.

## Solution
A LangGraph-based agent with:
- **Planner**: Decomposes goals into ordered subtasks
- **Executor**: Executes subtasks using 4 built-in tools (calculate, web_search, run_python, rag_lookup)
- **Verifier**: Checks every result for numeric accuracy, evidence grounding, and logical consistency
- **Diagnoser** (Phase 7): Classifies failures into 6 error categories
- **Corrector** (Phase 8): Applies targeted fixes per error class
- **Replanner** (Phase 9): Rewrites remaining plan after repeated failures

## Key Features
- Self-verification of every step before committing to state
- Structured error diagnosis with 6-class taxonomy
- Targeted self-correction strategies per error type
- Dynamic replanning when corrections are exhausted
- RAG-based knowledge grounding for factual accuracy
- Configurable correction attempt limit (default: 2)

## Technical Stack
- Python 3.10+
- LangGraph (state graph orchestration)
- Groq API (GPT-OSS-20B model)
- ChromaDB (vector database)
- Sentence-Transformers (embeddings)
- DuckDuckGo Search (web search)
- python-dotenv (environment management)

## Academic Context
MCA in Artificial Intelligence & Machine Learning
IILM University, Greater Noida
Academic Year: 2026