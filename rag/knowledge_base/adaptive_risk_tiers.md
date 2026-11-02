# Adaptive Risk-Based Verification Protocol

## Overview
In naive self-correcting agents, verifying every single execution step imposes massive latency overhead and excessive token consumption. The adaptive framework dynamically assigns verification intensity using risk scoring.

## Multi-Factor Risk Assessment
Each subtask result is evaluated across four risk dimensions:
1. Task Importance: Criticality of the subtask relative to the end goal.
2. Model Uncertainty: Degree of hesitation, semantic entropy, or variance in model generations.
3. Source Reliability: Trustworthiness and precision of the data source (e.g., deterministic AST tool vs open web crawl).
4. Downstream Dependency: How many subsequent plan steps rely directly on this step's output.

## Dynamic Risk Tiers
- LOW Tier: Trivial steps or formatting actions. Verification is bypassed to save latency and cost.
- MEDIUM Tier: Standard non-critical deductions. Employs a single lightweight consistency check.
- HIGH Tier: Subtasks affecting multiple downstream steps. Requires detailed evidence grounding or numeric checks.
- CRITICAL Tier: Irreversible or foundational calculations and assertions. Triggers full multi-method verification (numeric, evidence, and logical checks).

