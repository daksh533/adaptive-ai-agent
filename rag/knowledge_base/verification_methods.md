# Multi-Method Verification Framework

## Core Verification Protocols
To ensure robustness without relying solely on subjective model confidence, the framework specifies three rigorous verification methods:

### 1. Numeric Re-Check
Independently re-evaluates all quantitative claims and arithmetic deductions. Uses an Abstract Syntax Tree (AST) evaluator or deterministic Python execution to verify mathematical statements independently from the generation pass.

### 2. Evidence Grounding Check
Verifies whether textual assertions made in an intermediate result are directly supported by evidence extracted from RAG retrieval chunks or authenticated search snippets. Explicitly checks for hallucinated citations, distorted quotations, and unsupported extrapolations.

### 3. Logical Consistency Check
Evaluates whether the intermediate result directly and completely satisfies the objective requested by the current subtask. Flags non-sequiturs, contradictory statements, or partial answers that fail to advance the overarching goal.

