# Taxonomy of Errors in Autonomous Agentic Systems

## Error Classification Schema
The framework establishes a six-category classification taxonomy to systematically analyze and remediate execution mistakes:

### 1. MISSING_INFO
Occurs when an execution step fails because required factual context, document references, or intermediate variables are missing or unretrieved.

### 2. WRONG_CALCULATION
Occurs when arithmetic, algebraic, or numerical calculations deviate from mathematical truth. Commonly caused by LLM mental math approximations or malformed expressions passed to calculators.

### 3. TOOL_FAILURE
Occurs when an external tool throws an exception, encounters timeouts, suffers networking outages, or rejects input parameters due to schema violations.

### 4. RETRIEVAL_FAILURE
Occurs when the RAG retrieval module returns irrelevant chunks, empty result sets, or out-of-domain documents that fail to address the query.

### 5. REASONING_ERROR
Occurs when the agent arrives at invalid logical deductions despite possessing accurate factual evidence and correct tool outputs.

### 6. PLANNING_ERROR
Occurs when the decomposing Planner creates an infeasible order of steps, circular dependencies, or omits prerequisite actions.

