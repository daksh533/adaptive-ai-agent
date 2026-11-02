# Evaluation Benchmark and Experimental Methodology

## Comparative Configurations
To empirically demonstrate the value of adaptive verification, the system evaluates three distinct operating configurations:
1. Baseline Configuration: Standard ReAct/Plan-and-Solve agent with zero verification and no self-correction.
2. Fixed Self-Correcting Configuration: Full verification and error recovery executed unconditionally on every subtask.
3. Adaptive Self-Correcting Configuration: Novel risk-gated verification scaling verification depth dynamically.

## Key Evaluation Metrics
- Task Success Rate (%): Proportion of benchmark multi-step tasks completed accurately.
- Error Recovery Rate (%): Frequency with which detected failures are remediated into valid outputs.
- Latency (seconds): Wall-clock time required to complete multi-step goals end-to-end.
- Token and Call Overhead: Total LLM tokens consumed and count of verification calls per task.
- Efficiency Ratio: Ratio of accuracy gains relative to verification computational overhead.

