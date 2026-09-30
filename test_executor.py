from agent.executor import execute_step


previous_results = [
    {
        "step": "Calculate 25% of 800",
        "result": "200"
    }
]



result = execute_step(
    "Calculate 15% of the result from step 1",
    previous_results
)


print("\nExecution Result:")
print(result)