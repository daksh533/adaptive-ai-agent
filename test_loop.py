from agent.loop import run_agent


goal = (
    "Calculate 25% of 800, "
    "then calculate 15% of that result, "
    "then add 50 to the final result."
)


state = run_agent(goal)


print("\n==============================")
print("FINAL RESULTS")
print("==============================")

for result in state.results_so_far:
    print(f"\nStep: {result['step']}")
    print(f"Result: {result['result']}")

    