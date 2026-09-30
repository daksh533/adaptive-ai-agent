from agent.planner import create_plan


goal = (
    "Calculate 25% of 800, "
    "then calculate 15% of that result, "
    "then add 50."
)

plan = create_plan(goal)

print("\nGenerated Plan:\n")

for i, step in enumerate(plan, start=1):
    print(f"Step {i}: {step}")