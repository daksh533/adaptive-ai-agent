
from agent.state import AgentState


state = AgentState(
    goal="Calculate 25% of 800, then calculate 15% of that result."
)

state.plan = [
    "Calculate 25% of 800",
    "Calculate 15% of the first result"
]

state.results_so_far.append({
    "step": "Calculate 25% of 800",
    "result": 200
})

state.current_step = 1

print("Goal:", state.goal)
print("Plan:", state.plan)
print("Current step:", state.current_step)
print("Results:", state.results_so_far)

