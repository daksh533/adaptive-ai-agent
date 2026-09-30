from agent.state import AgentState
from agent.planner import create_plan
from agent.executor import execute_step


def run_agent(goal: str):
    """
    Run the complete multi-step agent loop.

    Flow:
    Plan → Execute → Observe → Update State → Next Step
    """

    # 1. Create initial state
    state = AgentState(goal=goal)

    # 2. Create the plan
    print("\nCreating plan...")
    state.plan = create_plan(goal)

    print("\nPlan:")
    for i, step in enumerate(state.plan, start=1):
        print(f"{i}. {step}")

    # 3. Execute each step
    while state.current_step < len(state.plan):

        current_subtask = state.plan[state.current_step]

        print(
            f"\nExecuting Step "
            f"{state.current_step + 1}: {current_subtask}"
        )

        # Execute current step
        result = execute_step(
            current_subtask,
            state.results_so_far
        )

        # Store result
        state.results_so_far.append(result)

        # Move to next step
        state.current_step += 1

        print(f"Result: {result['result']}")

    # 4. Return final state
    return state