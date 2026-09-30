import os
import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


def create_plan(goal: str) -> list[str]:
    """
    Convert the user's goal into an ordered list of subtasks.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a task planner for an AI agent. "
                "Break the user's goal into a small number of clear, "
                "ordered subtasks.\n\n"

                "Return ONLY valid JSON in exactly this format:\n"
                "{"
                "\"plan\": ["
                "\"step 1\", "
                "\"step 2\", "
                "\"step 3\""
                "]"
                "}\n\n"

                "Rules:\n"
                "1. Each item must be one clear subtask.\n"
                "2. Put subtasks in the correct execution order.\n"
                "3. Later steps may depend on results from earlier steps.\n"
                "4. Do not perform the task yourself.\n"
                "5. Do not include explanations outside the JSON."
            )
        },
        {
            "role": "user",
            "content": goal
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
    )

    content = response.choices[0].message.content

    data = json.loads(content)

    plan = data.get("plan")

    if not isinstance(plan, list) or not plan:
        raise ValueError("Planner returned an invalid plan.")

    if not all(isinstance(step, str) for step in plan):
        raise ValueError("Every plan step must be a string.")

    return plan