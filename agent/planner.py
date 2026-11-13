import os
import json
import time
import re

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError


load_dotenv()


client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


def call_with_retry(**kwargs):
    """
    Call Groq API with exponential backoff on HTTP 429 RateLimitError.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            wait_time = 2.0 * (1.5 ** attempt)
            ms_match = re.search(r"try again in ([\d\.]+)ms", str(e))
            s_match = re.search(r"try again in ([\d\.]+)s", str(e))
            if ms_match:
                wait_time = max(float(ms_match.group(1)) / 1000.0 + 0.5, 1.5)
            elif s_match:
                wait_time = max(float(s_match.group(1)) + 0.5, 2.0)

            print(f"[Planner] Rate limit hit. Pausing {wait_time:.1f}s before retrying...")
            time.sleep(wait_time)

    return client.chat.completions.create(**kwargs)


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
                "4. The agent uses built-in programmatic tools: calculate, web_search, run_python, and rag_lookup. "
                "Do not create subtasks asking to open external desktop apps or GUI programs.\n"
                "5. Do not perform the task yourself.\n"
                "6. Do not include explanations outside the JSON."
            )
        },
        {
            "role": "user",
            "content": goal
        }
    ]

    response = call_with_retry(
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
