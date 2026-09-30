import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from tools.registry import TOOL_SCHEMAS, dispatch


load_dotenv()


client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


def execute_step(
    subtask: str,
    previous_results: list[dict]
) -> dict:
    """
    Execute one subtask using the available tools.
    """

    previous_context = ""

    if previous_results:
        previous_context = (
            "\n\nResults from previous steps:\n"
            + json.dumps(previous_results, indent=2)
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are the executor of an AI agent.\n\n"

                "Your job is to execute ONE subtask at a time.\n\n"

                "You have access to these tools:\n"
                "- calculate: mathematical calculations\n"
                "- web_search: current information or internet search\n"
                "- run_python: Python computation or data processing\n\n"

                "Choose the appropriate tool when needed.\n"
                "Use previous step results when the current subtask "
                "depends on them.\n\n"

                "Do not try to execute future subtasks.\n"
                "Complete only the current subtask."
            )
        },
        {
            "role": "user",
            "content": (
                f"Current subtask:\n{subtask}"
                f"{previous_context}"
            )
        }
    ]

    while True:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        message = response.choices[0].message

        if message.tool_calls:

            messages.append(message)

            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name

                arguments = json.loads(
                    tool_call.function.arguments
                )

                print(f"Tool used: {tool_name}")

                tool_result = dispatch(
                    tool_name,
                    arguments
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    }
                )

            continue

        result = message.content

        return {
            "step": subtask,
            "result": result
        }