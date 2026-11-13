import os
import json
import time
import re

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError, RateLimitError

from tools.registry import TOOL_SCHEMAS, dispatch


load_dotenv()


client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


def call_with_retry(**kwargs):
    """
    Call Groq API with exponential backoff on HTTP 429 RateLimitError.
    """
    max_retries = 6
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

            print(f"[Executor] Rate limit hit (TPM). Pausing {wait_time:.1f}s before retrying...")
            time.sleep(wait_time)

    return client.chat.completions.create(**kwargs)


def execute_step(
    subtask: str,
    previous_results: list[dict]
) -> dict:
    """
    Execute one subtask using the available tools.
    Limits tool turns to avoid infinite loops and conserve TPM.
    """

    previous_context = ""

    if previous_results:
        condensed_results = []
        for r in previous_results:
            step_name = r.get("step", "")
            res_str = str(r.get("result", ""))
            if len(res_str) > 500:
                res_str = res_str[:500] + "... [truncated]"
            condensed_results.append({"step": step_name, "result": res_str})

        previous_context = (
            "\n\nResults from previous steps:\n"
            + json.dumps(condensed_results, indent=2)
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are the executor of an AI agent.\n\n"

                "Your job is to execute ONE subtask at a time.\n\n"

                "You have access to these tools:\n"
                "- calculate: mathematical calculations. Parameter: {'expression': '<math expr>'}\n"
                "- web_search: current information or internet search. Parameter: {'query': '<search query>'}\n"
                "- run_python: Python computation or data processing. Parameter: {'code': '<python code string>'}\n"
                "  IMPORTANT for run_python: Put the Python code in the 'code' parameter string, "
                "e.g. {\"code\": \"print(...) \"}. Do NOT put answers or stdout in arguments.\n"
                "- rag_lookup: search local knowledge base documents. Parameter: {'query': '<search query>'}\n\n"

                "Choose the appropriate tool when needed.\n"
                "Use previous step results when the current subtask depends on them.\n\n"

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

    retries = 0
    max_retries = 3
    tool_turns = 0
    max_tool_turns = 3

    while retries < max_retries:
        try:
            response = call_with_retry(
                model="openai/gpt-oss-20b",
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
        except BadRequestError as e:
            retries += 1
            print(f"[Executor] Tool call formatting error: {e}. Retrying step ({retries}/{max_retries})...")
            messages.append({
                "role": "user",
                "content": (
                    "Please provide the final result for this subtask in plain text without malformed tool arguments."
                )
            })
            continue

        message = response.choices[0].message

        if message.tool_calls:
            tool_turns += 1
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(tool_call.function.arguments)
                except Exception:
                    arguments = {"code": tool_call.function.arguments}

                print(f"Tool used: {tool_name}")

                tool_result = dispatch(tool_name, arguments)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)[:1000]
                    }
                )

            if tool_turns >= max_tool_turns:
                messages.append({
                    "role": "user",
                    "content": "You have received the tool results. Please now provide your complete final answer for this subtask."
                })

            continue

        result = message.content

        return {
            "step": subtask,
            "result": result
        }

    return {
        "step": subtask,
        "result": "Error: Exceeded maximum tool execution retries."
    }
