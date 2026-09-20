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


# Take input from the user
user_input = input("You: ")


messages = [
    {
        "role": "system",
        "content": (
            "You are an AI agent with access to three tools: "
            "calculate, web_search, and run_python. "
            "Use calculate for mathematical expressions. "
            "Use web_search when the user asks for current information "
            "or an internet search. "
            "Use run_python for computation, data processing, loops, "
            "or tasks that genuinely require Python. "
            "When calling run_python, its arguments MUST be a JSON object "
            "containing a 'code' field with the Python code as a string. "
            "Example: {\"code\": \"print(2 + 2)\"}. "
            "Never put raw Python code directly in the arguments field."
        )
    },
    {
        "role": "user",
        "content": user_input
    }
]


# Agent loop
while True:

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=TOOL_SCHEMAS,
    )

    message = response.choices[0].message

    # If the LLM wants to use a tool
    if message.tool_calls:

        # Add the assistant's tool request to conversation
        messages.append(message)

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print(f"Tool used: {tool_name}")

            # Send tool call to registry
            tool_result = dispatch(
                tool_name,
                arguments
            )

            # Add tool result to conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                }
            )

        # Continue the loop so the LLM can process
        # the tool result and give an answer
        continue

    # No tool call = final answer
    print("AI:", message.content)
    break