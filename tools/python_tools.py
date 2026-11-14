"""
tools/python_tools.py

Runs short Python snippets in a separate, isolated interpreter process.

This is a convenience execution tool, not a hardened security sandbox:
the child process is started with ``-I`` (isolated mode, no user site
packages / PYTHON* env vars), runs inside a temporary working directory, and
is killed after a timeout. It still runs with the current user's permissions,
so untrusted code should not be executed with it.
"""

import os
import subprocess
import sys
import tempfile


DEFAULT_TIMEOUT = 10
MAX_OUTPUT_CHARS = 4000


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"


def run_python(code: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Execute a short Python code snippet in a separate subprocess.

    Args:
        code:    Python code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        The program's output or an error message.
    """
    if not code or not code.strip():
        return "Error: No Python code provided."

    try:
        with tempfile.TemporaryDirectory(prefix="agent_python_") as work_dir:
            script_path = os.path.join(work_dir, "snippet.py")
            with open(script_path, "w", encoding="utf-8") as script:
                script.write(code)

            result = subprocess.run(
                [sys.executable, "-I", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
            )

        if result.returncode != 0:
            return "Python error:\n" + _truncate(result.stderr.strip())

        output = result.stdout.strip()
        return _truncate(output) if output else "(Code executed successfully with no output)"

    except subprocess.TimeoutExpired:
        return f"Error: Python execution timed out after {timeout} seconds."

    except Exception as exc:
        return f"Error: {exc}"


# Tool schema for the LLM
PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute a short Python code snippet and return its stdout. "
            "You MUST pass the Python code in the 'code' parameter string. "
            "Example: {\"code\": \"print(sum(range(1, 10)))\"}. "
            "Never put result values in this parameter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python source code to execute as a string.",
                }
            },
            "required": ["code"],
        },
    },
}


def handle_tool_call(arguments: dict):
    if not isinstance(arguments, dict):
        return "Error: Tool arguments must be a dictionary."

    for key in ("code", "script", "command", "source"):
        if key in arguments and arguments[key]:
            return run_python(str(arguments[key]))

    return (
        "Error: Missing 'code' parameter for run_python. "
        "Please provide the Python code to run: {'code': 'print(...)'}."
    )
