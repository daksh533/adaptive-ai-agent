import subprocess
import sys
import tempfile
import os


# Maximum time a Python program is allowed to run
DEFAULT_TIMEOUT = 7


def run_python(code: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Execute a short Python code snippet in a separate subprocess.

    Args:
        code: Python code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        The program's output or an error message.
    """

    if not code or not code.strip():
        return "Error: No Python code provided."

    temp_file = None

    try:
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as file:
            file.write(code)
            temp_file = file.name

        # Run the Python file in a separate subprocess
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # If the program produced an error
        if result.returncode != 0:
            return f"Python error:\n{result.stderr.strip()}"

        # Return program output
        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        return f"Error: Python execution timed out after {timeout} seconds."

    except Exception as e:
        return f"Error: {str(e)}"

    finally:
        # Delete temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


# Tool schema for the LLM
PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute a short Python code snippet and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Put the complete code inside this string."
                }
            },
            "required": ["code"],
            "additionalProperties": False
        }
    }
}


def handle_tool_call(arguments: dict):
    if "code" not in arguments:
        return "Error: Missing 'code' argument."

    return run_python(arguments["code"])