import ast
import math
import operator

# Allowed operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed functions and constants
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "factorial": math.factorial,
    "abs": abs,
    "round": round,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} not allowed.")
        return _OPERATORS[op_type](_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unary operator {op_type.__name__} not allowed.")
        return _OPERATORS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("Only whitelisted functions are allowed.")
        args = [_eval_node(arg) for arg in node.args]
        return _FUNCTIONS[node.func.id](*args)

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"Unknown identifier: {node.id}")

    raise ValueError(f"Disallowed expression element: {type(node).__name__}")

def calculate(expression: str):
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return result
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Error: {str(e)}"

CALCULATOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluates a mathematical expression safely. Supports +, -, *, /, //, %, **, and functions like sqrt(), log(), sin(), cos(), tan(), factorial(), abs(), round(), plus constants pi, e, tau.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. 'sqrt(81) + 15 * 0.2'.",
                }
            },
            "required": ["expression"],
        },
    },
}

def handle_tool_call(arguments: dict):
    if "expression" not in arguments:
        return "Error: Missing 'expression' argument."
    
    return calculate(arguments["expression"])