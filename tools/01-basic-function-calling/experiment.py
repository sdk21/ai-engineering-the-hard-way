"""
Basic Function Calling
----------------------
Defines tools as JSON schemas, implements an agentic loop that:
  1. Sends user message + tool definitions to the model
  2. Executes any tool calls the model requests
  3. Returns tool results and loops until the model gives a final answer
"""

import ast
import operator as op


# ---------------------------------------------------------------------------
# Tool implementations (the actual functions we'll let the model call)
# ---------------------------------------------------------------------------

FAKE_WEATHER = {
    "tokyo": "22°C, sunny",
    "london": "15°C, overcast",
    "new york": "18°C, partly cloudy",
    "sydney": "28°C, clear",
    "paris": "17°C, light rain",
}


def get_weather(city: str) -> str:
    return FAKE_WEATHER.get(city.lower(), f"No weather data for '{city}'.")


_SAFE_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.n
    if isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported operation: {type(node)}")


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool registry — maps names to callables and JSON schemas
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name."},
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a simple arithmetic expression and return the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A math expression, e.g. '2 ** 10'"},
            },
            "required": ["expression"],
        },
    },
]

TOOL_FN = {
    "get_weather": get_weather,
    "calculate": calculate,
}


def dispatch_tool(name: str, inputs: dict) -> str:
    """Call the named tool with the given inputs."""
    fn = TOOL_FN.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    return fn(**inputs)
