"""
Tool Composition
-----------------
Tools can be built from other tools: higher-order tools that call
lower-order tools, combining their results into a single interface.

Key concepts:
- Composite tools: a tool whose implementation calls other tools
- Tool pipelines: chain tools into a processing pipeline
- Tool adapters: wrap an existing tool with validation, transformation, or caching
- Meta-tools: tools that configure or control other tools
- Hiding complexity: a single "research" tool that internally calls
  Wikipedia, a calculator, and a translator

Three composition patterns:
  1. Pipeline: A → B → C (output of A feeds B feeds C)
  2. Aggregate: call A, B, C in parallel; merge results
  3. Conditional: call A; if result meets condition, call B, else C

Benefits of composition:
  - Reduces the number of tools the model needs to reason about
  - Encapsulates complex multi-step operations behind a clean interface
  - Allows reuse of tool logic across different composite tools
  - Simplifies the model's decision space for complex workflows
"""

from typing import Any, Callable


# ---------------------------------------------------------------------------
# Base tools (atomic)
# ---------------------------------------------------------------------------

def _get_weather_raw(city: str) -> dict:
    data = {
        "tokyo": {"temp_c": 22, "condition": "sunny", "humidity": 65},
        "london": {"temp_c": 15, "condition": "overcast", "humidity": 80},
        "paris": {"temp_c": 17, "condition": "light rain", "humidity": 85},
    }
    return data.get(city.lower(), {"error": f"No data for '{city}'"})


def _translate_raw(text: str, target_lang: str) -> str:
    translations = {
        ("hello", "japanese"): "こんにちは",
        ("hello", "french"): "Bonjour",
        ("goodbye", "japanese"): "さようなら",
        ("goodbye", "french"): "Au revoir",
    }
    return translations.get((text.lower(), target_lang.lower()), f"[{text} in {target_lang}]")


def _calculate_raw(expression: str) -> Any:
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}
    def ev(n):
        if isinstance(n, ast.Constant): return n.n
        if isinstance(n, ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(f"Unsupported: {n}")
    return ev(ast.parse(expression, mode="eval").body)


# ---------------------------------------------------------------------------
# Composition Pattern 1: Pipeline
# ---------------------------------------------------------------------------

def temperature_converter_pipeline(city: str, target_unit: str = "fahrenheit") -> str:
    """
    Composite tool: get_weather → unit_conversion pipeline.
    Internally calls get_weather, extracts temperature, converts units.
    The model just calls one tool with two arguments.
    """
    # Step 1: get raw weather
    weather = _get_weather_raw(city)
    if "error" in weather:
        return f"Error: {weather['error']}"

    temp_c = weather["temp_c"]

    # Step 2: convert
    if target_unit.lower() == "fahrenheit":
        converted = round(temp_c * 9/5 + 32, 1)
        unit_symbol = "°F"
    elif target_unit.lower() == "kelvin":
        converted = round(temp_c + 273.15, 2)
        unit_symbol = "K"
    else:
        converted = temp_c
        unit_symbol = "°C"

    return (
        f"{city.title()}: {converted}{unit_symbol}, "
        f"{weather['condition']}, humidity {weather['humidity']}%"
    )


# ---------------------------------------------------------------------------
# Composition Pattern 2: Aggregate
# ---------------------------------------------------------------------------

def weather_comparison(cities: list[str]) -> str:
    """
    Composite tool: calls get_weather for all cities and compares.
    Returns a formatted comparison without requiring multiple tool calls.
    """
    results = []
    for city in cities:
        data = _get_weather_raw(city)
        if "error" not in data:
            results.append((city.title(), data["temp_c"], data["condition"]))

    if not results:
        return "No weather data found for any of the specified cities."

    results.sort(key=lambda x: x[1], reverse=True)  # sort by temperature

    lines = [f"Weather comparison ({len(results)} cities):"]
    for city, temp, condition in results:
        lines.append(f"  {city}: {temp}°C, {condition}")
    lines.append(f"Warmest: {results[0][0]} ({results[0][1]}°C)")
    lines.append(f"Coolest: {results[-1][0]} ({results[-1][1]}°C)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Composition Pattern 3: Conditional branching
# ---------------------------------------------------------------------------

def smart_weather_advisory(city: str) -> str:
    """
    Composite tool: gets weather, then branches on condition to give advice.
    The model doesn't need to know about the branching logic.
    """
    data = _get_weather_raw(city)
    if "error" in data:
        return f"Error: {data['error']}"

    temp = data["temp_c"]
    condition = data["condition"].lower()

    # Conditional advice based on results
    advice = []

    if "rain" in condition:
        advice.append("Bring an umbrella.")
    elif "sunny" in condition and temp > 25:
        advice.append("Apply sunscreen and stay hydrated.")
    elif "overcast" in condition:
        advice.append("Light jacket recommended.")

    if temp < 10:
        advice.append("Dress warmly.")
    elif temp > 30:
        advice.append("Wear light clothing.")

    advice_str = " ".join(advice) if advice else "No special advisory."
    return f"{city.title()}: {temp}°C, {condition}. Advisory: {advice_str}"


# ---------------------------------------------------------------------------
# Composition Pattern 4: Tool adapter (wraps with transformation)
# ---------------------------------------------------------------------------

def _make_validated_calculator(
    max_value: float = 1e15,
    description: str = "Evaluate a math expression with bounds checking."
) -> dict:
    """
    Returns a tool schema + implementation for a calculator with validation.
    This is a factory pattern: the same base calculator, different configurations.
    """
    def validated_calc(expression: str) -> str:
        try:
            result = _calculate_raw(expression)
            if abs(result) > max_value:
                return f"Result {result} exceeds maximum value of {max_value:.0e}"
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return {
        "schema": {
            "name": "calculate",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        "fn": validated_calc,
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_weather_in_unit",
        "description": (
            "Get weather for a city and convert temperature to the specified unit. "
            "Handles both fetching and conversion in one call. "
            "Units: 'celsius' (default), 'fahrenheit', 'kelvin'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "target_unit": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"]},
            },
            "required": ["city"],
        },
    },
    {
        "name": "compare_weather",
        "description": (
            "Compare weather across multiple cities at once. "
            "Returns temperatures sorted from warmest to coolest. "
            "More efficient than calling get_weather for each city individually."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of city names to compare",
                }
            },
            "required": ["cities"],
        },
    },
    {
        "name": "weather_advisory",
        "description": (
            "Get weather conditions AND safety/clothing advisory for a city in one call. "
            "Combines weather lookup with context-aware recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

TOOL_FN = {
    "get_weather_in_unit": temperature_converter_pipeline,
    "compare_weather": lambda cities: weather_comparison(cities),
    "weather_advisory": smart_weather_advisory,
    "calculate": lambda expression: str(_calculate_raw(expression)),
}


def dispatch_tool(name: str, inputs: dict) -> str:
    fn = TOOL_FN.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        return fn(**inputs)
    except Exception as e:
        return f"Error: {e}"
