"""
Tool Error Handling
-------------------
Explores the different ways tools can fail and how to communicate those
failures to the model so it can recover, retry, or gracefully degrade.

Key concepts:
- Returning errors as strings vs. using is_error flag
- Actionable vs. silent errors
- Retry logic and idempotency
- Validation errors (bad arguments from the model)
- Timeout and rate limit patterns
- Graceful degradation when partial results are available

Three failure patterns demonstrated:
  1. Validation error  — model passes invalid arguments (wrong type, out of range)
  2. Resource error    — data not found, unavailable, or rate-limited
  3. Execution error   — unexpected exception during tool execution

The key insight: how you phrase the error determines whether the model
can recover. "Error: not found" → dead end. "Error: city 'Tokio' not
found. Did you mean 'Tokyo'?" → model retries with the corrected input.
"""

import random
import time
from typing import Any

# ---------------------------------------------------------------------------
# Simulated backend with configurable failure modes
# ---------------------------------------------------------------------------

class SimulatedBackend:
    """
    A fake data store with injectable failures.
    Used to demonstrate different error scenarios.
    """
    _call_counts: dict[str, int] = {}

    CITIES = {
        "tokyo": {"temp": 22, "condition": "sunny", "humidity": 65},
        "london": {"temp": 15, "condition": "overcast", "humidity": 80},
        "new york": {"temp": 18, "condition": "partly cloudy", "humidity": 70},
        "sydney": {"temp": 28, "condition": "clear", "humidity": 55},
        "paris": {"temp": 17, "condition": "light rain", "humidity": 85},
    }

    CONVERSIONS = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("USD", "JPY"): 149.50,
        ("EUR", "USD"): 1.09,
        ("GBP", "USD"): 1.27,
        ("JPY", "USD"): 0.0067,
    }

    RATE_LIMITED_AFTER = 3  # requests per tool before triggering a fake rate limit


backend = SimulatedBackend()


# ---------------------------------------------------------------------------
# Tool implementations with explicit error patterns
# ---------------------------------------------------------------------------

def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    Demonstrates:
    - Validation error (bad unit)
    - Resource error (unknown city) with a suggestion
    - Successful result
    """
    # Validate unit
    if unit not in ("celsius", "fahrenheit"):
        return {
            "error": f"Invalid unit '{unit}'. Must be 'celsius' or 'fahrenheit'.",
            "error_type": "validation",
            "hint": "Use 'celsius' for metric or 'fahrenheit' for imperial.",
        }

    city_key = city.lower().strip()
    data = backend.CITIES.get(city_key)

    if data is None:
        # Find close matches
        close = [c for c in backend.CITIES if city_key[:3] in c]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        return {
            "error": f"City '{city}' not found.{hint}",
            "error_type": "not_found",
            "available_cities": list(backend.CITIES.keys()),
        }

    temp = data["temp"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)

    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": data["condition"],
        "humidity_pct": data["humidity"],
    }


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Demonstrates:
    - Rate limiting (simulated)
    - Validation error (negative amount)
    - Unsupported currency pair with helpful list
    """
    # Track call count for rate limit simulation
    key = "convert_currency"
    backend._call_counts[key] = backend._call_counts.get(key, 0) + 1

    # Validate amount
    if amount <= 0:
        return {
            "error": f"Amount must be positive, got {amount}.",
            "error_type": "validation",
        }

    # Simulate rate limiting after N calls
    if backend._call_counts[key] > backend.RATE_LIMITED_AFTER:
        return {
            "error": "Rate limit exceeded. Please wait 60 seconds before retrying.",
            "error_type": "rate_limit",
            "retry_after_seconds": 60,
        }

    from_c = from_currency.upper()
    to_c = to_currency.upper()

    rate = backend.CONVERSIONS.get((from_c, to_c))
    if rate is None:
        supported = [f"{a}→{b}" for a, b in backend.CONVERSIONS.keys()]
        return {
            "error": f"Unsupported currency pair: {from_c} to {to_c}.",
            "error_type": "unsupported",
            "supported_pairs": supported,
        }

    converted = round(amount * rate, 2)
    return {
        "from_amount": amount,
        "from_currency": from_c,
        "to_amount": converted,
        "to_currency": to_c,
        "exchange_rate": rate,
    }


def divide_numbers(a: float, b: float) -> dict:
    """
    Demonstrates:
    - Execution error (division by zero) caught gracefully
    - Clear error message that helps the model understand the constraint
    """
    try:
        if b == 0:
            return {
                "error": "Cannot divide by zero. Please provide a non-zero divisor.",
                "error_type": "execution",
            }
        return {"result": a / b, "expression": f"{a} / {b}"}
    except Exception as e:
        return {
            "error": f"Unexpected error during calculation: {str(e)}",
            "error_type": "execution",
        }


def look_up_user(user_id: str) -> dict:
    """
    Demonstrates:
    - Validation (wrong ID format)
    - Permission error (some user IDs are restricted)
    - Successful lookup
    """
    # Validate ID format
    if not user_id.startswith("U") or not user_id[1:].isdigit():
        return {
            "error": f"Invalid user ID format: '{user_id}'. Expected format: 'U' followed by digits, e.g. 'U1042'.",
            "error_type": "validation",
        }

    user_num = int(user_id[1:])

    # Simulate restricted access for certain IDs
    if user_num < 1000:
        return {
            "error": f"Access denied: user ID {user_id} is in the restricted admin range (U0001–U0999).",
            "error_type": "permission",
            "hint": "Use an ID in the range U1000–U9999 for regular users.",
        }

    # Fake user data
    names = ["Alice Chen", "Bob Patel", "Carlos Rivera", "Diana Osei", "Ethan Park"]
    return {
        "user_id": user_id,
        "name": names[user_num % len(names)],
        "email": f"user{user_num}@example.com",
        "account_status": "active",
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Get current weather for a city. "
            "Supports 'celsius' (default) or 'fahrenheit' units. "
            "Available cities: tokyo, london, new york, sydney, paris. "
            "If the city is not found, the tool returns suggestions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "description": "Temperature unit: 'celsius' or 'fahrenheit'",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "convert_currency",
        "description": (
            "Convert an amount from one currency to another. "
            "Supported pairs: USD↔EUR, USD↔GBP, USD↔JPY. "
            "Amount must be positive. Returns the converted amount and exchange rate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert (must be > 0)"},
                "from_currency": {"type": "string", "description": "Source currency code, e.g. 'USD'"},
                "to_currency": {"type": "string", "description": "Target currency code, e.g. 'EUR'"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "divide_numbers",
        "description": "Divide two numbers. Returns an error if the divisor is zero.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Dividend"},
                "b": {"type": "number", "description": "Divisor (must not be zero)"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "look_up_user",
        "description": (
            "Look up a user by their ID. "
            "User IDs must be in the format 'U' followed by digits, e.g. 'U1042'. "
            "IDs in the range U0001–U0999 are admin-restricted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID in format 'U1234'"},
            },
            "required": ["user_id"],
        },
    },
]

TOOL_FN = {
    "get_weather": get_weather,
    "convert_currency": convert_currency,
    "divide_numbers": divide_numbers,
    "look_up_user": look_up_user,
}


def dispatch_tool(name: str, inputs: dict) -> tuple[str, bool]:
    """
    Returns (result_string, is_error).
    The is_error flag tells the API whether this is an error result.
    """
    fn = TOOL_FN.get(name)
    if fn is None:
        return f"Unknown tool: {name}", True

    try:
        result = fn(**inputs)
    except TypeError as e:
        return f"Invalid arguments for {name}: {e}", True
    except Exception as e:
        return f"Unexpected error in {name}: {e}", True

    if isinstance(result, dict):
        if "error" in result:
            # Format error nicely with hint if available
            msg = result["error"]
            if "hint" in result:
                msg += f" Hint: {result['hint']}"
            if "supported_pairs" in result:
                msg += f" Supported pairs: {', '.join(result['supported_pairs'][:6])}"
            if "available_cities" in result:
                msg += f" Available: {', '.join(result['available_cities'])}"
            return msg, True
        lines = []
        for k, v in result.items():
            if isinstance(v, list):
                lines.append(f"{k}: {', '.join(str(i) for i in v)}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines), False

    return str(result), False


def reset_rate_limits():
    """Reset simulated rate limit counters (useful for demo restarts)."""
    backend._call_counts.clear()
