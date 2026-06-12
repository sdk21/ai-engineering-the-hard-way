"""
Demo: Tool Error Handling
Usage:
    python demo.py --mock
    python demo.py --real

Try asking:
    "What's the weather in Tokio?"          (misspelled city — model retries)
    "Convert -50 USD to EUR."               (negative amount — validation error)
    "What's 100 divided by 0?"             (execution error)
    "Look up user U0042."                   (permission error)
    "Look up user ABC."                     (format validation error)
"""

import argparse
import os
import sys

from experiment import TOOLS, dispatch_tool, reset_rate_limits

# ---------------------------------------------------------------------------
# Mock: scripted scenarios showing each error type
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "label": "Validation error (bad unit)",
        "tool": "get_weather",
        "inputs": {"city": "Tokyo", "unit": "kelvin"},
    },
    {
        "label": "Resource error with suggestion (typo in city)",
        "tool": "get_weather",
        "inputs": {"city": "Tokio"},
    },
    {
        "label": "Successful call after correction",
        "tool": "get_weather",
        "inputs": {"city": "Tokyo"},
    },
    {
        "label": "Validation error (negative amount)",
        "tool": "convert_currency",
        "inputs": {"amount": -50, "from_currency": "USD", "to_currency": "EUR"},
    },
    {
        "label": "Unsupported currency pair",
        "tool": "convert_currency",
        "inputs": {"amount": 100, "from_currency": "USD", "to_currency": "CNY"},
    },
    {
        "label": "Execution error (divide by zero)",
        "tool": "divide_numbers",
        "inputs": {"a": 100, "b": 0},
    },
    {
        "label": "Validation error (wrong ID format)",
        "tool": "look_up_user",
        "inputs": {"user_id": "ABC"},
    },
    {
        "label": "Permission error (restricted ID range)",
        "tool": "look_up_user",
        "inputs": {"user_id": "U0042"},
    },
    {
        "label": "Successful user lookup",
        "tool": "look_up_user",
        "inputs": {"user_id": "U1042"},
    },
]


def mock_error_demo() -> None:
    print("\n=== Tool Error Handling Demo [MOCK] ===")
    print("Running scripted scenarios to show each error type:\n")

    for scenario in SCENARIOS:
        label = scenario["label"]
        tool = scenario["tool"]
        inputs = scenario["inputs"]

        print(f"  Scenario: {label}")
        print(f"  → {tool}({inputs})")
        result, is_error = dispatch_tool(tool, inputs)
        status = "[ERROR]" if is_error else "[OK]"
        print(f"  {status} {result}")
        print()


# ---------------------------------------------------------------------------
# Real: interactive session with error recovery
# ---------------------------------------------------------------------------

def real_session(user_message: str) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system = (
        "You are a helpful assistant with access to weather, currency conversion, "
        "math, and user lookup tools. "
        "When a tool returns an error, read the error message carefully. "
        "If the error includes a hint or suggestion, try again with the corrected input. "
        "If the error is a permission or hard limit, explain this to the user clearly."
    )

    messages = [{"role": "user", "content": user_message}]
    print(f"\nUser: {user_message}")

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            for block in assistant_content:
                if hasattr(block, "text"):
                    print(f"Assistant: {block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    print(f"  [tool_use] {block.name}({block.input})")
                    result, is_error = dispatch_tool(block.name, block.input)
                    status = "ERROR" if is_error else "OK"
                    print(f"  [{status}] {result[:100]}{'...' if len(result) > 100 else ''}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        **({"is_error": True} if is_error else {}),
                    })
            messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(use_mock: bool) -> None:
    reset_rate_limits()

    if use_mock:
        mock_error_demo()
        return

    mode = "REAL (Claude)"
    print(f"\n=== Tool Error Handling Demo [{mode}] ===")
    print("This demo shows how the model recovers from tool errors.")
    print("\nTry these queries:")
    queries = [
        '"What\'s the weather in Tokio?"        (typo → model retries with correction)',
        '"Convert -50 USD to EUR."              (validation error)',
        '"What\'s 100 divided by 0?"            (execution error)',
        '"Look up user U0042."                  (permission error)',
        '"Look up user ABC."                    (format error → model corrects)',
    ]
    for q in queries:
        print(f"  {q}")
    print("\nType a question or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input or user_input.lower() == "quit":
            break
        real_session(user_input)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock)
