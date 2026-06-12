"""
Demo: Tool Composition
Usage:
    python demo.py --mock
    python demo.py --real
"""

import argparse
import os
import sys

from experiment import TOOLS, dispatch_tool


def mock_composition() -> None:
    print("\n=== Tool Composition Demo [MOCK] ===")
    print("Showing composite tools that hide multi-step logic.\n")

    cases = [
        ("Pipeline: get_weather → unit conversion",
         "get_weather_in_unit", {"city": "Tokyo", "target_unit": "fahrenheit"}),
        ("Aggregate: weather comparison",
         "compare_weather", {"cities": ["Tokyo", "London", "Paris"]}),
        ("Conditional: weather advisory",
         "weather_advisory", {"city": "London"}),
        ("Conditional: weather advisory (hot city)",
         "weather_advisory", {"city": "Tokyo"}),
    ]

    for label, tool_name, inputs in cases:
        print(f"  [{label}]")
        print(f"  → {tool_name}({inputs})")
        result = dispatch_tool(tool_name, inputs)
        for line in result.splitlines():
            print(f"    {line}")
        print()


def real_session(user_message: str) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": user_message}]
    print(f"\nUser: {user_message}")

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"Assistant: {block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    print(f"  [{block.name}] → {result[:80]}")
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            messages.append({"role": "user", "content": tool_results})


def run_demo(use_mock: bool) -> None:
    if use_mock:
        mock_composition()
        return

    print(f"\n=== Tool Composition Demo [REAL] ===")
    print("Composite tools combine multiple operations behind a single interface.\n")
    for q in ["What's the weather in Tokyo in Fahrenheit?",
              "Compare the weather in Tokyo, London, and Paris.",
              "What should I wear in London today?"]:
        print(f"  • {q}")
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
