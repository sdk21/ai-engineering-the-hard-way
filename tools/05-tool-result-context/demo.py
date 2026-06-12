"""
Demo: Tool Result Context
Usage:
    python demo.py --mock
    python demo.py --real [--style bare|simple|rich]

Try asking:
    "What's the population of Tokyo?"
    "What's the population of Springfield?"
    "Tell me about GME stock."
    "How do you say hello in Japanese?"
    "How do you say hello in Arabic?"
"""

import argparse
import os
import sys

from experiment import TOOLS, dispatch_tool, set_result_style, ResultStyle


# ---------------------------------------------------------------------------
# Mock: show the same query answered with different result styles
# ---------------------------------------------------------------------------

def mock_style_comparison() -> None:
    print("\n=== Tool Result Context Demo [MOCK] ===")
    print("Comparing how result context affects the answer quality.\n")

    test_cases = [
        ("get_population", {"city": "Springfield"}),
        ("get_stock_info", {"ticker": "GME"}),
        ("translate_text", {"text": "hello", "target_language": "arabic"}),
    ]

    for tool_name, inputs in test_cases:
        print(f"{'='*60}")
        print(f"Tool: {tool_name}({inputs})\n")

        for style in [ResultStyle.BARE, ResultStyle.SIMPLE, ResultStyle.RICH]:
            result = dispatch_tool(tool_name, inputs, style=style)
            print(f"  [Style: {style.upper()}]")
            for line in result.splitlines():
                print(f"    {line}")
            print()

        print("  → Notice how the rich result gives the model much more to work with.")
        print()


# ---------------------------------------------------------------------------
# Real: interactive with configurable result style
# ---------------------------------------------------------------------------

def real_session(user_message: str) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system = (
        "You are a knowledgeable assistant. When you use tools, pay attention to ALL fields "
        "in the result — including confidence, source, freshness, caveats, and suggested_followups. "
        "Incorporate relevant caveats and confidence into your response. "
        "If confidence is low, say so. If there are important caveats, surface them. "
        "Don't just report the bare value — give the user the full picture."
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
                    result = dispatch_tool(block.name, block.input)
                    print("  [tool_result (rich context)]")
                    for line in result.splitlines():
                        print(f"    {line}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "What's the population of Tokyo?",
    "What's the population of Springfield?",
    "Tell me about GME stock.",
    "Tell me about NVDA stock.",
    "How do you say hello in Japanese?",
    "How do you say hello in Arabic?",
]


def run_demo(use_mock: bool, style: str = ResultStyle.RICH) -> None:
    set_result_style(style)

    if use_mock:
        mock_style_comparison()
        return

    mode = "REAL (Claude)"
    print(f"\n=== Tool Result Context Demo [{mode}] (style={style}) ===")
    print("This demo shows how structured context in tool results improves answer quality.")
    print("\nExample queries:")
    for q in EXAMPLE_QUERIES:
        print(f"  • {q}")
    print(f"\nCurrent result style: {style.upper()}")
    print("Run with --style bare or --style simple to compare.\n")

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
    parser.add_argument(
        "--style",
        choices=["bare", "simple", "rich"],
        default="rich",
        help="Result context style (real mode only)",
    )
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock, style=args.style)
