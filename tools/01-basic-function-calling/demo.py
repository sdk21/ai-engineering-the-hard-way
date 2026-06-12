"""
Demo: Basic Function Calling
Usage:
    python demo.py --mock
    python demo.py --real

Try asking:
    "What's the weather in Tokyo?"
    "What's 2 to the power of 16?"
    "Is it warmer in Sydney or London?"
"""

import argparse
import os
import sys

from experiment import TOOLS, dispatch_tool


# ---------------------------------------------------------------------------
# Mock agentic loop — simulates model deciding to call a tool
# ---------------------------------------------------------------------------

def mock_agent(user_message: str) -> None:
    print(f"\nUser: {user_message}")
    msg = user_message.lower()

    # Decide which tool to call (simple keyword heuristic)
    if any(w in msg for w in ["weather", "temperature", "forecast"]):
        # Extract city name (very naively)
        words = msg.split()
        city = "Tokyo"  # default
        for w in ["in", "for", "at"]:
            if w in words:
                idx = words.index(w)
                if idx + 1 < len(words):
                    city = words[idx + 1].rstrip("?.,")
                    break
        print(f"  [Model → tool_use] get_weather(city='{city}')")
        result = dispatch_tool("get_weather", {"city": city})
        print(f"  [tool_result] {result}")
        print(f"Assistant: The weather in {city.title()} is currently {result}.")

    elif any(w in msg for w in ["calculate", "compute", "what is", "how much", "+"]):
        # Pull out a simple expression
        for token in user_message.split():
            if any(c in token for c in "0123456789"):
                expr = user_message.split("is")[-1].strip().rstrip("?")
                break
        else:
            expr = "6 * 7"
        print(f"  [Model → tool_use] calculate(expression='{expr}')")
        result = dispatch_tool("calculate", {"expression": expr})
        print(f"  [tool_result] {result}")
        print(f"Assistant: The result of {expr} is {result}.")

    else:
        print("Assistant: I can check weather or do calculations. Try one of those!")


# ---------------------------------------------------------------------------
# Real agentic loop
# ---------------------------------------------------------------------------

def real_agent(user_message: str) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": user_message}]

    print(f"\nUser: {user_message}")

    # Agentic loop — keep going until the model stops calling tools
    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            tools=TOOLS,
            messages=messages,
        )

        # Collect assistant turn (may contain tool_use blocks)
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            # Final text response
            for block in assistant_content:
                if hasattr(block, "text"):
                    print(f"Assistant: {block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    print(f"  [Model → tool_use] {block.name}({block.input})")
                    result = dispatch_tool(block.name, block.input)
                    print(f"  [tool_result] {result}")
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
    "What's the weather in London?",
    "What's 2 to the power of 10?",
    "Is it warmer in Sydney or Paris?",
]


def run_demo(use_mock: bool) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Basic Function Calling Demo [{mode}] ===")
    print("Available tools:", [t["name"] for t in TOOLS])
    print("Example queries:", EXAMPLE_QUERIES)
    print("Type a question or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input or user_input.lower() == "quit":
            break
        if use_mock:
            mock_agent(user_input)
        else:
            real_agent(user_input)
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
