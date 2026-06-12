"""
Demo: Multi-Tool Agent
Usage:
    python demo.py --mock
    python demo.py --real

Try asking complex multi-step questions:
    "Give me a briefing on AcmeCorp: team size, finances, and current weather at HQ."
    "What's the capital of Japan, its weather, and calculate 37 * 42?"
    "Search our knowledge base for financial info and calculate our annual burn rate."
"""

import argparse
import os
import sys

from experiment import TOOLS, TOOL_FN, dispatch_tool, SYSTEM_PROMPT, _SCRATCHPAD


def mock_agent(user_message: str) -> None:
    print(f"\nUser: {user_message}")
    print("  [mock mode: showing tool call plan for this query]\n")
    msg = user_message.lower()

    steps = []
    if "acmecorp" in msg or "company" in msg or "team" in msg or "financ" in msg:
        steps.append(("scratchpad_write", {"note": "Plan: 1. Search KB for company info, 2. Get SF weather, 3. Synthesize"}))
        steps.append(("search_knowledge_base", {"query": "company overview team financial"}))
        if "weather" in msg or "hq" in msg:
            steps.append(("get_weather", {"city": "san francisco"}))
    elif "capital" in msg and "japan" in msg:
        steps.append(("wikipedia_search", {"topic": "tokyo"}))
        steps.append(("get_weather", {"city": "tokyo"}))
    if "calculat" in msg or "*" in msg or "burn" in msg:
        steps.append(("calculate", {"expression": "800000 * 12"}))

    if not steps:
        print("  Assistant: I can help with company info, calculations, weather, and more!")
        return

    for i, (tool_name, inputs) in enumerate(steps, 1):
        print(f"  [Step {i}] {tool_name}({inputs})")
        result = dispatch_tool(tool_name, inputs)
        print(f"  [Result] {result[:100]}{'...' if len(result) > 100 else ''}")

    print(f"\n  [Agent synthesizes {len(steps)} tool results into final answer]")
    print("  Assistant: [Would combine all results into a structured response]")


def real_agent(user_message: str) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": user_message}]
    print(f"\nUser: {user_message}")

    step_count = 0
    max_steps = 15  # safety limit

    while step_count < max_steps:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAssistant: {block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    step_count += 1
                    result = dispatch_tool(block.name, block.input)
                    print(f"  [{step_count}] {block.name}({block.input}) → {result[:60]}{'...' if len(result)>60 else ''}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})


def run_demo(use_mock: bool) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Multi-Tool Agent Demo [{mode}] ===")
    print("A full agent with 9 tools, scratchpad planning, and synthesis.\n")
    print("Example queries:")
    for q in [
        "Give me a briefing on AcmeCorp: team size, finances, and weather at HQ.",
        "What's the capital of Japan, its current weather, and calculate 37 * 42?",
        "Search our knowledge base for financial info and calculate annual burn rate.",
    ]:
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
