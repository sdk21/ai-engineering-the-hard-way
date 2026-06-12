"""
Demo: Tool Use with Memory
Usage:
    python demo.py --mock
    python demo.py --real

Suggested session:
    Turn 1: "What's the weather in Tokyo?"           (cache miss, fact stored)
    Turn 2: "What's the weather in Tokyo?"           (cache hit!)
    Turn 3: "Tell me about Tokyo."                   (uses stored fact)
    Turn 4: "What's AAPL stock price?"               (cache miss, fact stored)
    Turn 5: "Compare AAPL and MSFT stock prices."    (AAPL from cache, MSFT new)
    Turn 6: "memory"                                 (see stored facts)
    Turn 7: "stats"                                  (see cache hit rate)
"""

import argparse
import os
import sys

from experiment import TOOLS, TOOL_FN_MAP, MemoryAugmentedExecutor


def run_session(use_mock: bool) -> None:
    executor = MemoryAugmentedExecutor(TOOL_FN_MAP)
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Tool Use with Memory Demo [{mode}] ===")
    print("Tool results are cached and facts are extracted to memory.")
    print("Commands: 'memory' to see stored facts, 'stats' to see cache stats, 'quit' to exit.\n")
    print("Suggested session:")
    print("  1. What's the weather in Tokyo?")
    print("  2. What's the weather in Tokyo?  ← cache hit")
    print("  3. Tell me about Tokyo.          ← uses stored facts")
    print()

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "memory":
            ctx = executor.memory_context()
            print(ctx if ctx else "  [no facts stored yet]")
            print()
            continue
        if user_input.lower() == "stats":
            stats = executor.stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")
            print()
            continue

        if use_mock:
            _mock_turn(user_input, executor)
        else:
            _real_turn(user_input, messages, executor)
        print()


def _mock_turn(user_message: str, executor: MemoryAugmentedExecutor) -> None:
    print(f"\nUser: {user_message}")
    msg = user_message.lower()

    # Inject memory context
    memory_ctx = executor.memory_context()
    if memory_ctx:
        print(f"  [memory injected into context]")

    # Simple intent detection
    if "weather" in msg:
        for city in ["tokyo", "london", "paris", "sydney"]:
            if city in msg:
                result, hit = executor.execute("get_weather", {"city": city})
                status = "CACHE HIT" if hit else "API CALL"
                print(f"  [{status}] get_weather(city={city}) → {result}")
                print(f"Assistant: The weather in {city.title()} is: {result}")
                return
        print("Assistant: Which city's weather would you like?")

    elif any(t in msg.upper() for t in ["AAPL", "MSFT", "NVDA"]):
        for ticker in ["AAPL", "MSFT", "NVDA"]:
            if ticker in msg.upper():
                result, hit = executor.execute("get_stock_price", {"ticker": ticker})
                status = "CACHE HIT" if hit else "API CALL"
                print(f"  [{status}] get_stock_price(ticker={ticker}) → {result}")
                print(f"Assistant: {ticker} stock: {result}")
                return

    elif "about" in msg or "tell me" in msg:
        for topic in ["tokyo", "python", "london"]:
            if topic in msg:
                result, hit = executor.execute("wikipedia_lookup", {"topic": topic})
                status = "CACHE HIT" if hit else "API CALL"
                print(f"  [{status}] wikipedia_lookup(topic={topic}) → {result[:60]}...")
                print(f"Assistant: {result}")
                return

    else:
        print("Assistant: I can look up weather, stocks, or Wikipedia topics.")


def _real_turn(user_message: str, messages: list, executor: MemoryAugmentedExecutor) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Build system prompt with memory context
    memory_ctx = executor.memory_context()
    system = "You are a helpful assistant with tools for weather, stocks, and Wikipedia lookups."
    if memory_ctx:
        system += f"\n\n{memory_ctx}"

    messages.append({"role": "user", "content": user_message})
    print(f"\nUser: {user_message}")
    if memory_ctx:
        print(f"  [memory injected: {len(executor.facts.get_all_facts())} facts]")

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
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
                    result, hit = executor.execute(block.name, block.input)
                    status = "CACHE HIT" if hit else "API CALL"
                    print(f"  [{status}] {block.name}({block.input}) → {result[:60]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_session(use_mock=args.mock)
