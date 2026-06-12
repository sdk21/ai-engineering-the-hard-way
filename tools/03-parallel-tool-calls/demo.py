"""
Demo: Parallel Tool Calls
Usage:
    python demo.py --mock
    python demo.py --real

Try asking:
    "Give me a full report on AAPL."
    "Compare AAPL and MSFT — prices, ratings, and recent news."
    "What are the analyst ratings and latest headlines for NVDA and GOOGL?"
"""

import argparse
import json
import os
import sys
import time

from experiment import TOOLS, dispatch_tool, dispatch_tool_parallel

# ---------------------------------------------------------------------------
# Mock: simulate the model issuing parallel tool_use blocks
# ---------------------------------------------------------------------------

def mock_parallel(user_message: str) -> None:
    print(f"\nUser: {user_message}")
    msg = user_message.lower()

    # Detect tickers mentioned
    all_tickers = ["aapl", "msft", "googl", "amzn", "nvda"]
    tickers = [t.upper() for t in all_tickers if t in msg]
    if not tickers:
        tickers = ["AAPL"]  # default

    # Decide which tools to call based on query
    tools_to_call = []
    if any(w in msg for w in ["price", "stock", "trading", "report", "compare"]):
        for t in tickers:
            tools_to_call.append(("get_stock_price", {"ticker": t}))
    if any(w in msg for w in ["company", "info", "about", "sector", "employees", "report"]):
        for t in tickers:
            tools_to_call.append(("get_company_info", {"ticker": t}))
    if any(w in msg for w in ["analyst", "rating", "buy", "hold", "sell", "target", "report"]):
        for t in tickers:
            tools_to_call.append(("get_analyst_rating", {"ticker": t}))
    if any(w in msg for w in ["news", "headline", "latest", "recent", "report"]):
        for t in tickers:
            tools_to_call.append(("get_news_headlines", {"ticker": t}))

    if not tools_to_call:
        for t in tickers:
            tools_to_call.append(("get_stock_price", {"ticker": t}))

    # Show how the model would emit these as one batch
    print(f"\n  [Model → {len(tools_to_call)} parallel tool_use blocks]")
    for name, inputs in tools_to_call:
        print(f"    {name}({inputs})")

    # Execute in parallel (simulating concurrent execution)
    start = time.time()
    results = dispatch_tool_parallel(tools_to_call)
    elapsed = time.time() - start

    print(f"\n  [All {len(tools_to_call)} results received in {elapsed:.3f}s (executed concurrently)]")
    for (name, inputs), result in zip(tools_to_call, results):
        print(f"\n  [tool_result: {name}({list(inputs.values())[0]})]")
        for line in result.splitlines():
            print(f"    {line}")

    # Synthesize a summary response
    tickers_str = " and ".join(tickers)
    print(f"\nAssistant: Here's what I found for {tickers_str} (retrieved in parallel):")
    for (name, inputs), result in zip(tools_to_call, results):
        ticker = list(inputs.values())[0]
        first_line = result.splitlines()[0] if result.splitlines() else result
        print(f"  [{ticker} / {name}] {first_line}")


# ---------------------------------------------------------------------------
# Real: full agentic loop — model issues parallel blocks natively
# ---------------------------------------------------------------------------

def real_parallel(user_message: str) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system = (
        "You are a financial research assistant. "
        "When asked about stocks, use all relevant tools to gather complete information. "
        "When multiple tools can be called independently, call them all in a single response — "
        "do not wait for one result before requesting the next if they are independent. "
        "Synthesize all results into a clear, well-structured report."
    )

    messages = [{"role": "user", "content": user_message}]
    print(f"\nUser: {user_message}")

    round_num = 0
    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            for block in assistant_content:
                if hasattr(block, "text"):
                    print(f"\nAssistant: {block.text}")
            break

        if response.stop_reason == "tool_use":
            round_num += 1
            tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]
            print(f"\n  [Round {round_num}: model issued {len(tool_use_blocks)} tool call(s) in parallel]")

            # Execute all tool calls from this round concurrently
            tool_calls = [(b.name, b.input) for b in tool_use_blocks]
            start = time.time()
            results = dispatch_tool_parallel(tool_calls)
            elapsed = time.time() - start

            print(f"  [Executed in {elapsed:.3f}s]")

            tool_results = []
            for block, result in zip(tool_use_blocks, results):
                print(f"  [{block.name}({json.dumps(block.input)})] → {result.splitlines()[0] if result.splitlines() else result}")
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
    "Give me a full report on AAPL.",
    "Compare AAPL and MSFT — prices, analyst ratings, and recent news.",
    "What's the stock price and analyst consensus for NVDA and GOOGL?",
]


def run_demo(use_mock: bool) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Parallel Tool Calls Demo [{mode}] ===")
    print("This demo shows the model issuing multiple independent tool calls in a single response.")
    print("\nExample queries:")
    for q in EXAMPLE_QUERIES:
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
            mock_parallel(user_input)
        else:
            real_parallel(user_input)
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
