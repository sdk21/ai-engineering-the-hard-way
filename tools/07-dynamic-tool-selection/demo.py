"""
Demo: Dynamic Tool Selection
Usage:
    python demo.py --mock
    python demo.py --real [--top-k N]

Try asking:
    "What's the weather forecast for Paris this week?"
    "Search for Python packages for data visualization on GitHub."
    "What do I need to know about traveling to Japan?"
    "What's the stock price for NVDA and what do analysts think?"
"""

import argparse
import os
import sys

from experiment import (
    REGISTRY, SEARCH_TOOLS_SCHEMA, search_tools_fn,
)


# ---------------------------------------------------------------------------
# Mock: show which tools are selected for different queries
# ---------------------------------------------------------------------------

MOCK_QUERIES = [
    "What's the weather forecast for Paris this week?",
    "Search for Python packages for data visualization on GitHub.",
    "What do I need to know about traveling to Japan?",
    "What's the stock price for NVDA and what do analysts think?",
    "Translate 'hello' to French and calculate 2^10.",
]


def mock_selection_demo(top_k: int = 5) -> None:
    print(f"\n=== Dynamic Tool Selection Demo [MOCK] ===")
    print(f"Registry: {len(REGISTRY.all_tools)} tools across 6 categories")
    print(f"Selection strategy: top-{top_k} by TF-IDF similarity\n")

    for query in MOCK_QUERIES:
        print(f"Query: \"{query}\"")
        selected = REGISTRY.search(query, top_k=top_k)
        print(f"Selected {len(selected)}/{len(REGISTRY.all_tools)} tools:")
        for tool in selected:
            print(f"  [{tool.category}] {tool.name}: {tool.description[:60]}...")
        print()


# ---------------------------------------------------------------------------
# Real: two-stage retrieval → execution
# ---------------------------------------------------------------------------

def real_session(user_message: str, top_k: int = 5) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Stage 1: select relevant tools
    selected = REGISTRY.search(user_message, top_k=top_k)
    selected_schemas = REGISTRY.to_api_schemas(selected)

    # Always include the meta search_tools tool
    all_schemas = selected_schemas + [SEARCH_TOOLS_SCHEMA]

    print(f"\nUser: {user_message}")
    print(f"  [registry] Selected {len(selected)} tools: {[t.name for t in selected]}")

    system = (
        "You are a helpful assistant with access to a large tool catalogue. "
        "A relevant subset of tools has been pre-selected for your query. "
        "If you need a tool that's not in your current set, use 'search_tools' to find it. "
        "Use the tools that best match the user's request."
    )

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            tools=all_schemas,
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

                    # Handle meta search_tools specially
                    if block.name == "search_tools":
                        result = search_tools_fn(**block.input)
                        # Add newly discovered tools to the schema set
                        discovered = REGISTRY.search(block.input.get("query", ""), top_k=3)
                        for tool in discovered:
                            schema = REGISTRY.to_api_schemas([tool])[0]
                            if not any(s["name"] == schema["name"] for s in all_schemas):
                                all_schemas.append(schema)
                                print(f"  [registry] Added tool: {tool.name}")
                    else:
                        result = REGISTRY.dispatch(block.name, block.input)

                    print(f"  [tool_result] {result[:80]}{'...' if len(result) > 80 else ''}")
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
    "What's the weather forecast for Paris this week?",
    "Search for Python packages for data visualization on GitHub.",
    "What do I need to know about traveling to Japan?",
    "What's the stock price for NVDA and what do analysts think?",
    "Translate 'good morning' to Japanese and check the current weather in Tokyo.",
]


def run_demo(use_mock: bool, top_k: int = 5) -> None:
    if use_mock:
        mock_selection_demo(top_k=top_k)
        return

    mode = "REAL (Claude)"
    print(f"\n=== Dynamic Tool Selection Demo [{mode}] (top-{top_k}) ===")
    print(f"Tool catalogue: {len(REGISTRY.all_tools)} tools")
    print("Each query selects the most relevant tools dynamically.\n")
    print("Example queries:")
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
        real_session(user_input, top_k=top_k)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    parser.add_argument("--top-k", type=int, default=5, help="Number of tools to select per query")
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock, top_k=args.top_k)
