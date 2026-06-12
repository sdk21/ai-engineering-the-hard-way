"""
Demo: Tool Chaining
Usage:
    python demo.py --mock
    python demo.py --real

Try asking:
    "I want to fly from NYC to LAX in business class for 2 people. What will it cost?"
    "Find me a flight from SFO to ORD and check first class availability."
    "What's the cheapest NYC to LAX flight with available economy seats?"
"""

import argparse
import json
import os
import sys

from experiment import TOOLS, dispatch_tool

# ---------------------------------------------------------------------------
# Mock: hard-coded multi-step chain to illustrate the pattern
# ---------------------------------------------------------------------------

def mock_chain(user_message: str) -> None:
    print(f"\nUser: {user_message}")
    msg = user_message.lower()

    # Detect intent
    if "nyc" in msg and "lax" in msg:
        origin, dest = "NYC", "LAX"
    elif "sfo" in msg and "ord" in msg:
        origin, dest = "SFO", "ORD"
    elif "lax" in msg and "mia" in msg:
        origin, dest = "LAX", "MIA"
    else:
        origin, dest = "NYC", "LAX"

    seat_class = "business" if "business" in msg else "economy"
    num_passengers = 2 if "2 people" in msg or "two" in msg else 1

    # Step 1 — look up flights
    print(f"\n  [Step 1] Model → look_up_flight(origin='{origin}', destination='{dest}')")
    result1 = dispatch_tool("look_up_flight", {"origin": origin, "destination": dest})
    print(f"  [tool_result]\n{_indent(result1)}")

    # Extract first flight ID from the result text
    flight_id = None
    for line in result1.splitlines():
        if "flight_id:" in line.lower():
            flight_id = line.split(":")[-1].strip()
            break
    if not flight_id:
        print("  [chain stopped — no flights found]")
        return

    # Step 2 — get flight details
    print(f"\n  [Step 2] Model → get_flight_details(flight_id='{flight_id}')")
    result2 = dispatch_tool("get_flight_details", {"flight_id": flight_id})
    print(f"  [tool_result]\n{_indent(result2)}")

    # Extract base price
    base_price = None
    for line in result2.splitlines():
        if "base_price:" in line.lower():
            base_price = int(line.split(":")[-1].strip())
            break

    # Verify seat class is available
    available_classes = []
    for line in result2.splitlines():
        if "seat_options:" in line.lower():
            available_classes = [s.strip() for s in line.split(":")[-1].strip().strip("[]'").split(",")]
            break
    if seat_class not in [c.strip().strip("'") for c in available_classes]:
        seat_class = "economy"

    # Step 3 — check availability
    print(f"\n  [Step 3] Model → check_seat_availability(flight_id='{flight_id}', seat_class='{seat_class}')")
    result3 = dispatch_tool("check_seat_availability", {"flight_id": flight_id, "seat_class": seat_class})
    print(f"  [tool_result]\n{_indent(result3)}")

    if "sold out" in result3:
        print(f"\nAssistant: Unfortunately {seat_class} class on {flight_id} is sold out.")
        return

    # Step 4 — calculate total cost
    if base_price:
        print(f"\n  [Step 4] Model → calculate_total_cost(base_price={base_price}, seat_class='{seat_class}', num_passengers={num_passengers})")
        result4 = dispatch_tool("calculate_total_cost", {"base_price": base_price, "seat_class": seat_class, "num_passengers": num_passengers})
        print(f"  [tool_result]\n{_indent(result4)}")

        # Extract total
        total = None
        for line in result4.splitlines():
            if "total_cost:" in line.lower():
                total = line.split(":")[-1].strip()
                break

        print(f"\nAssistant: Flight {flight_id} ({origin}→{dest}) has {seat_class} seats available.")
        if total:
            print(f"           Total cost for {num_passengers} passenger(s): ${total} USD.")
    else:
        print(f"\nAssistant: Flight {flight_id} found. {seat_class.title()} class is available.")


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# ---------------------------------------------------------------------------
# Real: full agentic loop
# ---------------------------------------------------------------------------

def real_chain(user_message: str) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system = (
        "You are a helpful flight booking assistant. "
        "When the user asks about flights, use the available tools to look up flights, "
        "get details, check availability, and calculate costs. "
        "Always work through the chain step by step: first find flights, then get details, "
        "then check availability for the requested seat class, then calculate the total cost. "
        "Be explicit about each step you're taking."
    )

    messages = [{"role": "user", "content": user_message}]
    print(f"\nUser: {user_message}")

    step = 0
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
                    print(f"\nAssistant: {block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    step += 1
                    print(f"\n  [Step {step}] Model → {block.name}({json.dumps(block.input)})")
                    result = dispatch_tool(block.name, block.input)
                    print(f"  [tool_result]\n{_indent(result)}")
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
    "I want to fly from NYC to LAX in business class for 2 people. What will it cost?",
    "Find me a flight from SFO to ORD and check first class availability.",
    "What's the cheapest NYC to LAX option with available economy seats?",
]


def run_demo(use_mock: bool) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Tool Chaining Demo [{mode}] ===")
    print("This demo shows multi-step tool chains where each call depends on the previous result.")
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
            mock_chain(user_input)
        else:
            real_chain(user_input)
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
