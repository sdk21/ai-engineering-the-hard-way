"""
Demo: Sliding Window Memory
Usage:
    python demo.py --mock
    python demo.py --real
    python demo.py --mock --window 2   # tiny window to see forgetting quickly
"""

import argparse
import os
import sys

from experiment import SlidingWindowBuffer


def mock_chat(messages: list[dict], system: str) -> str:
    last_user = messages[-1]["content"] if messages else ""
    # Check if name appears in current window
    for m in messages:
        if "name is" in m["content"].lower():
            name = m["content"].lower().split("name is")[-1].strip().split()[0].rstrip(".,!")
            if "name" in last_user.lower():
                return f"Your name is {name.capitalize()} (it's still in my window)."
    # Check if name appears in dropped summary (in system prompt)
    if "name is" in system.lower() and "name" in last_user.lower():
        return "I can see from the earlier conversation summary that you mentioned your name."
    if "name" in last_user.lower():
        return "I don't have your name in my current context window — it was dropped!"
    return f"[Mock] I see {len(messages)} message(s) in my current window."


def real_chat(messages: list[dict], system: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def run_demo(use_mock: bool, window_size: int) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Sliding Window Demo [{mode}] | window={window_size} messages ===")
    print("Try: tell it your name, chat a few turns, then ask 'what's my name?'\n")

    buffer = SlidingWindowBuffer(window_size=window_size)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            print(f"[Window: {len(buffer)}/{window_size} msgs | Dropped: {buffer.dropped_count}]\n")
            continue

        buffer.add_user_message(user_input)
        system = buffer.get_effective_system()

        if use_mock:
            reply = mock_chat(buffer.get_messages(), system)
        else:
            reply = real_chat(buffer.get_messages(), system)

        buffer.add_assistant_message(reply)
        print(f"Assistant: {reply}")
        print(f"  [window={len(buffer)}/{window_size}, dropped={buffer.dropped_count}]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    parser.add_argument("--window", type=int, default=4, help="Max messages in context")
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock, window_size=args.window)
