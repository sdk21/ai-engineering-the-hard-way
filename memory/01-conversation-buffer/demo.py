"""
Demo: Conversation Buffer Memory
Usage:
    python demo.py --mock    # deterministic fake responses
    python demo.py --real    # calls Claude via Anthropic API
"""

import argparse
import os
import sys

from experiment import ConversationBuffer


# ---------------------------------------------------------------------------
# Mock LLM — returns canned responses so you can learn the flow without an API key
# ---------------------------------------------------------------------------

MOCK_RESPONSES = [
    "Nice to meet you! I'll remember that.",
    "You told me your name earlier in our conversation.",
    "Of course! I have our full conversation history, so I can refer back to anything you've shared.",
    "That's a great question. Based on what you've told me, I can answer using our conversation context.",
    "I'm a mock LLM — I don't actually understand anything, but the buffer is working!",
]

_mock_turn = 0


def mock_chat(messages: list[dict]) -> str:
    global _mock_turn
    response = MOCK_RESPONSES[_mock_turn % len(MOCK_RESPONSES)]
    _mock_turn += 1
    # Simulate recalling user name if they mentioned it
    for m in messages:
        if m["role"] == "user" and "name is" in m["content"].lower():
            name = m["content"].lower().split("name is")[-1].strip().split()[0].rstrip(".,!")
            if "name" in messages[-1]["content"].lower():
                return f"Your name is {name.capitalize()}! I remembered from earlier in our conversation."
    return response


# ---------------------------------------------------------------------------
# Real LLM via Anthropic SDK
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Demo loop
# ---------------------------------------------------------------------------

def run_demo(use_mock: bool) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Conversation Buffer Demo [{mode}] ===")
    print("Type your messages. The full history is replayed every turn.")
    print("Commands: 'stats' to see memory size, 'clear' to reset, 'quit' to exit.\n")

    buffer = ConversationBuffer(system_prompt="You are a helpful assistant with a great memory.")

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
        if user_input.lower() == "clear":
            buffer.clear()
            print("[Memory cleared]\n")
            continue
        if user_input.lower() == "stats":
            print(f"[Turns: {len(buffer)}, ~{buffer.token_estimate()} tokens in context]\n")
            continue

        buffer.add_user_message(user_input)

        if use_mock:
            reply = mock_chat(buffer.get_messages())
        else:
            reply = real_chat(buffer.get_messages(), buffer.system_prompt)

        buffer.add_assistant_message(reply)
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="Use mock LLM (no API key needed)")
    group.add_argument("--real", action="store_true", help="Use real Claude API")
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock)
