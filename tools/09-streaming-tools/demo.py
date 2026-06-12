"""
Demo: Streaming with Tools
Usage:
    python demo.py --mock
    python demo.py --real

Try asking:
    "What's the weather in Tokyo?"
    "Calculate 2 to the power of 20."
    "Is it warmer in Tokyo or Paris?"
"""

import argparse
import os
import sys
import time

from experiment import (
    TOOLS, dispatch_tool, StreamAccumulator,
    _make_text_stream, ContentBlock,
)


# ---------------------------------------------------------------------------
# Mock: process simulated streaming events with visible rendering
# ---------------------------------------------------------------------------

def mock_streaming(user_message: str) -> None:
    print(f"\nUser: {user_message}")
    msg = user_message.lower()

    # Decide what to simulate
    if "weather" in msg and "tokyo" in msg and "paris" in msg:
        events = _make_text_stream(
            "Let me check both cities for you.",
            tool_calls=[
                {"id": "t1", "name": "get_weather", "input": {"city": "tokyo"}},
                {"id": "t2", "name": "get_weather", "input": {"city": "paris"}},
            ],
        )
    elif "weather" in msg:
        city = "Tokyo"
        for c in ["tokyo", "paris", "london"]:
            if c in msg:
                city = c.title()
        events = _make_text_stream(
            "",
            tool_calls=[{"id": "t1", "name": "get_weather", "input": {"city": city}}],
        )
    elif any(w in msg for w in ["calculate", "compute", "2 to", "power"]):
        events = _make_text_stream(
            "",
            tool_calls=[{"id": "t1", "name": "calculate", "input": {"expression": "2 ** 20"}}],
        )
    else:
        events = _make_text_stream("I can help with weather or math calculations!")

    print("Assistant: ", end="", flush=True)
    in_text = False

    def on_text(chunk: str):
        nonlocal in_text
        in_text = True
        print(chunk, end="", flush=True)

    def on_tool_start(name: str, id_: str):
        if in_text:
            print()
        print(f"  \n  [streaming tool_use: {name}] ", end="", flush=True)

    def on_tool_complete(name: str, id_: str, inputs: dict):
        print(f"→ input received: {inputs}", flush=True)

    acc = StreamAccumulator(
        on_text_delta=on_text,
        on_tool_start=on_tool_start,
        on_tool_complete=on_tool_complete,
    )

    for event in events:
        acc.process(event)
        time.sleep(0.05)  # simulate network arrival

    print()

    if acc.stop_reason == "tool_use":
        tool_results = []
        for name, id_, inputs in acc.tool_calls:
            print(f"  [executing {name}({inputs})...]", flush=True)
            result = dispatch_tool(name, inputs)
            print(f"  [result] {result}")
            tool_results.append((id_, result))

        # Simulate final response after tool results
        city_results = "; ".join(r for _, r in tool_results)
        print(f"Assistant: Here are the results: {city_results}")


# ---------------------------------------------------------------------------
# Real: live streaming from Anthropic API
# ---------------------------------------------------------------------------

def real_streaming(user_message: str) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": user_message}]
    print(f"\nUser: {user_message}")

    while True:
        print("Assistant: ", end="", flush=True)

        acc = StreamAccumulator(
            on_text_delta=lambda chunk: print(chunk, end="", flush=True),
            on_tool_start=lambda name, id_: print(f"\n  [tool_use: {name}] ", end="", flush=True),
            on_tool_complete=lambda name, id_, inp: print(f"→ {inp}", flush=True),
        )

        # Stream the response
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            for event in stream:
                # Map SDK events to our StreamEvent format
                event_type = event.type if hasattr(event, "type") else ""

                if event_type == "content_block_start":
                    from experiment import StreamEvent
                    acc.process(StreamEvent("content_block_start", {
                        "index": event.index,
                        "content_block": {
                            "type": event.content_block.type,
                            **({"id": event.content_block.id, "name": event.content_block.name}
                               if event.content_block.type == "tool_use" else {}),
                        },
                    }))

                elif event_type == "content_block_delta":
                    from experiment import StreamEvent
                    delta = event.delta
                    delta_dict = {"type": delta.type}
                    if delta.type == "text_delta":
                        delta_dict["text"] = delta.text
                    elif delta.type == "input_json_delta":
                        delta_dict["partial_json"] = delta.partial_json
                    acc.process(StreamEvent("content_block_delta", {
                        "index": event.index,
                        "delta": delta_dict,
                    }))

                elif event_type == "content_block_stop":
                    from experiment import StreamEvent
                    acc.process(StreamEvent("content_block_stop", {"index": event.index}))

                elif event_type == "message_delta":
                    from experiment import StreamEvent
                    acc.process(StreamEvent("message_delta", {
                        "delta": {"stop_reason": event.delta.stop_reason}
                    }))

        print()
        messages.append({"role": "assistant", "content": acc.to_content_list()})

        if acc.stop_reason != "tool_use":
            break

        # Execute tools and continue
        tool_results = []
        for name, id_, inputs in acc.tool_calls:
            print(f"  [executing {name}({inputs})]")
            result = dispatch_tool(name, inputs)
            print(f"  [result] {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": id_,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(use_mock: bool) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Streaming with Tools Demo [{mode}] ===")
    print("Watch text stream in real-time, with tool calls interleaved.\n")
    print("Example queries:")
    for q in ["What's the weather in Tokyo?", "Calculate 2 to the power of 20.", "Is it warmer in Tokyo or Paris?"]:
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
            mock_streaming(user_input)
        else:
            real_streaming(user_input)
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
