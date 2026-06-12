"""
Streaming with Tools
---------------------
The Anthropic streaming API delivers responses as a sequence of events.
When tool use is involved, streaming changes the execution model:
the tool_use block arrives incrementally across multiple events.

Key concepts:
- Streaming event types: content_block_start, content_block_delta, content_block_stop
- Accumulating a tool_use block from streaming deltas
- The streaming agentic loop: emit text as it arrives, pause when tool call detected
- Partial rendering: show "thinking..." or a spinner while tool executes
- Reconstructing the full assistant turn from stream events for the messages array

Two streaming patterns:
  1. Text-then-tool: model emits some text, then calls a tool
  2. Tool-then-text: model calls tools first, then emits the final answer

Both require the same accumulation logic but different UX:
pattern 1 shows partial text, then pauses; pattern 2 shows a loading
indicator until the first text block arrives.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generator


# ---------------------------------------------------------------------------
# Streaming event simulation (matches Anthropic SSE event structure)
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    type: str
    data: dict = field(default_factory=dict)


def _make_text_stream(text: str, tool_calls: list[dict] = None) -> list[StreamEvent]:
    """
    Simulate the sequence of streaming events for a response that contains
    text blocks and optionally tool_use blocks.
    """
    events = [StreamEvent("message_start", {"message": {"id": "msg_sim", "role": "assistant"}})]

    # Text block
    if text:
        events.append(StreamEvent("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}))
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            events.append(StreamEvent("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": chunk}}))
        events.append(StreamEvent("content_block_stop", {"index": 0}))

    # Tool use blocks
    tool_calls = tool_calls or []
    for i, tc in enumerate(tool_calls):
        idx = (1 if text else 0) + i
        events.append(StreamEvent("content_block_start", {
            "index": idx,
            "content_block": {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": {}},
        }))
        # Tool input arrives as JSON chunks
        input_json = json.dumps(tc["input"])
        chunk_size = max(1, len(input_json) // 3)
        for start in range(0, len(input_json), chunk_size):
            chunk = input_json[start:start + chunk_size]
            events.append(StreamEvent("content_block_delta", {
                "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": chunk},
            }))
        events.append(StreamEvent("content_block_stop", {"index": idx}))

    stop_reason = "tool_use" if tool_calls else "end_turn"
    events.append(StreamEvent("message_delta", {"delta": {"stop_reason": stop_reason}}))
    events.append(StreamEvent("message_stop", {}))
    return events


# ---------------------------------------------------------------------------
# Stream accumulator — builds the assistant message from events
# ---------------------------------------------------------------------------

@dataclass
class ContentBlock:
    type: str          # "text" or "tool_use"
    text: str = ""     # for text blocks
    tool_id: str = ""  # for tool_use blocks
    tool_name: str = ""
    input_buffer: str = ""  # raw JSON accumulation
    input: dict = field(default_factory=dict)  # parsed input


class StreamAccumulator:
    """
    Accumulates streaming events into a complete assistant message.
    Provides callbacks for real-time rendering.
    """

    def __init__(
        self,
        on_text_delta: Callable[[str], None] = None,
        on_tool_start: Callable[[str, str], None] = None,
        on_tool_complete: Callable[[str, str, dict], None] = None,
    ):
        self._blocks: dict[int, ContentBlock] = {}
        self._stop_reason = ""
        self.on_text_delta = on_text_delta or (lambda _: None)
        self.on_tool_start = on_tool_start or (lambda name, id_: None)
        self.on_tool_complete = on_tool_complete or (lambda name, id_, inp: None)

    def process(self, event: StreamEvent) -> None:
        t = event.type
        d = event.data

        if t == "content_block_start":
            idx = d["index"]
            cb = d["content_block"]
            block = ContentBlock(type=cb["type"])
            if cb["type"] == "tool_use":
                block.tool_id = cb["id"]
                block.tool_name = cb["name"]
                self.on_tool_start(block.tool_name, block.tool_id)
            self._blocks[idx] = block

        elif t == "content_block_delta":
            idx = d["index"]
            delta = d["delta"]
            block = self._blocks[idx]

            if delta["type"] == "text_delta":
                block.text += delta["text"]
                self.on_text_delta(delta["text"])

            elif delta["type"] == "input_json_delta":
                block.input_buffer += delta["partial_json"]

        elif t == "content_block_stop":
            idx = d["index"]
            block = self._blocks.get(idx)
            if block and block.type == "tool_use" and block.input_buffer:
                block.input = json.loads(block.input_buffer)
                self.on_tool_complete(block.tool_name, block.tool_id, block.input)

        elif t == "message_delta":
            self._stop_reason = d.get("delta", {}).get("stop_reason", "")

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def text_blocks(self) -> list[str]:
        return [b.text for b in self._blocks.values() if b.type == "text" and b.text]

    @property
    def tool_calls(self) -> list[tuple[str, str, dict]]:
        """Returns list of (tool_name, tool_id, input_dict)."""
        return [
            (b.tool_name, b.tool_id, b.input)
            for b in self._blocks.values()
            if b.type == "tool_use"
        ]

    def to_content_list(self) -> list[dict]:
        """Convert accumulated blocks to the format expected by the messages API."""
        result = []
        for block in self._blocks.values():
            if block.type == "text":
                result.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                result.append({
                    "type": "tool_use",
                    "id": block.tool_id,
                    "name": block.tool_name,
                    "input": block.input,
                })
        return result


# ---------------------------------------------------------------------------
# Example tools for the streaming demo
# ---------------------------------------------------------------------------

def slow_weather(city: str) -> str:
    """Simulates a slow API call."""
    import time
    time.sleep(0.5)
    data = {"tokyo": "22°C sunny", "paris": "17°C rainy", "london": "15°C cloudy"}
    return data.get(city.lower(), f"No data for {city}")


def quick_calc(expression: str) -> str:
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}
    def ev(node):
        if isinstance(node, ast.Constant): return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError(f"Unsupported: {node}")
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

TOOL_FN = {"get_weather": slow_weather, "calculate": quick_calc}


def dispatch_tool(name: str, inputs: dict) -> str:
    fn = TOOL_FN.get(name)
    return fn(**inputs) if fn else f"Unknown tool: {name}"
