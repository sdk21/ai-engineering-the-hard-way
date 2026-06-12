# Lesson: Streaming with Tools

**Vertical:** Tools | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [Why Streaming Matters for Tool Use](#1-why-streaming-matters-for-tool-use)
2. [The Streaming Event Model](#2-the-streaming-event-model)
3. [Accumulating a Tool Call from Deltas](#3-accumulating-a-tool-call-from-deltas)
4. [The Streaming Agentic Loop](#4-the-streaming-agentic-loop)
5. [Text-Then-Tool vs Tool-Then-Text](#5-text-then-tool-vs-tool-then-text)
6. [Reconstructing the Messages Array](#6-reconstructing-the-messages-array)
7. [UX Considerations](#7-ux-considerations)
8. [Key Principles](#9-key-principles)
9. [In the Real World](#10-in-the-real-world)
10. [Running the Experiment](#11-running-the-experiment)

---

## 1. Why Streaming Matters for Tool Use

Without streaming, the user sees nothing until the model finishes its entire response — including any internal reasoning before a tool call. With a slow tool chain, this could mean 5–10 seconds of blank screen.

With streaming:
- Text tokens arrive as they're generated — the user sees the response forming
- Tool call arguments arrive incrementally — your UI can show "calling get_weather..." immediately
- The total time to first visible output drops from seconds to milliseconds

For chat interfaces, streaming is nearly always worth the added complexity. For background processing, it may not matter.

---

## 2. The Streaming Event Model

The Anthropic streaming API produces Server-Sent Events. The relevant events for tool use:

```
message_start           → new response begins
content_block_start     → a new content block begins (text or tool_use)
content_block_delta     → incremental data for the current block
  type=text_delta       → text chunk (for text blocks)
  type=input_json_delta → partial JSON (for tool_use blocks)
content_block_stop      → content block is complete
message_delta           → stop_reason arrives ("end_turn" or "tool_use")
message_stop            → response is done
```

A response with one text block and one tool_use block produces this event sequence:

```
message_start
content_block_start     (index=0, type=text)
content_block_delta     (index=0, text_delta: "Let me check")
content_block_delta     (index=0, text_delta: " that for you.")
content_block_stop      (index=0)
content_block_start     (index=1, type=tool_use, name="get_weather", id="tu_123")
content_block_delta     (index=1, input_json_delta: '{"city":')
content_block_delta     (index=1, input_json_delta: ' "Tokyo"}')
content_block_stop      (index=1)
message_delta           (stop_reason="tool_use")
message_stop
```

---

## 3. Accumulating a Tool Call from Deltas

The tool_use input arrives as partial JSON fragments. You must accumulate them into a complete JSON string before parsing:

```python
class StreamAccumulator:
    def __init__(self):
        self._blocks = {}  # index → ContentBlock

    def process(self, event):
        if event.type == "content_block_start":
            block = ContentBlock(type=event.content_block.type)
            if block.type == "tool_use":
                block.tool_id = event.content_block.id
                block.tool_name = event.content_block.name
            self._blocks[event.index] = block

        elif event.type == "content_block_delta":
            block = self._blocks[event.index]
            if event.delta.type == "text_delta":
                block.text += event.delta.text
            elif event.delta.type == "input_json_delta":
                block.input_buffer += event.delta.partial_json  # accumulate

        elif event.type == "content_block_stop":
            block = self._blocks.get(event.index)
            if block and block.type == "tool_use":
                block.input = json.loads(block.input_buffer)  # parse once complete
```

Key: **never try to parse partial JSON**. Accumulate the full string first, then parse on `content_block_stop`.

---

## 4. The Streaming Agentic Loop

The streaming loop mirrors the standard loop but processes events as they arrive:

```python
while True:
    acc = StreamAccumulator(on_text_delta=lambda t: print(t, end="", flush=True))

    with client.messages.stream(...) as stream:
        for event in stream:
            acc.process(event)

    messages.append({"role": "assistant", "content": acc.to_content_list()})

    if acc.stop_reason != "tool_use":
        break  # done

    # Execute tools (blocking — no streaming here)
    tool_results = []
    for name, id_, inputs in acc.tool_calls:
        result = dispatch_tool(name, inputs)
        tool_results.append({"type": "tool_result", "tool_use_id": id_, "content": result})

    messages.append({"role": "user", "content": tool_results})
    # Loop — stream the next response
```

The tool execution itself is not streamed (tools return complete results). Only the model's text generation streams.

---

## 5. Text-Then-Tool vs Tool-Then-Text

**Text-then-tool:** The model writes some text before calling a tool.
```
"Let me check the weather for you."  [text streams]
[get_weather called]
"The weather in Tokyo is 22°C."      [text streams]
```
UX: the user sees some text, then a pause, then the final answer. Show the intermediate text immediately.

**Tool-then-text:** The model calls tools first, then writes a response.
```
[get_weather called — user sees nothing yet]
"The weather in Tokyo is 22°C."      [text streams after tool returns]
```
UX: the user sees nothing until the tool completes. Show a loading indicator during tool execution.

You don't control which pattern the model uses — it decides. Handle both cases in your accumulator by tracking whether any text has been emitted yet.

---

## 6. Reconstructing the Messages Array

To continue the conversation after streaming, you need the full assistant message in the format the API expects. The `to_content_list()` method produces this:

```python
acc.to_content_list()
# [
#   {"type": "text", "text": "Let me check that for you."},
#   {"type": "tool_use", "id": "tu_123", "name": "get_weather", "input": {"city": "Tokyo"}}
# ]

messages.append({"role": "assistant", "content": acc.to_content_list()})
```

This is identical to the non-streaming case — the difference is that you build this list from accumulated events rather than from `response.content`.

---

## 7. UX Considerations

**Show text as it arrives.** The biggest benefit of streaming is that users see the response forming. Don't buffer — flush immediately.

**Show tool call indicators.** When a tool_use block starts, show the tool name immediately:
```
on_tool_start → print(f"🔍 Looking up {name}...")
```

**Show tool execution feedback.** After the tool completes:
```
on_tool_complete → print(f"✓ Got weather data")
```

**Handle long-running tools.** If the tool takes > 1 second, show a spinner or elapsed time counter. The model is waiting — the user should know something is happening.

**Don't show JSON during streaming.** The input JSON arrives as partial fragments. Don't render them directly — wait for `content_block_stop` and render the parsed input.

---

## 8. Key Principles

> **Principle 1 — Accumulate first, parse once.**
> Tool input JSON arrives in fragments. Buffer everything until `content_block_stop`, then parse. Partial JSON is not valid JSON.

> **Principle 2 — The reconstructed content list must be complete.**
> When appending the assistant message to `messages`, include every block — text and tool_use. An incomplete message causes the next API call to fail.

> **Principle 3 — Tool execution is still synchronous.**
> Streaming improves text latency. Tool execution happens between streaming rounds and is always blocking from the user's perspective. Fast tools are a UX priority.

> **Principle 4 — Streaming and non-streaming loops are structurally identical.**
> The only difference is how you build the accumulator. The messages array, the tool dispatch logic, and the loop termination condition are the same.

---

## 9. In the Real World

**ChatGPT / Claude.ai**
Both stream text and show tool call indicators in real time. "Searching the web..." appears as soon as the tool_use block starts — before the search even completes. This is the on_tool_start callback.

**Vercel AI SDK**
The most widely used React streaming library for AI apps. It provides hooks (`useChat`) that handle streaming accumulation, tool call extraction, and message array management out of the box.

**LangChain Streaming**
LangChain's `StreamingStdOutCallbackHandler` is the equivalent of an `on_text_delta` callback — it prints tokens as they arrive. LangChain also supports streaming with tool agents via `AgentExecutor` with `return_intermediate_steps=True`.

**GitHub Copilot Inline Suggestions**
Copilot's code suggestions appear character by character — this is streaming at the IDE integration layer. The model generates tokens and the IDE renders them incrementally.

---

## 10. Running the Experiment

```bash
# From the project root

# Mock mode — see streaming events with artificial delays
uv run python tools/09-streaming-tools/demo.py --mock

# Real mode — live streaming from Claude API
ANTHROPIC_API_KEY=sk-... uv run python tools/09-streaming-tools/demo.py --real
```

**Suggested exercises:**
1. Add a `time.time()` measurement to see the first-token latency vs. total latency.
2. Implement a spinner that displays while a tool is executing.
3. Buffer the last 3 words of text output before showing, to avoid rendering partial words. Measure whether this improves perceived quality.
4. Handle the case where the model starts streaming text but then interrupts itself to make a tool call mid-sentence.

---

*Previous: [Programmatic Tool Generation](../08-programmatic-tool-generation/) · Next: [Tool Use with Memory](../10-tool-use-with-memory/)*
