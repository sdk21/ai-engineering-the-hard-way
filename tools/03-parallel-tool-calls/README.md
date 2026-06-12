# Lesson: Parallel Tool Calls

**Vertical:** Tools | **Difficulty:** Beginner–Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [What Are Parallel Tool Calls?](#1-what-are-parallel-tool-calls)
2. [How the Model Issues Parallel Calls](#2-how-the-model-issues-parallel-calls)
3. [The Response Structure](#3-the-response-structure)
4. [Executing in Parallel — Your Code's Role](#4-executing-in-parallel--your-codes-role)
5. [When the Model Parallelizes vs. Sequences](#5-when-the-model-parallelizes-vs-sequences)
6. [Latency Impact](#6-latency-impact)
7. [Returning Multiple Results](#7-returning-multiple-results)
8. [Failure Handling in Parallel Calls](#8-failure-handling-in-parallel-calls)
9. [Key Principles](#9-key-principles)
10. [In the Real World](#10-in-the-real-world)
11. [Running the Experiment](#11-running-the-experiment)

---

## 1. What Are Parallel Tool Calls?

When a task requires multiple independent pieces of information, the model can emit several `tool_use` blocks in a single response. This is parallel tool calling.

**Sequential (expensive — 4 round-trips):**
```
User → API → [tool_use: get_price(AAPL)]
→ [tool_result] → API → [tool_use: get_price(MSFT)]
→ [tool_result] → API → [tool_use: get_rating(AAPL)]
→ [tool_result] → API → [tool_use: get_rating(MSFT)]
→ [tool_result] → API → final answer
```

**Parallel (efficient — 1 round-trip):**
```
User → API → [tool_use: get_price(AAPL), tool_use: get_price(MSFT),
               tool_use: get_rating(AAPL), tool_use: get_rating(MSFT)]
→ [4 tool_results] → API → final answer
```

The model recognizes when calls are independent and batches them automatically. You don't request parallel execution — you just implement the agentic loop correctly and the model does it.

---

## 2. How the Model Issues Parallel Calls

The model's response content is a list. When it wants parallel calls, it includes multiple `tool_use` blocks:

```python
response.content = [
    TextBlock(text="Let me look up all four data points at once."),  # optional thought
    ToolUseBlock(id="tu_1", name="get_stock_price", input={"ticker": "AAPL"}),
    ToolUseBlock(id="tu_2", name="get_stock_price", input={"ticker": "MSFT"}),
    ToolUseBlock(id="tu_3", name="get_analyst_rating", input={"ticker": "AAPL"}),
    ToolUseBlock(id="tu_4", name="get_analyst_rating", input={"ticker": "MSFT"}),
]
response.stop_reason == "tool_use"
```

Each block has a unique `id`. You must return results with the matching IDs. The loop code is identical to the sequential case — the difference is that your tool result list has multiple entries:

```python
tool_results = []
for block in [b for b in response.content if b.type == "tool_use"]:
    result = dispatch_tool(block.name, block.input)
    tool_results.append({
        "type": "tool_result",
        "tool_use_id": block.id,   # must match the tool_use id
        "content": result,
    })
# Return ALL results in one message
messages.append({"role": "user", "content": tool_results})
```

---

## 3. The Response Structure

A parallel response packs multiple `tool_use` blocks into a single assistant turn:

```
Assistant turn (stop_reason = "tool_use"):
  content[0]: TextBlock — "I'll retrieve all data in parallel."  (optional)
  content[1]: ToolUseBlock — id="tu_1", name="get_stock_price", input={"ticker": "AAPL"}
  content[2]: ToolUseBlock — id="tu_2", name="get_stock_price", input={"ticker": "MSFT"}
  content[3]: ToolUseBlock — id="tu_3", name="get_analyst_rating", input={"ticker": "AAPL"}
  content[4]: ToolUseBlock — id="tu_4", name="get_analyst_rating", input={"ticker": "MSFT"}

User turn (your code):
  content[0]: ToolResultBlock — tool_use_id="tu_1", content="price: $189.30, ..."
  content[1]: ToolResultBlock — tool_use_id="tu_2", content="price: $415.80, ..."
  content[2]: ToolResultBlock — tool_use_id="tu_3", content="consensus: Buy, ..."
  content[3]: ToolResultBlock — tool_use_id="tu_4", content="consensus: Strong Buy, ..."
```

The API requires that all tool results for a round are returned together in a single user message. You cannot send them piecemeal.

---

## 4. Executing in Parallel — Your Code's Role

The model issues parallel *requests* — it's your code that decides whether to execute them concurrently. The simplest (but slower) approach executes them sequentially:

```python
# Sequential execution of "parallel" model calls — safe but slow
for block in tool_use_blocks:
    result = dispatch_tool(block.name, block.input)
    tool_results.append({"tool_use_id": block.id, "content": result})
```

For genuine speed improvement, execute with a thread pool:

```python
import concurrent.futures

def execute_parallel(tool_use_blocks):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            block.id: executor.submit(dispatch_tool, block.name, block.input)
            for block in tool_use_blocks
        }
        return [
            {"type": "tool_result", "tool_use_id": id_, "content": f.result()}
            for id_, f in futures.items()
        ]
```

**Use threads for I/O-bound tools** (HTTP calls, database queries — the common case).
**Use asyncio** if your entire stack is async (httpx, asyncpg, etc.).
**Avoid multiprocessing** for tool dispatch — the overhead is not worth it unless tools are CPU-intensive.

---

## 5. When the Model Parallelizes vs. Sequences

The model uses tool descriptions and conversational context to decide:

**Parallelizes when:**
- Tools take the same type of input (multiple tickers, multiple cities)
- Tools retrieve independent data (price AND rating for the same ticker)
- There is no dependency between outputs

**Sequences when:**
- Tool B requires the output of tool A (see tool chaining experiment)
- The description says "call X first before Y"
- The user query implies a conditional: "if the price is above $200, get the analyst rating"

**You can influence this** with descriptions. Adding "This tool can be called alongside other tools simultaneously" in a description signals independence. Adding "You must call look_up_flight first" signals a dependency.

---

## 6. Latency Impact

Parallel calls directly reduce end-to-end latency. The comparison:

| Scenario | Sequential | Parallel |
|----------|-----------|---------|
| 4 independent tools, 300ms each | 1,200ms | ~300ms |
| 8 independent tools, 200ms each | 1,600ms | ~200ms |
| 2 dependent chains of 3 tools   | 1,800ms | ~900ms |

In production, each tool call is typically an HTTP request (10–500ms). For queries that trigger 4–8 tool calls, parallel execution can mean a 4–8x latency reduction.

**Note:** The API call itself still takes one round-trip per batch. You save round-trips, not API latency.

---

## 7. Returning Multiple Results

The tool results message must return results for ALL pending tool calls in a single user turn. Returning a partial set will likely cause an API error or confuse the model.

```python
# Wrong — sends one result at a time
for block in tool_use_blocks:
    result = dispatch_tool(block.name, block.input)
    messages.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": block.id, "content": result}
    ]})  # DON'T DO THIS

# Right — sends all results together
all_results = [
    {"type": "tool_result", "tool_use_id": block.id, "content": dispatch_tool(block.name, block.input)}
    for block in tool_use_blocks
]
messages.append({"role": "user", "content": all_results})  # one message
```

---

## 8. Failure Handling in Parallel Calls

When one of several parallel tools fails, you have three options:

**Option A — Return the error, let the model decide:**
```python
{"type": "tool_result", "tool_use_id": block.id, "content": "Error: rate limit exceeded"}
```
The model will incorporate the error into its response (e.g., "I got an error for MSFT but here are results for AAPL").

**Option B — Mark as error with is_error:**
```python
{"type": "tool_result", "tool_use_id": block.id, "content": "rate limit exceeded", "is_error": True}
```
The model treats this as a tool failure, not an informational result.

**Option C — Retry failed calls** before returning (if transient):
```python
for attempt in range(3):
    try:
        result = dispatch_tool(name, inputs)
        break
    except RateLimitError:
        time.sleep(2 ** attempt)
```

In production, option A is usually sufficient. The model handles partial results gracefully.

---

## 9. Key Principles

> **Principle 1 — Parallel calls are the model's decision, not yours.**
> You write the same agentic loop. The model decides to issue multiple tool_use blocks when it recognizes independence. Good descriptions help it recognize independence.

> **Principle 2 — The model's parallelism is at the API level; yours is at the execution level.**
> The model batches tool requests. Your code decides whether to execute them concurrently. These are independent decisions — you can execute "parallel" model calls sequentially, and you can execute single model calls with internal concurrency.

> **Principle 3 — Return all results in one message.**
> The API requires all tool results for a round to arrive together. Partial responses are not supported.

> **Principle 4 — Parallel execution compounds with tool chaining.**
> A 3-step chain where each step calls 4 parallel tools produces 12 executions in 3 round-trips. Design your tools to maximize independent calls per round.

> **Principle 5 — Partial failures are normal.**
> In production, some tools in a parallel batch will occasionally fail. Return errors for failed tools and let the model synthesize the partial results gracefully.

---

## 10. In the Real World

**Perplexity / Search Agents**
When a search query spawns multiple sub-queries, they run in parallel. Answering "Compare the market cap of FAANG companies" triggers 5 simultaneous searches.

**Customer Support Agents**
A support agent may simultaneously call `get_account_status`, `get_recent_orders`, and `get_open_tickets` when a customer contacts support — 3 independent data sources fetched in one round.

**Code Review Agents**
When analyzing a pull request, an agent calls `read_file` for each changed file in parallel — then runs `lint_code` on all results in the next round.

**RAG with Multiple Indices**
A document Q&A agent may search 3 different knowledge bases simultaneously (product docs, support history, pricing) and synthesize results.

**Browser Use / Computer Agents**
Agents that control browsers often process multiple page elements in parallel — reading text from multiple sections of a page simultaneously rather than one at a time.

---

## 11. Running the Experiment

```bash
# From the project root

# Mock mode — see parallel dispatch without an API key
uv run python tools/03-parallel-tool-calls/demo.py --mock

# Real mode — watch Claude batch independent tool calls
ANTHROPIC_API_KEY=sk-... uv run python tools/03-parallel-tool-calls/demo.py --real
```

**Suggested queries:**
- `"Give me a full report on AAPL."` — model calls all 4 tools for one ticker in parallel
- `"Compare AAPL and MSFT — prices, analyst ratings, and recent news."` — up to 6 calls in parallel
- `"What's the stock price for NVDA?"` — single call (observe no parallelism)

**Suggested exercises:**
1. Add a 5th tool `get_earnings_history(ticker)`. Observe how the model incorporates it into the parallel batch.
2. Add `time.sleep(0.5)` to each tool function and measure the wall-clock difference between sequential and parallel execution.
3. Deliberately return an error from `get_analyst_rating` for MSFT and observe how the model handles partial results.
4. Modify the system prompt to say "call each tool separately, one at a time" and observe whether the model obeys.

---

*Previous: [Tool Chaining](../02-tool-chaining/) · Next: [Tool Error Handling](../04-tool-error-handling/)*
