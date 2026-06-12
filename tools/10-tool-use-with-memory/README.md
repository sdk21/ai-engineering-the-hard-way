# Lesson: Tool Use with Memory

**Vertical:** Tools | **Difficulty:** Intermediate–Advanced | **Status:** ✅ Ready

---

## Table of Contents

1. [The Problem: Stateless Tool Calls](#1-the-problem-stateless-tool-calls)
2. [Two Memory Strategies for Tools](#2-two-memory-strategies-for-tools)
3. [Semantic Caching](#3-semantic-caching)
4. [Fact Extraction from Results](#4-fact-extraction-from-results)
5. [Injecting Tool Memory into the Prompt](#5-injecting-tool-memory-into-the-prompt)
6. [Cache TTL Design](#6-cache-ttl-design)
7. [Cache Invalidation](#7-cache-invalidation)
8. [Key Principles](#8-key-principles)
9. [In the Real World](#9-in-the-real-world)
10. [Running the Experiment](#10-running-the-experiment)

---

## 1. The Problem: Stateless Tool Calls

Every tool call in the standard agentic loop is stateless — the same tool called with the same arguments makes the same API call again. In a multi-turn conversation:

```
Turn 1: "What's the weather in Tokyo?"
  → get_weather("Tokyo") → API call #1

Turn 3: "Is Tokyo weather still sunny?"
  → get_weather("Tokyo") → API call #2  (same data, redundant call)

Turn 5: "Remind me what Tokyo's weather was."
  → get_weather("Tokyo") → API call #3  (still the same data)
```

Three API calls for the same data. In production this means:
- **Cost:** 3x the tool API calls (often billed per call)
- **Latency:** 3x the wait time
- **Inconsistency:** prices or weather may have changed between calls, making answers contradictory within one session

---

## 2. Two Memory Strategies for Tools

**Strategy 1 — Result caching:** Store `(tool_name, arguments) → result`. On future calls with the same arguments, return the cached result instead of calling the API.

```python
cache = {}

def execute_with_cache(name, args):
    key = (name, json.dumps(args, sort_keys=True))
    if key in cache:
        return cache[key]  # cache hit
    result = actual_tool_call(name, args)
    cache[key] = result
    return result
```

**Strategy 2 — Fact extraction:** Parse tool results to extract structured facts. Store these facts separately and inject them into the system prompt, so the model "remembers" things it learned from tools without needing to call them again.

```python
# After get_weather("Tokyo") returns "22°C, sunny"
# Extract and store: {entity: "Tokyo", attribute: "temperature", value: "22°C"}

# Inject into next turn's system prompt:
# "## Facts from previous tool calls
#  Tokyo / temperature: 22°C (from get_weather, 30s ago)"
```

The two strategies complement each other:
- Caching prevents redundant calls for the same arguments
- Fact extraction enables the model to reason about past results without calling any tool

---

## 3. Semantic Caching

A basic cache uses exact argument matching. A semantic cache uses similarity — it returns cached results for "similar" queries even when the exact arguments differ:

```
Cache: get_weather("Tokyo") → "22°C sunny"
Query: get_weather("tokyo") → exact match (case-insensitive)
Query: get_weather("Tokyo, Japan") → fuzzy match
Query: "What's the weather like in the capital of Japan?" → semantic match (embedding-based)
```

This experiment implements exact + case-insensitive matching. For full semantic caching, embed the query and cache entry, then return hits above a similarity threshold.

Semantic caching is powerful for knowledge-heavy tools (Wikipedia, documentation search) where users often rephrase the same question.

---

## 4. Fact Extraction from Results

Tool results are unstructured text. Fact extraction converts them to structured memory:

```python
# get_weather("Tokyo") returns: "22°C, sunny, humidity 65%"
# Extracted facts:
{entity: "Tokyo", attribute: "temperature", value: "22°C"}
{entity: "Tokyo", attribute: "weather_condition", value: "sunny"}

# get_stock_price("AAPL") returns: "$189.30 (+0.64%)"
# Extracted facts:
{entity: "AAPL", attribute: "stock_price", value: "$189.30"}
```

These structured facts are stored in a fact store and injected into the system prompt. The model can reason over them directly without needing to call any tool.

**Extraction approaches by complexity:**
- **Regex:** fast, works for structured outputs with known patterns
- **LLM extraction:** flexible, works for any format, but adds latency
- **Schema-driven parsing:** for tools that return JSON, just read the fields

---

## 5. Injecting Tool Memory into the Prompt

The stored facts are injected into the system prompt as a "known facts" section:

```python
system = "You are a helpful assistant."
if memory_context:
    system += f"\n\n{memory_context}"

# memory_context:
# ## Facts from previous tool calls
#   Tokyo / temperature: 22°C (from get_weather, 45s ago)
#   Tokyo / weather_condition: sunny (from get_weather, 45s ago)
#   AAPL / stock_price: $189.30 (from get_stock_price, 2m ago)
```

The model reads these facts and can answer questions like "what was Tokyo's weather?" without making any tool call.

This is the bridge between the tools and memory verticals: tools generate facts; memory stores and retrieves them; the system prompt delivers them to the model.

---

## 6. Cache TTL Design

Different data has different useful lifespans:

| Tool | TTL | Rationale |
|------|-----|-----------|
| `calculate` | Never expires | Math doesn't change |
| `wikipedia_lookup` | 1 hour | Content rarely changes |
| `get_weather` | 5 minutes | Weather changes slowly |
| `convert_currency` | 2 minutes | FX rates change frequently |
| `get_stock_price` | 60 seconds | Prices change every second |

TTL is a per-tool configuration:

```python
TTL = {
    "calculate": None,          # None = never expire
    "wikipedia_lookup": 3600,
    "get_weather": 300,
    "convert_currency": 120,
    "get_stock_price": 60,
}
```

Choose TTLs conservatively — a stale cached result is worse than no result for time-sensitive data.

---

## 7. Cache Invalidation

The cache should be invalidatable when you know the data has changed:

```python
# Explicit invalidation
cache.invalidate("get_weather", {"city": "Tokyo"})

# Pattern invalidation
cache.invalidate_all("get_stock_price")  # clear all stock prices

# User-triggered bypass
result = executor.execute("get_weather", {"city": "Tokyo"}, bypass_cache=True)
```

The `bypass_cache=True` parameter lets the model or user force a fresh fetch when they explicitly want current data.

---

## 8. Key Principles

> **Principle 1 — Tool results are inputs to memory, not just outputs to the model.**
> Every tool call produces information. That information doesn't have to be ephemeral. Storing it enables the agent to build up knowledge across a session without redundant API calls.

> **Principle 2 — Cache by what the model asked for, not just by tool arguments.**
> Semantically equivalent queries ("Tokyo" vs "tokyo" vs "Tokyo, Japan") should hit the same cache entry. Normalize inputs before caching.

> **Principle 3 — TTL should match data volatility, not convenience.**
> Setting all TTLs to "forever" creates stale data bugs. Setting all TTLs to "never" defeats the purpose of caching. Match TTL to how fast the underlying data actually changes.

> **Principle 4 — Facts from tools should be first-class citizens in the system prompt.**
> Don't bury tool-derived facts in conversation history. Inject them explicitly into the system prompt where the model's attention is reliable.

> **Principle 5 — Measure cache hit rate.**
> A cache you can't measure is a cache you can't tune. Track hits, misses, and TTL expirations. In a well-tuned session, hit rates of 40–70% are common for repeat-topic conversations.

---

## 9. In the Real World

**Redis as Tool Cache**
Production AI agents almost always use Redis as a shared tool result cache. Any agent instance can benefit from calls made by other instances. TTL is set at the Redis key level, matching the data volatility per tool.

**Perplexity Search Cache**
Perplexity caches web search results within a session. Asking follow-up questions about the same topic returns results from the first search, not new API calls.

**LangChain Caching**
LangChain supports `InMemoryCache` and `RedisCache` at the LLM call level. This is LLM-level caching (caching the model's responses), not tool-level, but the pattern is identical.

**API Gateway Caching (AWS)**
In microservices, API gateways cache responses at the HTTP layer. This is infrastructure-level tool result caching — any caller (human or AI agent) benefits from the cache.

**Salesforce Einstein Activity History**
CRM AI features store tool-call-derived facts (call summaries, deal stages) in the activity history. Future AI queries draw on this structured fact store rather than re-calling the CRM APIs.

---

## 10. Running the Experiment

```bash
# From the project root

# Mock mode — see cache hits and fact extraction
uv run python tools/10-tool-use-with-memory/demo.py --mock

# Real mode — interactive session with cache + memory
ANTHROPIC_API_KEY=sk-... uv run python tools/10-tool-use-with-memory/demo.py --real
```

**Suggested session to see cache behavior:**
1. `"What's the weather in Tokyo?"` → API call, result cached, fact stored
2. `"What's the weather in Tokyo?"` → cache hit, no API call
3. `"Tell me about Tokyo."` → uses stored facts
4. `"memory"` → see all stored facts
5. `"stats"` → see hit rate

**Suggested exercises:**
1. Add a `bypass_cache=True` mode that ignores cached results. Verify the TTL kicks in after waiting.
2. Implement LLM-based fact extraction: pass the tool result to Claude and ask it to extract structured facts.
3. Add a `get_recent_calls()` method to the executor and inject the last 5 calls into the system prompt.
4. Implement cross-session persistence: save the fact store to JSON on exit, reload on startup.

---

*Previous: [Streaming with Tools](../09-streaming-tools/) · Next: [Multi-Tool Agent](../11-multi-tool-agent/)*
