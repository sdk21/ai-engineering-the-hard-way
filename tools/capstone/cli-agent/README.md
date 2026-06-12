# Capstone: CLI Agent

**Vertical:** Tools | **Type:** Capstone | **Status:** ✅ Ready

---

## What This Capstone Is

The tools vertical experiments (01–12) each teach one concept in isolation. This capstone synthesises all of them into a single production-quality CLI agent:

| Experiment | Concept | Where it appears here |
|-----------|---------|----------------------|
| 01 | Basic function calling | Agentic loop foundation |
| 02 | Tool chaining | Multi-step sequential tool calls |
| 03 | Parallel calls | Concurrent execution of independent tools |
| 04 | Error handling | Actionable errors, recovery patterns |
| 05 | Result context | Confidence + caveats in tool results |
| 06 | Human-in-the-loop | Approval gate for write_note, send_summary |
| 07 | Dynamic selection | Tool registry with top-k retrieval |
| 08 | Programmatic generation | @tool decorator for tool definitions |
| 09 | Streaming | Token-by-token output via streaming API |
| 10 | Tool use with memory | Result caching + fact extraction |
| 11 | Multi-tool agent | Scratchpad planning + synthesis |
| 12 | Tool composition | Composite tools (compare_weather, weather_advisory) |

---

## Production Concerns

### 1. Persistence

Session history saved to disk:
```
~/.config/ai-hardway/cli-agent/
    session.json    ← conversation history + session metadata
```

Restart and continue where you left off.

### 2. Session Resume

If you restart within 4 hours of your last message, the session resumes automatically:
```
╔══ CLI Agent [REAL] ══╗
  Resumed session a3f2b1c9 (6 turn(s))
```

After 4 hours, a new session starts.

### 3. Human-in-the-Loop

Some tools require approval before executing:
- `write_note` — CONFIRM (shows preview, asks y/n)
- `send_summary` — CONFIRM (shows preview, asks y/n)
- `delete_all_notes` — BLOCK (never executes, explains why)

Read-only and compute tools execute automatically.

### 4. Tool Result Cache

Repeated tool calls return cached results without re-executing:
- `get_weather` — 5 minute TTL
- `get_stock_price` — 60 second TTL
- `calculate` — no expiry
- `wikipedia_search` — 1 hour TTL

View cache stats with the `stats` command.

---

## Commands

| Command | What it does |
|---------|-------------|
| `stats` | Cache hit rate, call counts, session info |
| `scratchpad` | Show the agent's planning notes |
| `notes` | List all saved notes |
| `quit` | Exit and save session |

---

## Running the Agent

```bash
# From the project root

# Mock mode — no API key, deterministic responses
uv run python tools/capstone/cli-agent/demo.py --mock

# Real mode — full Claude API
ANTHROPIC_API_KEY=sk-... uv run python tools/capstone/cli-agent/demo.py --real

# Force a new session
uv run python tools/capstone/cli-agent/demo.py --real --new

# Wipe all saved data
uv run python tools/capstone/cli-agent/demo.py --real --clear

# Custom data directory
uv run python tools/capstone/cli-agent/demo.py --mock --data-dir /tmp/test-agent
```

---

## Suggested Exercise Sequence

**Run 1 — basic tool use:**
```
You: What's the weather in Tokyo?
You: What's NVDA's stock price?
You: Calculate 2 ** 32
You: stats    ← see 0% cache hit rate (first calls)
You: quit
```

**Run 2 — session resume + cache:**
```
(restart within 4 hours)
You: What's the weather in Tokyo?    ← CACHE HIT
You: stats                           ← see 100% hit rate for these calls
You: compare the weather in Tokyo, London, Paris
```

**Run 3 — approval gate:**
```
You: Write a note about our Q3 plan  ← CONFIRM prompt
(approve or deny)
You: Send a summary to team@example.com  ← CONFIRM prompt
You: Delete all my notes             ← BLOCKED
You: notes                           ← see saved notes
```

**Run 4 — multi-step reasoning:**
```
You: Give me a full briefing: AcmeCorp overview, HQ weather, and AAPL stock
(observe: scratchpad planning, parallel tool calls, synthesis)
You: scratchpad   ← see planning notes the agent wrote
```

---

## Differences from Experiment 11

| Concern | Experiment 11 | This Capstone |
|---------|--------------|---------------|
| Persistence | In-memory only | Session saved to JSON |
| Session resume | Not supported | 4-hour resume window |
| Approval gate | Not implemented | CONFIRM/BLOCK for writes |
| Tool cache | Not implemented | TTL-based result cache |
| Composite tools | Not present | compare_weather, weather_advisory |
| Stats command | Not present | Cache hit rate, call counts |

---

*Part of the [Tools Vertical](../../) · See also: [Multi-Tool Agent experiment](../../11-multi-tool-agent/)*
