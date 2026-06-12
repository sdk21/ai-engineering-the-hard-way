# Lesson: Multi-Tool Agent

**Vertical:** Tools | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

All previous experiments demonstrated isolated tool-use concepts: chaining, parallelism, errors, context, approval, selection, generation, streaming, memory. This experiment combines them into a single agent that uses 9 tools across 4 categories to accomplish complex, open-ended tasks.

The additions over previous experiments:

**Scratchpad tool** — The agent has a tool it uses on itself: `scratchpad_write` and `scratchpad_read`. Before tackling a complex request, it writes a plan. It records key findings as it goes. This makes reasoning explicit and inspectable.

**Planning before acting** — The system prompt instructs the agent to plan before using domain tools. This is the ReAct pattern applied via prompt: the model alternates between writing to the scratchpad (reasoning) and calling domain tools (acting).

**Synthesis** — Complex requests require combining multiple tool results into a coherent answer. The agent gathers all needed information first, then synthesizes. This is different from the simple "call tool → return result" pattern.

---

## The Scratchpad Pattern

The scratchpad is a tool that lets the agent write notes to itself:

```
Agent: [scratchpad_write("Plan: 1. Get company overview, 2. Check SF weather, 3. Calculate burn rate")]
Agent: [search_knowledge_base("company overview financial")]
Agent: [get_weather("san francisco")]
Agent: [calculate("800000 * 12")]
Agent: [scratchpad_read()]  ← review accumulated notes
Agent: [Final synthesis answer]
```

Why the scratchpad improves quality:
- Forces the model to articulate a plan before acting
- Creates a checkpoint to review all gathered information before answering
- Makes the reasoning process visible and debuggable

---

## Architecture

```
User query
    ↓
System prompt with role + scratchpad instructions
    ↓
Tool loop (max 15 iterations)
    ├── scratchpad_write (plan step)
    ├── search_knowledge_base
    ├── get_weather / get_stock_price
    ├── calculate
    ├── wikipedia_search / get_definition
    └── scratchpad_read (review step)
    ↓
Final synthesis response
```

---

## Running the Experiment

```bash
# From the project root
uv run python tools/11-multi-tool-agent/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python tools/11-multi-tool-agent/demo.py --real
```

**Suggested queries:**
- `"Give me a briefing on AcmeCorp: team size, finances, and current weather at HQ."`
- `"What's the capital of Japan, its current weather, and calculate 37 * 42?"`
- `"Search our knowledge base for financial info and calculate annual burn rate."`

**Suggested exercises:**
1. Remove the scratchpad from the tool list and observe how reasoning quality changes.
2. Add a `done(answer)` tool that the agent must call to submit its final answer, with a `max_steps` budget counter that it can read at any time.
3. Implement a `search_knowledge_base` with real vector similarity using the approach from the memory vertical (experiment 04).

---

*Previous: [Tool Use with Memory](../10-tool-use-with-memory/) · Next: [Tool Composition](../12-tool-composition/)*
