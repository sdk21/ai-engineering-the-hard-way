# Lesson: Orchestrator + Subagents

**Vertical:** Multi-Agent | **Difficulty:** Beginner | **Status:** ✅ Ready

---

## What This Teaches

An orchestrator decomposes a complex goal into subtasks and delegates each to a specialized subagent, then assembles the results.

```
Goal: "What is FastAPI and should I use it?"
     │
     ▼
[Orchestrator] → decomposes
     │
     ├── [researcher]  "Find what FastAPI is and its key features"
     ├── [analyst]     "Analyze FastAPI's tradeoffs for REST API development"
     └── [writer]      "Write a recommendation based on the findings"
     │
     ▼
[Assembler] → final response
```

---

## Key Concepts

**Decomposition** — the orchestrator's core job is to break work into pieces that match agent specializations. A poor decomposition leads to duplicated or missed work.

**Context passing** — each subagent receives results from prior agents as context. The writer agent sees what the researcher and analyst found.

**Assembler step** — a final pass integrates all subagent outputs into a coherent response. This is cleaner than asking the last subagent to do everything.

**Compared to Router (exp 01):**

| Router | Orchestrator |
|---|---|
| One agent handles the request | Multiple agents each handle a piece |
| Dispatches to ONE subagent | Coordinates MANY subagents |
| Best for: classification and handoff | Best for: complex, multi-domain tasks |

---

## Running the Experiment

```bash
uv run python multi-agent/02-orchestrator/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/02-orchestrator/demo.py --real
```

---

*Previous: [Router](../01-router/) · Next: [Parallel Fan-out](../03-parallel-fanout/)*
