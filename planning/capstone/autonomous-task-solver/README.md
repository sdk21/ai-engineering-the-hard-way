# Capstone: Autonomous Task Solver

**Vertical:** Planning | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

This capstone integrates all major planning techniques from the vertical into a single autonomous agent:

| Technique | Source | Role in Solver |
|---|---|---|
| Hierarchical Planning | exp 06 | Decomposes goal → sub-goals → tasks |
| Task Graph | exp 05 | Resolves dependencies, runs independent tasks in parallel |
| Plan-and-Execute | exp 07 | Full plan upfront; tools available during execution |
| Self-Reflection | exp 08 | Critiques draft answer, revises if needed |
| Replanning | exp 12 | Regenerates sub-plans when tasks fail |

---

## Architecture

```
Goal
 │
 ▼
[Hierarchical Planner]
  Goal → 2-4 Sub-goals
  Each Sub-goal → 2-4 Tasks (with tool assignments)
 │
 ▼
[For each Sub-goal: Topological Execution]
  Resolve task dependencies → layers
  Execute layers (parallel tasks run concurrently)
  On failure → [Replanner] generates replacement tasks
 │
 ▼
[Drafter]
  All task results → draft answer
 │
 ▼
[Self-Reflector]
  Critique draft → revise if needed → final answer
```

---

## Available Tools

| Tool | Input | Use Case |
|---|---|---|
| `search` | query string | Look up facts, technology descriptions |
| `calculate` | math expression | Numeric computations |
| `summarize` | text | Condense long results |

---

## Example Run

```
Goal: Design and recommend a tech stack for a real-time chat application.

Sub-goal 1: Research backend technologies
  ✓ [t1] [search] Research WebSocket support → "FastAPI has built-in WebSocket support..."
  ✓ [t2] [search] Research message broker options → "Redis Pub/Sub for real-time messaging..."

Sub-goal 2: Research database and storage options
  ✓ [t3] [search] Research database for chat history → "PostgreSQL for persistent storage..."
  ✓ [t4] [calculate] Estimate storage needs → "17.39 GB/year for 1000 active users"

Sub-goal 3: Synthesize recommendation
  ✓ [t5] Write tech stack recommendation → "FastAPI + Redis + PostgreSQL + React"

[Self-reflection: critique → improved answer]

Final Answer:
  FastAPI (WebSockets) + Redis Pub/Sub + PostgreSQL + React...
```

---

## Running the Capstone

```bash
uv run python planning/capstone/autonomous-task-solver/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/capstone/autonomous-task-solver/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python planning/capstone/autonomous-task-solver/demo.py --real --verbose
```

**Note:** Real mode makes ~10-15 API calls. Expect ~20-30 seconds with Haiku.

---

## What to Observe

- **Hierarchical structure**: the agent doesn't just list tasks — it groups them into coherent sub-goals
- **Parallel execution**: independent tasks within a sub-goal run concurrently
- **Tool selection**: the planner assigns tools to tasks during planning, not ad-hoc during execution
- **Self-reflection**: compare `draft_answer` vs `final_answer` — reflection often adds missing details
- **Replanning** (rare): introduce a forced failure to see the replanning path

---

*This is the capstone for the Planning vertical. The next vertical is: Multi-Agent Systems.*
