# Lesson: Task Graph (DAG Planning)

**Vertical:** Planning | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## What This Teaches

A task graph represents a plan as a **directed acyclic graph (DAG)**. Nodes are tasks; edges represent dependencies. Tasks without dependencies between them can run in parallel.

```
design_db ──────→ impl_db ──────────────────────────────┐
design_api ─────→ impl_api ──→ write_tests ──→ integration_test → deploy
design_ui ──────→ impl_ui ────────────────────────────────┘
```

The three design tasks have no dependencies — they can all start at the same time. `integration_test` must wait for all three implementation tasks.

---

## Key Concepts

**Topological layers** — groups of tasks that can run concurrently:
```
Layer 1: [design_db, design_api, design_ui]    ← all in parallel
Layer 2: [impl_db, impl_api, impl_ui]          ← parallel within layer
Layer 3: [write_tests]
Layer 4: [integration_test]
Layer 5: [deploy]
```

**Critical path** — the longest dependency chain. Determines the minimum possible completion time regardless of parallelism. Shortening any task on the critical path reduces total time; shortening off-path tasks does not.

**Speedup** — if 9 tasks totaling 870 minutes reduce to a critical path of 285 minutes with parallel execution, that's a 3×+ speedup with the same total work.

---

## Model Output Format

The model generates a JSON task graph:
```json
{
  "tasks": [
    {"id": "design_api", "description": "Design REST API", "depends_on": [], "estimated_minutes": 45},
    {"id": "impl_api", "description": "Implement API", "depends_on": ["design_api"], "estimated_minutes": 180}
  ]
}
```

JSON is machine-parseable into a proper graph structure — unlike the numbered list from task decomposition (exp 04).

---

## Running the Experiment

```bash
uv run python planning/05-task-graph/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/05-task-graph/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python planning/05-task-graph/demo.py --real --execute
```

---

*Previous: [Task Decomposition](../04-task-decomposition/) · Next: [Hierarchical Planning](../06-hierarchical-planning/)*
