# Lesson: Parallel Fan-out

**Vertical:** Multi-Agent | **Difficulty:** Beginner | **Status:** ✅ Ready

---

## What This Teaches

Parallel fan-out dispatches multiple independent subtasks simultaneously and aggregates the results — trading sequential latency for parallel wall-clock time.

```
Goal: "Adopting Kubernetes for a 20-person startup"
     │
     ▼
[Orchestrator] → fans out
     │
     ├──→ [technical agent]  ─────┐
     ├──→ [business agent]   ─────┤ (all running simultaneously)
     ├──→ [risk agent]       ─────┤
     └──→ [UX agent]         ─────┘
                                  │
                                  ▼
                           [Aggregator]
```

4 agents × ~300ms each = 300ms wall time (not 1200ms).

---

## Key Concepts

**Independence requirement** — fan-out only works when tasks don't depend on each other's output. If task B needs task A's result, use sequential orchestration (exp 02).

**Speedup** — wall-clock time ≈ slowest single task (not sum). With 4 agents each taking 300ms, fan-out takes ~300ms vs ~1200ms sequential. **4× speedup.**

**ThreadPoolExecutor** — Python's `concurrent.futures.ThreadPoolExecutor` handles the parallelism. Each agent runs in a separate thread making its own API call.

**Aggregation** — a final LLM call synthesizes all perspectives into a coherent response. The aggregator sees all agent outputs at once.

---

## Running the Experiment

```bash
uv run python multi-agent/03-parallel-fanout/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/03-parallel-fanout/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/03-parallel-fanout/demo.py --real --agents 2
```

---

*Previous: [Orchestrator](../02-orchestrator/) · Next: [Sequential Pipeline](../04-sequential-pipeline/)*
