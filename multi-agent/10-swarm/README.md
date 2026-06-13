# Lesson: Swarm

**Vertical:** Multi-Agent | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

Many identical (or similar) agents process items from a shared work queue in parallel. No orchestrator assigns work — agents pick items themselves. Emergent coordination from simple local rules.

```
Work queue: [chunk_1, chunk_2, chunk_3, chunk_4, chunk_5]

worker-1 picks chunk_1 → annotates → done
worker-2 picks chunk_2 → annotates → done
worker-3 picks chunk_3 → annotates → done
worker-1 picks chunk_4 → annotates → done  (picks next available)
worker-2 picks chunk_5 → annotates → done

→ [Aggregator] → final summary
```

---

## Key Concepts

**No central orchestrator** — workers self-assign from the queue. A thread lock ensures no two workers pick the same item.

**Homogeneous agents** — all workers have the same system prompt. The "swarm intelligence" comes from the quantity and parallelism, not specialization.

**Work queue pattern** — `status: pending → processing → done` prevents double-assignment. This is the fundamental pattern for distributed task processing.

**Emergent efficiency** — fast workers naturally take more items. Slow items don't block other work.

**Compared to Parallel Fan-out (exp 03):**

| Parallel Fan-out | Swarm |
|---|---|
| Fixed N tasks, N agents | Dynamic: M tasks, N workers (M ≠ N) |
| Each agent has a specific task | Agents self-select from queue |
| Orchestrator assigns work | No orchestrator |

---

## Running the Experiment

```bash
uv run python multi-agent/10-swarm/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/10-swarm/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/10-swarm/demo.py --real --workers 5 --doc 2
```

---

*Previous: [Shared Blackboard](../09-shared-blackboard/) · Next: [Task Auction](../11-task-auction/)*
