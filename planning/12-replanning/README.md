# Lesson: Replanning

**Vertical:** Planning | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

Replanning handles catastrophic step failures by generating a completely new plan that works around the failure, building on what was already completed.

```
Goal: Get AAPL stock price and calculate P/E ratio

Plan v1:
  ✓ Step 1: Query Yahoo Finance for AAPL price → $182.50
  ✗ Step 2: Query SEC EDGAR for EPS → FAILED: 503 Service Unavailable

  [Replanning...]

Plan v2 (new steps only):
  ✓ Step 3: Search MarketWatch for AAPL TTM EPS → $6.13
  ✓ Step 4: Calculate P/E = 182.50 / 6.13 → 29.77
  ✓ Step 5: Summarize findings
```

---

## Key Concepts

**Failure detection** — the executor returns `FAILED: <reason>` for steps that can't complete. This structured signal triggers replanning.

**Context preservation** — the replanner receives: original goal + all completed steps + the failed step and reason. It builds on completed work rather than starting from scratch.

**New approach requirement** — the replanner prompt explicitly says "do NOT retry the same approach." This prevents infinite loops on the same failure.

**Replan budget** — `--max-replans` limits how many times the plan can be regenerated. After the budget is exhausted, execution stops with partial results.

---

## Compared to Related Patterns

| Pattern | Trigger | Scope | Use When |
|---|---|---|---|
| Self-reflection (08) | Output quality | Revise response | Output is OK but could be better |
| Backtracking (10) | Constraint violation | Undo last assignment | Hard constraint violated |
| Adaptive planning (11) | Step succeeds, new info | Tweak remaining steps | Plan is still valid but should be updated |
| **Replanning (12)** | **Step fails** | **Regenerate remaining plan** | **Step is impossible; need new route** |

---

## Running the Experiment

```bash
uv run python planning/12-replanning/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/12-replanning/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python planning/12-replanning/demo.py --real --max-replans 2
```

---

*Previous: [Adaptive Planning](../11-adaptive-planning/) · Next: [Capstone: Autonomous Task Solver](../capstone/autonomous-task-solver/)*
