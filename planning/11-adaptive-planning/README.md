# Lesson: Adaptive Planning

**Vertical:** Planning | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

Adaptive planning modifies the remaining plan based on what each step reveals — not because a step failed, but because new information makes future steps unnecessary, different, or reordered.

```
Goal: Research best Python web framework and write recommendation

Initial Plan:
  ○ Step 1: Research top frameworks
  ○ Step 2: Compare performance benchmarks
  ○ Step 3: Check community adoption
  ○ Step 4: Write recommendation

After Step 1 → [Adapter]: "Step 1 already covered benchmarks — skip Step 2"

Adapted Plan:
  ✓ Step 1: Research top frameworks → "FastAPI vs Flask..."
  ⊘ Step 2: Compare benchmarks  [skipped — made redundant by Step 1]
  ✓ Step 3: Check community adoption
  ✓ Step 4: Write recommendation
```

---

## Key Concepts

**Adaptation triggers** — a step succeeds but reveals that future steps are now unnecessary, redundant, or should be in a different order.

**Adapter model** — after each step, a dedicated LLM call checks whether the remaining plan needs updating. This keeps the adapter's focus narrow.

**Adaptation types:**
- Skip: remove now-unnecessary steps
- Insert: add new steps that the current results make necessary
- Reorder: change the sequence of remaining steps

---

## Adaptive Planning vs. Replanning

| | Adaptive Planning | Replanning (exp 12) |
|---|---|---|
| Trigger | Step succeeds, new info revealed | Step fails |
| Scope | Modify remaining steps | Regenerate entire plan |
| Example | "Found what we needed early — skip remaining research" | "API is down — find alternative data source" |

---

## Running the Experiment

```bash
uv run python planning/11-adaptive-planning/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/11-adaptive-planning/demo.py --real
```

---

*Previous: [Backtracking](../10-backtracking/) · Next: [Replanning](../12-replanning/)*
