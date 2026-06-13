# Lesson: Backtracking

**Vertical:** Planning | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

Backtracking is a constraint satisfaction technique: assign variables one at a time, validate constraints after each assignment, and undo (backtrack) when a constraint is violated.

```
slot_1 = C  → ✗ Violates: "C must not be in slot 1" → backtrack
slot_1 = A  → ✓
slot_2 = B  → ✓ (A before B satisfied)
slot_3 = C  → ✓
slot_4 = D  → ✓ (D in last slot satisfied)
```

---

## Key Concepts

**Constraint satisfaction** — problems where variables must be assigned values from a domain, subject to hard constraints (not soft preferences).

**Hard vs. soft failures:**
- Self-reflection (exp 08): soft failure → improve quality iteratively
- Backtracking: hard failure → this assignment is **wrong**, undo it

**Failure memory** — failed (variable, value) pairs are recorded and passed back to the proposer so the model doesn't repeat the same mistake.

**Domain exhaustion** — if all values in the domain have been tried for a variable, backtrack further to the previous variable.

**Variable ordering** — the order in which variables are assigned affects efficiency. Assign the most constrained variable first (fewest legal values remaining) to detect failures early.

---

## Algorithm

```
function backtrack(assignment):
  if complete(assignment): return assignment
  var = select_unassigned_variable()
  for value in domain(var):
    if consistent(var, value, assignment):
      assignment[var] = value
      result = backtrack(assignment)
      if result is not None: return result
      del assignment[var]  # undo
  return None  # failure, trigger backtrack
```

---

## Running the Experiment

```bash
uv run python planning/10-backtracking/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/10-backtracking/demo.py --real --problem 1
ANTHROPIC_API_KEY=sk-... uv run python planning/10-backtracking/demo.py --real --problem 2
```

---

*Previous: [Tree of Thoughts](../09-tree-of-thoughts/) · Next: [Adaptive Planning](../11-adaptive-planning/)*
