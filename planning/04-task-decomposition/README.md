# Lesson: Task Decomposition

**Vertical:** Planning | **Difficulty:** Beginner–Intermediate | **Status:** ✅ Ready

---

## What This Teaches

Task decomposition breaks a complex goal into a flat ordered list of concrete, executable steps. It separates two concerns:

1. **Planning** — the model generates the step list
2. **Execution** — a runner executes each step in order

```
Goal: "Write and publish a blog post about exercise benefits"

Plan:
  1. Research the latest scientific studies on exercise benefits
  2. Outline the blog post structure
  3. Write the first draft
  4. Review and edit for clarity
  5. Add statistics and citations
  6. Proofread
  7. Publish and share
```

The key design question: **how granular should steps be?** Too coarse ("write the blog post") and the executor can't act. Too fine ("type the letter 'T'") and the plan is useless. The right granularity is: each step should be something a capable executor can complete in one focused session.

---

## Plan Quality Criteria

A good decomposition satisfies:
- **Completeness** — executing all steps achieves the goal
- **Actionability** — each step is concrete enough to execute
- **Ordering** — steps are in dependency order
- **Appropriate granularity** — not too coarse, not too fine
- **Non-redundancy** — no two steps do the same thing

Prompting tip: include these criteria in the system prompt. The model will try to satisfy them.

---

## Limitations of Flat Decomposition

| Limitation | Addressed by |
|-----------|-------------|
| No parallelism | Task Graph (exp 05) |
| No hierarchy | Hierarchical Planning (exp 06) |
| No recovery from failure | Backtracking (exp 10), Replanning (exp 12) |
| No plan revision | Adaptive Planning (exp 11) |

Flat decomposition is the right starting point — understand its limits before adding complexity.

---

## Running the Experiment

```bash
uv run python planning/04-task-decomposition/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/04-task-decomposition/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python planning/04-task-decomposition/demo.py --real --execute
```

---

*Previous: [ReAct](../03-react/) · Next: [Task Graph](../05-task-graph/)*
