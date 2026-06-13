# Lesson: Plan-and-Execute

**Vertical:** Planning | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## What This Teaches

Plan-and-Execute is a two-phase agent pattern that separates *thinking about what to do* from *doing it*:

```
Phase 1: PLAN
  Goal → [Planner LLM] → Ordered step list (with tools)

Phase 2: EXECUTE
  For each step:
    If tool step → dispatch tool, record result
    If synthesis step → [Executor LLM] with full plan + prior results

Phase 3: SYNTHESIZE
  All results → [Synthesizer LLM] → Final answer
```

---

## Key Concepts

**Upfront planning** — the model produces a complete plan before taking any action. This allows the model (and the user) to review and catch errors before they compound.

**Separation of concerns** — planning uses a different prompt (and could use a different model) than execution. The planner focuses on *what* to do; executors focus on *how* to do one specific step.

**Tool dispatch** — execution steps specify which tool to use (`search`, `calculate`, `none`). Deterministic tools run without an LLM call; reasoning steps call the executor model.

**Inspectability** — the plan is a first-class artifact. You can log it, display it to a user for approval, or modify it before execution begins.

---

## Compared to ReAct (exp 03)

| ReAct | Plan-and-Execute |
|---|---|
| Interleaves thinking and acting | Plans fully upfront, then acts |
| No global overview | Complete plan visible before first action |
| More adaptive to surprises | More predictable, auditable |
| Best for: open-ended exploration | Best for: structured tasks with known steps |

---

## Running the Experiment

```bash
uv run python planning/07-plan-and-execute/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/07-plan-and-execute/demo.py --real
```

---

*Previous: [Hierarchical Planning](../06-hierarchical-planning/) · Next: [Self-Reflection](../08-self-reflection/)*
