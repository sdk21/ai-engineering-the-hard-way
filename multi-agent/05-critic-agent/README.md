# Lesson: Critic Agent

**Vertical:** Multi-Agent | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## What This Teaches

A critic agent is a dedicated evaluator with its own system prompt and explicit scoring criteria — separate from the agent that generated the content.

```
Task → [Generator] → Draft
                       │
                  [Critic] → score=6, issues=[...]
                       │
                  [Reviser] → Revised draft
                       │
                  [Critic] → score=9, approved=true ✓
```

---

## Key Concepts

**Independence** — the critic has a different system prompt from the generator. It's not "the same model thinking twice" — it's a different agent with a different perspective and explicit evaluative instructions.

**Structured critique** — the critic returns a score, a boolean approval, a list of specific issues, and a verdict. Vague feedback ("this could be better") doesn't help the reviser.

**Approval threshold** — `approved=true` when score ≥ 7 AND no critical issues. This prevents the loop from running forever on a high-scoring draft with one minor issue.

**Compared to Self-Reflection (planning exp 08):**

| Self-Reflection | Critic Agent |
|---|---|
| Same model generates and critiques | Separate agent for critique |
| One prompt switch (generation → critic mode) | Fully separate system prompts |
| Cheaper (fewer agents) | More independent perspective |

---

## Running the Experiment

```bash
uv run python multi-agent/05-critic-agent/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/05-critic-agent/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/05-critic-agent/demo.py --real --max-rounds 4
```

---

*Previous: [Sequential Pipeline](../04-sequential-pipeline/) · Next: [Debate](../06-debate/)*
