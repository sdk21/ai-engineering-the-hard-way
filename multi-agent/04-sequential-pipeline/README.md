# Lesson: Sequential Pipeline

**Vertical:** Multi-Agent | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## What This Teaches

A sequential pipeline passes data through a fixed chain of agents, each transforming or enriching it for the next stage.

```
Raw feedback
    │
    ▼
[cleaner]      "The app is very slow when I try to export my data..."
    │
    ▼
[classifier]   {type: "bug_report", sentiment: "negative", urgency: "high"}
    │
    ▼
[extractor]    {problem: "...", components: ["export"], action: "..."}
    │
    ▼
[summarizer]   "[BUG_REPORT] [HIGH] — performance\nProblem: ..."
```

---

## Key Concepts

**Single responsibility** — each stage does exactly one thing. The cleaner doesn't classify; the classifier doesn't extract.

**Data flow** — each stage receives the original input plus all prior stage outputs as context. The summarizer can see the raw text AND the classification AND the extracted entities.

**Composability** — stages are independently testable. You can swap the classifier for a rule-based one without touching the extractor.

**Compared to Orchestrator (exp 02):**

| Orchestrator | Sequential Pipeline |
|---|---|
| Decomposition is dynamic | Fixed stage sequence |
| Stages may be parallel | Always sequential |
| One agent per domain | Each stage is a transformation |

---

## Running the Experiment

```bash
uv run python multi-agent/04-sequential-pipeline/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/04-sequential-pipeline/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/04-sequential-pipeline/demo.py --real --stages 2
```

---

*Previous: [Parallel Fan-out](../03-parallel-fanout/) · Next: [Critic Agent](../05-critic-agent/)*
