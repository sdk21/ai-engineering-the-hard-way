# Lesson: Agent Router

**Vertical:** Multi-Agent | **Difficulty:** Beginner | **Status:** ✅ Ready

---

## What This Teaches

An agent router classifies an incoming request and dispatches it to the appropriate specialized agent. It's the simplest form of multi-agent orchestration.

```
User message
     │
     ▼
  [Router]  ← classifies intent
     │
     ├── billing_agent    → invoices, charges, subscriptions
     ├── technical_agent  → bugs, API issues, how-tos
     ├── refund_agent     → refunds, cancellations, returns
     └── general_agent    → everything else
```

---

## Key Concepts

**Specialization beats generalization for depth** — a focused agent with a narrow prompt and targeted tools outperforms a general agent on its specific domain.

**Two routing strategies:**

| Strategy | How | Accuracy | Cost |
|---|---|---|---|
| Rule-based | Keyword matching | Medium | Zero |
| LLM-based | Model classifies intent | High | ~1 extra call |

**Subagent design:**
- Narrow system prompt — only what the agent needs to know
- Focused scope — one domain, not five
- Clear handoff — structured response with status

---

## Running the Experiment

```bash
uv run python multi-agent/01-router/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/01-router/demo.py --real --strategy rule
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/01-router/demo.py --real --strategy llm
```

---

*Next: [Orchestrator + Subagents](../02-orchestrator/)*
