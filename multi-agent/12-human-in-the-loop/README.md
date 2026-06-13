# Lesson: Human-in-the-Loop

**Vertical:** Multi-Agent | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

The agent pauses at designated checkpoints and requests human approval, clarification, or editing before continuing.

```
Complaint: "Charged $250 for cancelled service. Disputing TODAY."
     │
[Agent drafts response]
     │
[Confidence checker] → risk_level=HIGH, needs_review=true
     │
┌─ CHECKPOINT [REVIEW] ───────────────────────────────
│  This involves a $250 refund. Review before sending.
│  Options: [a]pprove / [r]eject / [e]dit / [s]escalate
└─────────────────────────────────────────────────────
     │ human: [e]dit → "Add 3-5 day processing timeframe"
     │
[Agent revises] → final response sent ✓
```

---

## Key Concepts

**Checkpoint types:**
- **Approval gate** — agent proposes action, human approves/rejects
- **Clarification** — agent is uncertain, asks for input
- **Review gate** — agent shows completed work before delivery
- **Escalation** — agent can't handle this, routes to human

**Confidence-based triggering** — a dedicated classifier checks whether the draft needs human review. Low-risk, low-stakes responses go directly; high-risk ones get a checkpoint.

**HITL is a design choice, not a fallback** — you choose which decisions require human judgment. It's not about the agent failing; it's about maintaining appropriate control.

**`--auto-approve` flag** — runs the full pipeline without human prompts. Useful for comparing HITL vs. fully autonomous behavior.

---

## Running the Experiment

```bash
uv run python multi-agent/12-human-in-the-loop/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/12-human-in-the-loop/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/12-human-in-the-loop/demo.py --real --auto-approve
```

---

*Previous: [Task Auction](../11-task-auction/) · Next: [Hierarchical Multi-Agent](../13-hierarchical/)*
