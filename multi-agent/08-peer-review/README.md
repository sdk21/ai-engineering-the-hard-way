# Lesson: Peer Review

**Vertical:** Multi-Agent | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## What This Teaches

An author agent writes content. A reviewer agent provides structured feedback WITH specific suggestions. The author revises. The reviewer signs off.

```
Task → [Author] → Draft
                    │
             [Reviewer] → score=6, comments=[{issue, suggestion}, ...]
                    │
              [Author] → Revised draft (addressing each comment)
                    │
             [Reviewer] → APPROVED ✓
```

---

## Key Concepts

**Suggestions, not just issues** — the reviewer must provide a concrete suggestion for every issue. "This is unclear" alone doesn't help the author; "Replace X with Y" does.

**Severity levels** — comments are tagged as `minor`, `major`, or `critical`. This helps the author prioritize and helps the reviewer decide whether to approve.

**Sign-off loop** — after revision, the reviewer does a focused final check: "Did the author address my comments?" This is cheaper than a full re-review.

**Compared to Critic Agent (exp 05):**

| Critic Agent | Peer Review |
|---|---|
| Critic identifies issues | Reviewer identifies issues AND suggests fixes |
| Generator revises | Author (original agent) revises |
| Focus: quality improvement | Focus: collaborative refinement |

---

## Running the Experiment

```bash
uv run python multi-agent/08-peer-review/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/08-peer-review/demo.py --real
```

---

*Previous: [Consensus Voting](../07-consensus-voting/) · Next: [Shared Blackboard](../09-shared-blackboard/)*
