# Lesson: Consensus Voting

**Vertical:** Multi-Agent | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## What This Teaches

N independent agents each answer the same question. The most common answer wins (majority vote). Weighted voting accounts for agent confidence.

```
Question: "Bat + ball = $1.10, bat costs $1 more. Ball costs?"

[logical]      → $0.05 (confidence: 95%)
[intuitive]    → $0.10 (confidence: 60%)  ← cognitive bias trap
[skeptical]    → $0.05 (confidence: 90%)
[conservative] → $0.05 (confidence: 85%)
[creative]     → $0.05 (confidence: 80%)

Majority: $0.05 (4/5) — strong consensus
Weighted: $0.05
```

---

## Key Concepts

**Independence** — agents run in parallel with no communication. Each forms its own opinion without being influenced by others.

**Majority vote** — the answer that appears most often wins. Simple and robust.

**Weighted vote** — each vote is multiplied by its confidence score. Higher-confidence agents have more influence. Useful when agents have calibrated confidence.

**Consensus strength:**
- Strong: ≥80% agreement
- Moderate: ≥60% agreement
- Weak: <60% — disagreement is itself informative

**Compared to Debate (exp 06):**

| Debate | Consensus Voting |
|---|---|
| Adversarial: agents argue opposing sides | Independent: agents don't interact |
| Judge synthesizes arguments | Aggregator counts votes |
| Best for: tradeoff decisions | Best for: factual questions, classification |

---

## Running the Experiment

```bash
uv run python multi-agent/07-consensus-voting/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/07-consensus-voting/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/07-consensus-voting/demo.py --real --voters 3
```

---

*Previous: [Debate](../06-debate/) · Next: [Peer Review](../08-peer-review/)*
