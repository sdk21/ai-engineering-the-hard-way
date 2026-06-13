# Lesson: Debate

**Vertical:** Multi-Agent | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## What This Teaches

Two agents argue opposing positions while a judge evaluates the arguments and delivers a reasoned verdict.

```
Topic: "Should a startup use microservices from day one?"

[PRO] Opening argument → "Independent scaling, modern tooling..."
[CON] Rebuttal        → "Operational complexity, Shopify used a monolith..."
[PRO] Response        → "PaaS removes K8s overhead; refactoring is painful..."
[JUDGE]               → Winner: CON — "operational burden is real and well-documented"
```

---

## Key Concepts

**Adversarial structure** — each agent is instructed to make the strongest possible case for their position. This surfaces arguments a neutral agent might soften or omit.

**Rebuttal** — the CON agent sees the PRO argument before responding. This produces genuine back-and-forth rather than two independent monologues.

**Independent judge** — the judge has a different system prompt with explicit evaluation criteria. It weighs argument quality, not just who sounds more confident.

**When to use:**
- Technology or architecture decisions where both options are defensible
- Policy questions with legitimate tradeoffs
- Anywhere you want to stress-test a recommendation before committing

---

## Running the Experiment

```bash
uv run python multi-agent/06-debate/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/06-debate/demo.py --real --debate 1
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/06-debate/demo.py --real --debate 2 --rounds 1
```

---

*Previous: [Critic Agent](../05-critic-agent/) · Next: [Consensus Voting](../07-consensus-voting/)*
