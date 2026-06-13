# Lesson: Task Auction

**Vertical:** Multi-Agent | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

Tasks are auctioned to the most capable agent. Each agent self-assesses and submits a capability bid. The highest bidder wins and executes the task.

```
Tasks: [analyze sales data, write Python fn, write API docs, compare databases]

       data_agent  code_agent  writing_agent  research_agent
t1:       9.5         5.0          3.0            6.0      ← data_agent wins
t2:       3.0         9.8          2.0            3.5      ← code_agent wins
t3:       4.0         5.5          9.7            6.0      ← writing_agent wins
t4:       7.0         6.0          6.5            9.6      ← research_agent wins
```

---

## Key Concepts

**Self-assessment** — agents honestly evaluate their own capability for each task. A good bidder bids high where it's strong and low where it's not, maximizing win rate on suitable tasks.

**Greedy assignment** — each task goes to the highest bidder. Simple and fast. More sophisticated variants (Hungarian algorithm) optimize global assignment.

**Parallel bidding** — all agents bid on all tasks simultaneously. `N_agents × N_tasks` API calls in parallel.

**No pre-assignment** — unlike the router (exp 01) where routing logic is hard-coded, auctions are dynamic. Add a new agent with new specialties and it will naturally win appropriate tasks.

**Contract Net Protocol** — the formal name for this pattern from 1980s multi-agent systems (FIPA CNP). Still used in robotics and distributed systems.

---

## Running the Experiment

```bash
uv run python multi-agent/11-task-auction/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/11-task-auction/demo.py --real
```

---

*Previous: [Swarm](../10-swarm/) · Next: [Human-in-the-Loop](../12-human-in-the-loop/)*
