# Lesson: Hierarchical Multi-Agent

**Vertical:** Multi-Agent | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

Agents organized in tiers of authority. The executive delegates to managers; managers delegate to workers. Each level operates at a different scope.

```
[CTO] "Build user authentication"
  │
  ├── [Backend Lead] "Implement JWT endpoints"
  │     ├── [API Developer]      → /auth/login + /register with JWT
  │     └── [DB Developer]       → users table + indexes
  │
  ├── [Frontend Lead] "Build login UI with secure token handling"
  │     ├── [UI Developer]       → LoginForm component with validation
  │     └── [State Developer]    → httpOnly cookie auth + route guards
  │
  └── [QA Lead] "Ensure auth is tested and secure"
        ├── [Test Engineer]      → 12 integration tests, 100% coverage
        └── [Security Reviewer]  → Rate limiting, bcrypt ≥12, HTTPS-only
```

---

## Key Concepts

**Authority tiers** — the CTO doesn't know individual task details; the API developer doesn't know the frontend plan. Each level operates at the right scope.

**Parallel team execution** — all three teams (backend, frontend, QA) run concurrently. Workers within each team also run in parallel. This is a 2-level fan-out.

**Compared to flat Orchestrator (exp 02):**

| Orchestrator | Hierarchical |
|---|---|
| 1 orchestrator → N workers | Executive → Managers → Workers |
| Single coordination layer | Multiple coordination layers |
| Works for 3-5 subtasks | Scales to 20+ tasks without bottleneck |
| Simple | More complex, but more scalable |

**Reporting up** — results flow upward: workers → manager summary → executive report. Each level synthesizes before reporting.

---

## Running the Experiment

```bash
uv run python multi-agent/13-hierarchical/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/13-hierarchical/demo.py --real --goal 1
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/13-hierarchical/demo.py --real --goal 2
```

**Note:** Makes ~10 API calls total (1 executive + 3 managers + 6 workers). Parallel execution keeps wall time reasonable.

---

*Previous: [Human-in-the-Loop](../12-human-in-the-loop/) · Next: [Capstone: Research Team](../capstone/research-team/)*
