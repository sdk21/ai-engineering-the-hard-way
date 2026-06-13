# Lesson: Hierarchical Planning

**Vertical:** Planning | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## What This Teaches

Hierarchical planning decomposes a goal into progressively more concrete levels:

```
Level 0: Goal
└── Level 1: Sub-goals (phases / workstreams)
    └── Level 2: Tasks (concrete activities)
        └── Level 3: Actions (specific steps)  ← optional
```

Each level is planned independently. Level-1 planning focuses on "what major phases are needed?" Level-2 planning focuses on "what concrete tasks does this phase require?" — without worrying about other phases.

---

## Key Concepts

**Abstract-to-concrete refinement** — plan at high abstraction first; each node becomes a sub-plan when you zoom in. A node that looks like one unit at level 1 may expand to a 10-task network at level 2.

**Focus** — level-2 planning for "Marketing" can focus entirely on marketing without the noise of the "Engineering" phase. Separate prompts per sub-goal improve quality.

**Reuse** — common sub-goals (testing, deployment, security review) appear across many plans. Recognize and reuse these patterns.

**Scope control** — stop refining when a task is atomic enough to execute directly. The `is_atomic` flag marks these leaf nodes.

**HTN (Hierarchical Task Network)** — the classical AI planning formalism this mirrors. HTN planners decompose non-primitive tasks recursively until all leaves are primitive (executable) actions.

---

## Structure

```
Goal: Launch new SaaS product
├── Sub-goal 1: Product Development           [6 weeks]
│   ├── Task 1.1: Define requirements         [3 days]  (atomic)
│   ├── Task 1.2: Build MVP                   [4 weeks]
│   └── Task 1.3: Security and perf testing   [1 week]  (atomic)
├── Sub-goal 2: Marketing & Go-to-Market      [4 weeks]
│   ├── Task 2.1: Create landing page         [1 week]  (atomic)
│   ├── Task 2.2: Write launch blog post      [3 days]  (atomic)
│   └── Task 2.3: Email campaign setup        [2 days]  (atomic)
└── Sub-goal 3: Release & Operations          [1 week]
    ├── Task 3.1: Staged rollout              [2 days]  (atomic)
    ├── Task 3.2: Monitor metrics             [ongoing] (atomic)
    └── Task 3.3: Gather user feedback        [ongoing] (atomic)
```

---

## Compared to Task Graph (exp 05)

| Task Graph | Hierarchical Planning |
|---|---|
| Flat nodes with dependency edges | Tree structure, nodes contain sub-plans |
| Shows parallelism between tasks | Shows abstraction levels |
| Best for: scheduling, critical path | Best for: complex goals, delegation |

These complement each other — a hierarchical plan defines *what* to do; a task graph defines *in what order*.

---

## Running the Experiment

```bash
uv run python planning/06-hierarchical-planning/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/06-hierarchical-planning/demo.py --real --depth 1
ANTHROPIC_API_KEY=sk-... uv run python planning/06-hierarchical-planning/demo.py --real --depth 2
```

---

*Previous: [Task Graph](../05-task-graph/) · Next: [Plan-and-Execute](../07-plan-and-execute/)*
