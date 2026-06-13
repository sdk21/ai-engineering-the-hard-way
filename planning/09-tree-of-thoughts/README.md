# Lesson: Tree of Thoughts

**Vertical:** Planning | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

Tree of Thoughts (ToT) explores multiple reasoning paths simultaneously, using an evaluator to prune dead ends and focus on promising branches.

```
Problem
├── Thought A [score: 8.5] ★ SOLUTION
├── Thought B [score: 4.0] ◉ PROMISING
│   └── Thought B1 [score: 2.0] ✗ DEAD END
└── Thought C [score: 3.0] ✗ DEAD END
```

Standard CoT follows one path. ToT branches at decision points and evaluates each branch before deciding where to go deeper.

---

## Key Concepts

**Thought generation** — at each node, the model generates `k` distinct next thoughts. Diversity is important: the same first thought repeated k times defeats the purpose.

**Evaluation** — a separate call scores each thought on a 1-10 scale. Scores below the threshold are pruned (marked DEAD_END).

**Search strategy:**

| BFS | DFS |
|---|---|
| Expand all nodes at depth d before going deeper | Follow one path to max depth, then backtrack |
| Finds shallowest solution | Finds any solution quickly |
| Better for comparing many options | Better for deep reasoning chains |

**Pruning** — low-scored nodes are pruned (not expanded). This is the core efficiency gain over exhaustive search.

**Solution checking** — at promising nodes, a verifier checks if the accumulated reasoning fully answers the question.

---

## When to Use ToT

- **Puzzles** with discrete solution states (water jug, missionaries and cannibals)
- **Proofs** where early wrong steps lead to dead ends
- **Creative writing** with hard constraints (write a poem where every line starts with 'A')
- Any task where the first plausible-sounding path is often wrong

---

## Running the Experiment

```bash
uv run python planning/09-tree-of-thoughts/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/09-tree-of-thoughts/demo.py --real --strategy bfs
ANTHROPIC_API_KEY=sk-... uv run python planning/09-tree-of-thoughts/demo.py --real --strategy dfs --depth 3 --breadth 3 --threshold 6.0
```

**Note:** ToT makes multiple API calls per node (generate + evaluate + check). Use `--depth 2 --breadth 2` to limit cost.

---

*Previous: [Self-Reflection](../08-self-reflection/) · Next: [Backtracking](../10-backtracking/)*
