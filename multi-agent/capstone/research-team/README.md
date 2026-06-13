# Capstone: Research Team

**Vertical:** Multi-Agent | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

A collaborative research pipeline combining six multi-agent techniques into a single system:

| Technique | Source | Role |
|---|---|---|
| Orchestrator | exp 02 | Decomposes topic into sub-questions |
| Parallel Fan-out | exp 03 | 3 specialist researchers run simultaneously |
| Shared Blackboard | exp 09 | All agents read/write to a shared state |
| Critic Agent | exp 05 | Reviews draft report for gaps and errors |
| Consensus Voting | exp 07 | Validates key claims with confidence scores |
| Human-in-the-Loop | exp 12 | Optional review checkpoint before delivery |

---

## Architecture

```
Topic: "The rise of WebAssembly in web development"
  │
[Orchestrator] → 3 sub-questions (technical / business / contextual)
  │
  ├──→ [Technical Researcher]  ──→ blackboard['technical']
  ├──→ [Business Researcher]   ──→ blackboard['business']    (parallel)
  └──→ [Context Researcher]    ──→ blackboard['context']
  │
[Drafter] reads full blackboard → draft report
  │
[Critic] → issues? → [Reviser] → improved report
  │
[Consensus] → validates 2-3 key claims with confidence scores
  │
[HITL Checkpoint] (optional) → human reviews → approve/edit/reject
  │
Final Report
```

---

## Example Output

```
[Orchestrator] Decomposing research topic...
[Research Team] Researching in parallel...
[Blackboard] Three research sections written
[Drafter] Writing draft report...
[Critic] Reviewing draft...
[Consensus] Validating key claims...
[Done] Final report ready

Final Report:
  WebAssembly (Wasm) is a binary instruction format that runs at near-native
  speed inside browser VMs, designed as a compilation target for C++, Rust,
  and Go. Benchmarks show 10-50× speedups for compute-heavy workloads.
  Figma, Google Earth, and AutoCAD Web rely on Wasm for rendering...
```

---

## Running the Capstone

```bash
uv run python multi-agent/capstone/research-team/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/capstone/research-team/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/capstone/research-team/demo.py --real --hitl --verbose
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/capstone/research-team/demo.py --real --topic 2
```

**Note:** Makes ~10-12 API calls. Parallel execution keeps wall time to ~10 seconds.

---

*This is the capstone for the Multi-Agent vertical. The next vertical is: RAG.*
