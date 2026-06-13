# Lesson: Shared Blackboard

**Vertical:** Multi-Agent | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## What This Teaches

A blackboard is a shared data structure visible to all agents. Each agent reads what others have written and contributes to its own section. Unlike a pipeline, every agent sees ALL prior contributions.

```
Blackboard: { topic, facts, analysis, critique, report }

[researcher] reads: topic            → writes: facts
[analyst]    reads: topic + facts    → writes: analysis
[critic]     reads: topic + facts + analysis → writes: critique
[synthesizer] reads: everything      → writes: report
```

---

## Key Concepts

**Global visibility** — every agent sees the full blackboard before writing. The analyst can reference specific facts; the critic can challenge specific analysis claims.

**Write isolation** — each agent writes only to its designated section. This prevents agents from overwriting each other's work.

**Richer context than pipelines** — in a sequential pipeline, each stage only sees the previous stage's output. On a blackboard, the synthesizer sees the raw facts, the analysis, AND the critique simultaneously.

**Classical origin** — the Blackboard Architecture was used in 1970s AI systems (HEARSAY speech recognition). Multiple "knowledge sources" collaborated on a shared partial solution. The LLM version is the same pattern with prompts instead of symbolic rules.

---

## Running the Experiment

```bash
uv run python multi-agent/09-shared-blackboard/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/09-shared-blackboard/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/09-shared-blackboard/demo.py --real --verbose
```

---

*Previous: [Peer Review](../08-peer-review/) · Next: [Swarm](../10-swarm/)*
