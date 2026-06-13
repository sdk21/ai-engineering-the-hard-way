# Lesson: ReAct (Reasoning + Acting)

**Vertical:** Planning | **Difficulty:** Beginner–Intermediate | **Status:** ✅ Ready

---

## What This Teaches

ReAct (Yao et al., 2022) interleaves explicit **Thought** steps with **Action** (tool) calls and **Observation** (tool result) steps. The model alternates: reason about what to do → do it → observe the result → reason again.

```
Question: Is Mount Everest taller than the cruising altitude of commercial aircraft?

Thought: I need the height of Mount Everest first.
Action: search[mount everest height]
Observation: Mount Everest is 8,849 meters (29,032 feet).

Thought: Now I need the cruising altitude of aircraft.
Action: search[commercial aircraft cruising altitude]
Observation: Commercial aircraft cruise at 35,000–42,000 feet.

Thought: Everest is 29,032 ft. Aircraft fly at 35,000+ ft. Aircraft fly higher.
Final Answer: No — Everest (29,032 ft) is below commercial cruising altitude (35,000–42,000 ft).
```

---

## ReAct vs. Basic Tool Use

| | Basic Tool Use (tools/01) | ReAct |
|--|--------------------------|-------|
| Reasoning | Implicit — inside the model | Explicit — written as Thought steps |
| Traceability | Black box | Every step is readable |
| Debuggability | Hard — you only see actions | Easy — you see why each action was taken |
| Format | API-native tool_use blocks | Text-based Thought/Action/Observation |

ReAct is particularly useful when you need an audit trail of the model's reasoning, or when you're building the planning layer before connecting real tools.

---

## The Trace as State

The accumulated Thought/Action/Observation trace is the agent's state. Each prompt includes the full trace so far, and the model continues from where it left off. This is how the agent "remembers" what it has already done and learned.

---

## Key Principles

> **Principle 1 — Thought before Action.** Forcing a Thought step before every Action prevents the model from taking actions it hasn't reasoned about. The Thought step is cheap insurance.

> **Principle 2 — Observations ground reasoning.** Without real observations, the model hallucinates. The Observation step is where real information enters the loop.

> **Principle 3 — Traces are debugging tools.** The full Thought/Action/Observation trace lets you see exactly why the model did what it did. This is invaluable for improving prompts and fixing errors.

---

## Running the Experiment

```bash
uv run python planning/03-react/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/03-react/demo.py --real
```

**Exercises:**
1. Ask "Which is longer — the Amazon or the Nile?" and trace the full reasoning path.
2. Remove the `REACT_FEW_SHOT` example from the prompt and observe whether reasoning quality degrades.
3. Add a `finish[answer]` action format instead of `Final Answer:` and update the parser.

---

*Previous: [Zero-Shot CoT](../02-zero-shot-cot/) · Next: [Task Decomposition](../04-task-decomposition/)*
