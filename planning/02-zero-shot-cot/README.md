# Lesson: Zero-Shot Chain-of-Thought

**Vertical:** Planning | **Difficulty:** Beginner | **Status:** ✅ Ready

---

## What This Teaches

Zero-shot CoT adds a single trigger phrase to the prompt — no worked examples required. The model generates reasoning steps from the trigger alone.

The original paper (Kojima et al., 2022) showed that appending **"Let's think step by step."** to a question dramatically improved accuracy on math and logic benchmarks — with no examples, no fine-tuning, just eight words.

---

## The Two-Stage Pattern

A single-stage prompt can generate reasoning that contradicts the final answer. The two-stage pattern separates them:

```
Stage 1 prompt: "Q: [question]\nA: Let's think step by step."
Stage 1 output: [reasoning chain]

Stage 2 prompt: "Q: [question]\nA: [reasoning]\nTherefore, the final answer is:"
Stage 2 output: [clean final answer]
```

Stage 2 forces the model to commit to an answer *after* reading its own reasoning — reducing reasoning-answer mismatches.

---

## Self-Consistency

Sample N reasoning paths from the same prompt, extract the answer from each, take the majority vote:

```
Path 1: reasoning A → answer "5 cents"
Path 2: reasoning B → answer "5 cents"
Path 3: reasoning C → answer "10 cents"
Majority vote → "5 cents"  ✓
```

Self-consistency improves accuracy by 5–15% on hard problems, at the cost of N× the API calls. It works because even when one reasoning path fails, most succeed.

---

## Trigger Phrase Variants

| Trigger | Style |
|---------|-------|
| `"Let's think step by step."` | Original, most studied |
| `"Let's think through this carefully, step by step."` | Emphasises care |
| `"Let's solve this step by step and verify each step."` | Adds verification |
| `"As an expert, let me reason through this systematically."` | Persona framing |
| `"Let me break this problem into smaller parts."` | Decomposition framing |

Modern models (Claude 3+, GPT-4) respond well to all variants. The original is still a reliable default.

---

## Running the Experiment

```bash
uv run python planning/02-zero-shot-cot/demo.py --mock

ANTHROPIC_API_KEY=sk-... uv run python planning/02-zero-shot-cot/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python planning/02-zero-shot-cot/demo.py --real --trigger verify
ANTHROPIC_API_KEY=sk-... uv run python planning/02-zero-shot-cot/demo.py --real --self-consistency --samples 5
```

**Exercises:**
1. Run with `--trigger none` vs `--trigger original` on the trick questions and compare accuracy.
2. Try `--self-consistency --samples 3` on `p1` (bat-and-ball). Does majority vote correct the wrong paths?
3. Add a new trigger phrase and test whether it outperforms the original on any problem category.

---

*Previous: [Chain-of-Thought](../01-chain-of-thought/) · Next: [ReAct](../03-react/)*
