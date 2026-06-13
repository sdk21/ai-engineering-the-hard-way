# Lesson: Self-Reflection

**Vertical:** Planning | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## What This Teaches

Self-reflection is an iterative improvement loop where a model critiques and revises its own output:

```
Task → [Drafter] → Draft
              ↓
         [Critic] → Critique
              ↓
         "NO_ISSUES"? → Done ✓
              ↓ No
         [Reviser] → Revised draft → (repeat)
```

---

## Key Concepts

**Generation vs. critique** — models are better at *finding* errors than *avoiding* them. A dedicated critique pass catches mistakes that slipped through in generation mode.

**Termination signal** — the critic uses a structured `NO_ISSUES` signal to stop the loop. Without this, reflection runs indefinitely.

**Separate prompts** — the drafter, critic, and reviser use different system prompts and different perspectives. The critic is deliberately adversarial.

**Round budget** — always set a `max_rounds` to prevent runaway loops. In practice, 2-3 rounds is usually sufficient.

---

## Example

**Task:** "Explain why the sky is blue for a 10-year-old."

**Round 1 Draft:** "The sky is blue because of Rayleigh scattering..."
**Critique:** Uses jargon a 10-year-old won't understand.

**Round 2 Draft:** "The sky looks blue because of the way sunlight plays with the air..."
**Critique:** `NO_ISSUES` — accepted.

---

## Variants

| Variant | Description |
|---|---|
| Single-model | Same model generates and critiques (this experiment) |
| Multi-model | Separate models for generator and critic |
| Constitutional AI | Critic checks against a fixed list of principles |
| Self-consistency | Generate N drafts, pick best (see exp 02) |

---

## Running the Experiment

```bash
uv run python planning/08-self-reflection/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/08-self-reflection/demo.py --real
ANTHROPIC_API_KEY=sk-... uv run python planning/08-self-reflection/demo.py --real --rounds 4 --verbose
```

---

*Previous: [Plan-and-Execute](../07-plan-and-execute/) · Next: [Tree of Thoughts](../09-tree-of-thoughts/)*
