# Lesson: Chain-of-Thought Prompting

**Vertical:** Planning | **Difficulty:** Beginner | **Status:** ✅ Ready

---

## Table of Contents

1. [The Problem: LLMs Rush to Answers](#1-the-problem-llms-rush-to-answers)
2. [What Is Chain-of-Thought?](#2-what-is-chain-of-thought)
3. [Zero-Shot CoT](#3-zero-shot-cot)
4. [Few-Shot CoT](#4-few-shot-cot)
5. [Why CoT Works](#5-why-cot-works)
6. [Structured CoT Formats](#6-structured-cot-formats)
7. [When CoT Helps and When It Doesn't](#7-when-cot-helps-and-when-it-doesnt)
8. [CoT Faithfulness and Confabulation](#8-cot-faithfulness-and-confabulation)
9. [Failure Modes](#9-failure-modes)
10. [Key Principles](#10-key-principles)
11. [In the Real World](#11-in-the-real-world)
12. [Running the Experiment](#12-running-the-experiment)

---

## 1. The Problem: LLMs Rush to Answers

LLMs trained on next-token prediction have a strong tendency to produce plausible-sounding answers immediately, without working through the reasoning. This works fine for factual recall ("What is the capital of France?") but breaks down for multi-step problems:

```
Question: "If Alice is twice as old as Bob, and Bob is 3 years older than Carol 
           who is 10, how old is Alice?"

Without CoT:
  Model → "Alice is 26."  (wrong — produced a number without checking work)

With CoT:
  Model → "Carol is 10. Bob is 10 + 3 = 13. Alice is 2 × 13 = 26." (correct)
```

In the second case, producing the reasoning tokens forces the model to work through the problem step by step before committing to an answer — and it happens to get the right answer. The chain of reasoning is both the mechanism and the output.

---

## 2. What Is Chain-of-Thought?

Chain-of-thought (CoT) is a prompting technique that elicits step-by-step reasoning from a model before it produces a final answer. The model "thinks out loud," and the intermediate reasoning steps serve as scaffolding for the final answer.

The key insight from the original 2022 Google paper (Wei et al.) was that larger models, when prompted to show reasoning, dramatically outperformed smaller models on multi-step benchmarks — and even matched or exceeded the performance of models 5–10× larger that were prompted directly.

CoT is not fine-tuning. It is purely a prompting technique: you change the prompt, not the model.

---

## 3. Zero-Shot CoT

The simplest form of CoT requires no examples. You simply append a reasoning trigger to your prompt:

```
Prompt: "Solve this step by step: [question]"
Prompt: "Let's think through this carefully: [question]"
Prompt: "Think step by step before answering: [question]"
```

The original paper used **"Let's think step by step"** as the trigger phrase. This phrase became so strongly associated with CoT that it became almost a magic incantation for eliciting reasoning — the model has seen it associated with careful reasoning in its training data, so it activates that pattern.

Modern models (GPT-4, Claude 3+) have CoT behavior so deeply baked in that extensive triggers are often unnecessary. But explicit triggers still help for harder problems.

---

## 4. Few-Shot CoT

Few-shot CoT provides worked examples in the prompt itself:

```
Example 1:
Q: "Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many?"
A: "Roger starts with 5 balls. 2 cans × 3 balls = 6 more balls. 5 + 6 = 11 balls."

Example 2:
Q: "The cafeteria had 23 apples. They used 20 for lunch and bought 6 more. How many?"
A: "Start: 23. Used: 23 - 20 = 3. Bought: 3 + 6 = 9 apples."

Now answer:
Q: [new question]
A: [model generates step-by-step reasoning]
```

Few-shot CoT is more reliable than zero-shot for complex domains because:
- It shows the model the *format* of reasoning you want
- It implicitly specifies granularity (how many steps, how much detail)
- It can encode domain-specific reasoning patterns (e.g., how to set up a physics problem)

The downside is prompt length — 3–5 worked examples can add hundreds of tokens to every request.

---

## 5. Why CoT Works

The mechanistic explanation is still debated in the research community, but several factors contribute:

**Computation budget** — Each token the model generates is one step of "computation." Forcing the model to generate reasoning tokens before the answer gives it more computational steps to arrive at a correct answer. Direct answers compress all reasoning into zero tokens.

**Training signal** — Models are trained on human-written text, which frequently shows reasoning before conclusions (essays, proofs, worked examples, tutorials). CoT triggers activate these reasoning-rich patterns.

**Error checking** — Writing out intermediate steps surfaces opportunities for self-correction. If step 3 contradicts step 2, the model may catch it in step 4 in a way it couldn't if jumping to the conclusion.

**Reduced search space** — In multi-step problems, committing to sub-conclusions early (even if they turn out to be wrong) reduces the branching factor for subsequent steps. The reasoning acts as a constraint on subsequent generation.

---

## 6. Structured CoT Formats

Beyond free-form reasoning, several structured formats have emerged:

**XML tags (Claude-specific)**
```
<thinking>
Let me work through this step by step...
[reasoning]
</thinking>
[final answer]
```

**ReAct format (Thought / Action / Observation)**
```
Thought: I need to find the population of Tokyo.
Action: search("Tokyo population")
Observation: Tokyo has a population of 13.96 million.
Thought: Now I can answer.
Answer: Tokyo's population is approximately 14 million.
```

**Scratchpad**
```
Scratchpad:
- Given: ...
- Step 1: ...
- Step 2: ...
Final Answer: ...
```

Structured formats make the reasoning machine-parseable and enable downstream systems to inspect, log, or interrupt the reasoning process.

---

## 7. When CoT Helps and When It Doesn't

**CoT helps with:**
- Multi-step arithmetic and algebra
- Logical reasoning and deduction
- Tasks requiring planning or decomposition
- Problems where intermediate results constrain later steps
- Anything where "showing work" is part of the correct answer

**CoT doesn't help (or hurts) with:**
- Simple factual recall ("What year did WWII end?")
- Tasks that are inherently parallel rather than sequential
- Very short outputs where the reasoning overhead exceeds the answer length
- Tasks where reasoning may confabulate false justifications for a wrong answer
- Fast, latency-sensitive applications where the extra tokens slow the response

The general rule: if a human expert would need to "show their work" to solve it correctly, CoT probably helps. If a human expert answers in one second without a scratch pad, it probably doesn't.

---

## 8. CoT Faithfulness and Confabulation

A critical issue: **the model's stated reasoning may not reflect its actual internal computation.**

Experiments have shown that LLMs sometimes produce incorrect reasoning chains that lead to correct answers (getting lucky), or correct-looking reasoning chains that lead to wrong answers (reasoning failure), or plausible reasoning chains post-hoc rationalized from an already-chosen answer (confabulation).

The chain of thought is itself a generation — it is subject to the same hallucination risks as any other model output. You cannot blindly trust that the reasoning is valid just because it sounds coherent.

This doesn't mean CoT is useless — in practice it substantially improves accuracy on hard tasks. But it does mean you should treat CoT outputs as "reasoning attempts," not proofs.

---

## 9. Failure Modes

**Reasoning-answer mismatch** — The model produces correct reasoning but then states a different answer that contradicts it. The answer generation sometimes ignores the preceding CoT.

**Verbose but empty reasoning** — The model generates many words of reasoning that don't actually constrain the answer. "This is a complex problem that requires careful analysis..." followed by a guess.

**Compounding errors** — In long chains, an error in step 3 propagates through steps 4, 5, 6... making the final answer systematically wrong in a way that's hard to detect.

**Overthinking simple problems** — CoT on trivial tasks can make the model second-guess correct instincts, degrading accuracy on problems where direct response would have been fine.

---

## 10. Key Principles

> **Principle 1 — Tokens are computation.**
> Generating reasoning tokens before the answer gives the model more "compute" to work with. This is the core mechanism of CoT improvement.

> **Principle 2 — Show the format you want.**
> Few-shot examples don't just help with the problem — they calibrate reasoning granularity, format, and domain conventions.

> **Principle 3 — CoT is a prompt, not a guarantee.**
> The model can produce plausible-looking wrong reasoning. Validate the conclusions, not just the reasoning chain.

> **Principle 4 — Match technique to task.**
> CoT has real token and latency cost. Use it when the task complexity warrants it; skip it for simple lookups and generation tasks.

---

## 11. In the Real World

**OpenAI o1 / o3 — "thinking" models**
OpenAI's o-series models perform extended internal chain-of-thought reasoning before producing their response. This reasoning is hidden from the user (shown as a collapsed "thinking" block) but is the primary source of their superior performance on math, coding, and science benchmarks. The entire o-series is essentially "CoT trained to be fast and accurate."

**Anthropic Claude — extended thinking**
Claude supports an "extended thinking" mode where it generates a long chain-of-thought reasoning block before its response. This is explicitly a CoT mechanism, and Anthropic's research shows it substantially improves performance on hard reasoning tasks.

**Google Gemini — thinking mode**
Gemini 2.0 Flash Thinking implements the same pattern: visible reasoning tokens before the final answer. Google's research traces this to the AlphaCode and Minerva work showing that generating "scratchpad" tokens improves code and math problem solving.

**DeepSeek-R1**
DeepSeek's R1 model was trained with reinforcement learning to produce long chains of thought. Its reasoning traces are unusually explicit and verbose — sometimes thousands of tokens of step-by-step work. It became notable for showing that open-source models could match proprietary performance with CoT training.

**LangChain — `LLMChain` with reasoning templates**
LangChain provides prompt templates that inject CoT instructions. Their "Structured Output" chains use CoT to produce parseable JSON reasoning before a final structured answer — a common pattern for extraction tasks.

**Wolfram Alpha integration**
When LLMs integrate with Wolfram Alpha, the query formulation step often uses CoT: "I need to compute X. To use Wolfram, I should phrase this as Y." The tool use itself is guided by the chain of thought.

**Medical / Legal AI (Harvey, Nabla)**
High-stakes domains use CoT as an audit trail. Showing reasoning is not just about accuracy — it is legally and ethically important that AI recommendations come with visible justification. CoT makes the model's reasoning inspectable.

---

## 12. Running the Experiment

```bash
# From the project root (experiment coming soon)

uv run python planning/01-chain-of-thought/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python planning/01-chain-of-thought/demo.py --real
```

**Planned exercises:**
1. Compare direct vs. CoT prompting on the same set of math word problems — measure accuracy difference.
2. Try zero-shot CoT ("think step by step") vs. few-shot CoT on the same problem.
3. Deliberately give the model a problem where CoT hurts (simple recall) and observe the difference.
4. Try to construct a case where the reasoning chain is wrong but the final answer is correct.

---

*Next experiment: [Zero-Shot CoT](../02-zero-shot-cot/)*
