# Lesson: LLM-as-Judge

**Vertical:** Eval | **Difficulty:** Beginner | **Status:** 🔜 Coming Soon

---

## Table of Contents

1. [The Problem: How Do You Evaluate Open-Ended Outputs?](#1-the-problem-how-do-you-evaluate-open-ended-outputs)
2. [What Is LLM-as-Judge?](#2-what-is-llm-as-judge)
3. [The Basic Pattern](#3-the-basic-pattern)
4. [Rubric Design](#4-rubric-design)
5. [Evaluation Modes](#5-evaluation-modes)
6. [Pairwise vs. Absolute Scoring](#6-pairwise-vs-absolute-scoring)
7. [Calibrating the Judge](#7-calibrating-the-judge)
8. [Bias in LLM Judges](#8-bias-in-llm-judges)
9. [When LLM-as-Judge Works and When It Doesn't](#9-when-llm-as-judge-works-and-when-it-doesnt)
10. [Failure Modes](#10-failure-modes)
11. [Key Principles](#11-key-principles)
12. [In the Real World](#12-in-the-real-world)
13. [Running the Experiment](#13-running-the-experiment)

---

## 1. The Problem: How Do You Evaluate Open-Ended Outputs?

Traditional software testing is binary: a function either returns the right value or it doesn't. LLM outputs aren't like that. Consider evaluating whether an LLM's summary of a document is "good":

- Is it accurate?
- Is it complete?
- Is it appropriately concise?
- Is the tone right?
- Did it hallucinate anything?

There is no single ground-truth string to compare against. Multiple summaries could all be correct and varying in quality. Human raters would agree that some summaries are better than others, but capturing that judgment programmatically is hard.

This is the core challenge of LLM evaluation: **how do you measure quality for tasks where quality is subjective or multidimensional?**

---

## 2. What Is LLM-as-Judge?

LLM-as-Judge uses a (typically stronger or more carefully prompted) LLM to evaluate the output of another LLM. The judge model reads the output and produces a score, label, or critique.

```
System under test:
  Input:  "Summarize the French Revolution in 2 sentences."
  Output: "The French Revolution began in 1789 and overthrew the monarchy.
           It established republican ideals of liberty and equality in France."

Judge LLM:
  Input:  [original question] + [model output] + [evaluation rubric]
  Output: Score: 4/5. The summary is accurate and appropriately brief. 
          Minor issue: doesn't mention the Reign of Terror or Napoleon.
```

The judge provides a score and (optionally) a rationale. The rationale is useful for diagnosing failure patterns — you learn not just *that* the model failed but *why*.

---

## 3. The Basic Pattern

```python
JUDGE_PROMPT = """
You are an expert evaluator. Rate the following response on a scale of 1-5.

Question: {question}
Response: {response}

Evaluate on:
- Accuracy (is the information correct?)
- Completeness (does it answer the full question?)
- Clarity (is it easy to understand?)

Return JSON: {{"score": <1-5>, "rationale": "<one sentence>"}}
"""

def judge(question: str, response: str) -> dict:
    result = llm(JUDGE_PROMPT.format(question=question, response=response))
    return parse_json(result)
```

The judge prompt is the key artifact. Writing a good judge prompt is a skill analogous to writing a good test suite.

---

## 4. Rubric Design

A **rubric** makes the evaluation criteria explicit and consistent. Without a rubric, different judge invocations will apply different implicit standards.

A good rubric:
- Names each dimension of quality separately (accuracy, completeness, tone, etc.)
- Defines what each score level means with specific examples
- Specifies what to do in edge cases (what counts as a hallucination? what if the question is ambiguous?)

Example rubric for a customer support response:

```
Accuracy (1-5):
  1 — Contains factual errors
  2 — Mostly correct with minor inaccuracies
  3 — Correct but incomplete
  4 — Correct and sufficiently complete
  5 — Correct, complete, and cites relevant policy

Tone (1-5):
  1 — Rude or dismissive
  2 — Neutral but cold
  3 — Appropriately professional
  4 — Warm and professional
  5 — Empathetic and perfectly calibrated to the customer's emotional state
```

Detailed rubrics produce more consistent and actionable judge outputs.

---

## 5. Evaluation Modes

**Reference-free** — The judge evaluates the response with no "correct answer" to compare against. Useful when no ground truth exists. Prone to the judge using its own priors rather than external standards.

```
Judge: "Is this a good response to [question]?"
```

**Reference-based** — The judge compares the response against a reference answer (human-written ground truth). More consistent but requires curating reference answers.

```
Judge: "Compare this response to the reference answer. Is the model's response
        equally correct, better, or worse?"
```

**Context-based** — For RAG systems, the judge evaluates whether the response is grounded in the provided context (faithfulness) and whether it answers the question (relevance).

```
Judge: "Given this context: [retrieved docs], does the model's response accurately
        represent the information in the context without adding unsupported claims?"
```

---

## 6. Pairwise vs. Absolute Scoring

**Absolute scoring** — The judge assigns a numeric score (1–5, 0–10, etc.) to a single response.

```
Response: "..."
Score: 3/5
```

**Pairwise comparison** — The judge compares two responses and picks the better one.

```
Response A: "..."
Response B: "..."
Which is better? A
```

Pairwise comparison is generally more reliable than absolute scoring because:
- Relative judgments ("A is better than B") are easier than absolute ones ("A is a 3.7 out of 5")
- It avoids calibration drift (where the judge's sense of "3/5" shifts over time)
- It mirrors how human preference data (RLHF) is collected

The downside: N responses require N(N-1)/2 pairwise comparisons — O(N²) judge calls. For large eval sets, this is expensive.

---

## 7. Calibrating the Judge

A judge that gives every response a 4/5 is useless even if it's consistent. You need to calibrate it:

**Anchor examples** — Include examples of responses that should score 1, 3, and 5 in the judge prompt. These anchors prevent the judge from clustering around the middle of the scale.

**Agreement with humans** — Sample a set of responses, have both humans and the judge score them, and compute agreement (Cohen's kappa, Spearman correlation). A well-calibrated judge should correlate at ≥0.7 with human raters.

**Consistency checks** — Present the same response multiple times with shuffled wording. The judge's scores should be consistent (low variance). High variance indicates the judge is relying on surface features rather than meaning.

**Cross-judge agreement** — Run the same eval with two different judge models. High disagreement signals that the rubric is underspecified or the task is genuinely ambiguous.

---

## 8. Bias in LLM Judges

LLM judges have well-documented systematic biases:

**Verbosity bias** — Longer responses are rated higher, even when additional length adds no value. A terse, accurate answer may score lower than a verbose, meandering one.

**Position bias** — In pairwise comparisons, the judge favors whichever response appears first (primacy bias) or last (recency bias). Mitigation: always evaluate both orderings (A vs. B and B vs. A) and average.

**Self-preference bias** — A judge model tends to prefer responses stylistically similar to its own outputs. Using GPT-4 as a judge may favor GPT-4-like responses over responses from other models.

**Sycophancy** — If the judge prompt includes preambles like "this is an excellent question," the judge may rate responses more generously. Keep judge prompts neutral.

**Authority bias** — The judge may penalize responses that contradict its own training knowledge, even when the response is correct and the judge's knowledge is outdated.

Knowing these biases doesn't eliminate them, but it guides mitigation strategies: position shuffling, multi-model judges, calibration datasets, and rubrics that explicitly penalize length over substance.

---

## 9. When LLM-as-Judge Works and When It Doesn't

**Works well for:**
- Evaluating fluency, coherence, and readability
- Checking for hallucinations against provided context (faithfulness)
- Detecting harmful or inappropriate content
- Comparing relative quality when exact correctness is not the measure
- Dimensions where human raters agree but articulating rules is hard (tone, helpfulness)

**Works poorly for:**
- Tasks with objective ground truth (math, code correctness, factual accuracy) — use exact match or execution-based eval instead
- Evaluating responses in domains where the judge lacks expertise (niche technical fields, specialized law)
- High-stakes decisions where the judge's own biases could cause systematic errors
- Very long outputs where the judge loses attention or becomes inconsistent

---

## 10. Failure Modes

**Judge hallucination** — The judge confidently claims the response contains an error that it doesn't, or misses a real error. Judges are LLMs — they hallucinate too.

**Rubric underspecification** — The rubric doesn't cover a situation that arises. The judge makes up criteria, producing inconsistent scores.

**Distribution shift** — The judge is calibrated on one distribution of responses, then applied to a different one (e.g., calibrated on English responses, applied to bilingual responses).

**Score inflation** — Judges tend to be lenient over time without recalibration, especially if they are exposed to a high proportion of good responses. The "average score" slowly drifts upward.

**Circular eval** — The same model is both the system under test and the judge. The model will tend to prefer its own outputs, inflating self-evaluation scores.

---

## 11. Key Principles

> **Principle 1 — The judge is a model, not a ground truth.**
> LLM judges have biases, failure modes, and hallucinations just like the systems they evaluate. Treat judge scores as noisy estimates of quality, not facts.

> **Principle 2 — Invest in rubric design.**
> A bad rubric produces noisy, inconsistent scores that mislead rather than inform. The rubric is the spec for what you're measuring — it deserves as much care as the system prompt.

> **Principle 3 — Validate the judge before trusting it.**
> Compute human-judge agreement on a sample before using judge scores to drive decisions. An un-validated judge may be measuring something other than what you think.

> **Principle 4 — Use the right eval for the task.**
> LLM-as-judge is powerful for subjective quality dimensions. For tasks with objective answers, exact match, execution-based eval, or unit tests are more reliable and cheaper.

> **Principle 5 — Eval is a product, not a one-time setup.**
> Evaluation suites need maintenance as the system changes. What counts as "good" evolves with requirements. Treat your eval suite with the same care as production code.

---

## 12. In the Real World

**Chatbot Arena (LMSYS)**
Chatbot Arena is the largest public LLM evaluation platform. Users submit prompts, receive two anonymous responses, and pick the better one — pairwise human evaluation at scale. The LLM-as-judge literature frequently validates automated judges against Chatbot Arena's human preferences.

**OpenAI RLHF / Constitutional AI (Anthropic)**
Both OpenAI's RLHF and Anthropic's Constitutional AI use a form of LLM-as-judge. In Constitutional AI, the model evaluates its own outputs against a set of principles and revises them — a self-judging loop. The judge produces preference labels that train the reward model.

**LangChain — `QAEvalChain`**
LangChain provides `QAEvalChain` which implements LLM-as-judge for question-answering tasks. You provide questions, reference answers, and model outputs; it uses an LLM to evaluate whether each answer is correct. It is the primary built-in eval primitive in LangChain.

**LlamaIndex — `LLMRelevancyEvaluator`**
LlamaIndex's evaluators (`FaithfulnessEvaluator`, `RelevancyEvaluator`, `CorrectnessEvaluator`) are all LLM-as-judge implementations. They are particularly designed for RAG eval: checking whether answers are grounded in retrieved context.

**Ragas**
Ragas is an open-source framework specifically for RAG evaluation. Its metrics — faithfulness, answer relevancy, context precision, context recall — are all implemented as LLM-as-judge prompts. It has become a standard benchmarking tool for RAG pipelines.

**Braintrust**
Braintrust is a production eval platform that uses LLM-as-judge at scale. It provides pre-built judges for common dimensions (factuality, tone, safety) and tooling for running eval suites in CI/CD pipelines. It is representative of how production teams operationalize continuous evaluation.

**Promptfoo**
Promptfoo is an open-source tool for LLM evaluation and red-teaming. Its `llm-rubric` assertion type implements LLM-as-judge with custom rubrics. Engineers use it to run regression tests on prompts as part of their development workflow.

**Anthropic model evaluation**
Anthropic uses LLM-as-judge as one component of evaluating Claude during training and post-deployment. Claude is evaluated by other models (and humans) on dimensions including helpfulness, harmlessness, and honesty — the same dimensions tested in Constitutional AI.

---

## 13. Running the Experiment

```bash
# From the project root (experiment coming soon)

uv run python eval/01-llm-as-judge/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python eval/01-llm-as-judge/demo.py --real
```

**Planned exercises:**
1. Write a judge for a summarization task. Run it on 5 summaries of varying quality and verify the scores match your intuition.
2. Test position bias: swap the order of two responses in a pairwise comparison and check if the winner flips.
3. Compare judge scores from two different models (e.g., Haiku vs. Sonnet). Measure their agreement.
4. Build a simple eval pipeline: generate 10 responses, judge them all, compute the average score, then change your system prompt and re-run to see if the score improves.

---

*Next experiment: Regression Testing with Eval Suites (coming soon)*
