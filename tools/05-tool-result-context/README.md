# Lesson: Tool Result Context

**Vertical:** Tools | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [Beyond the Bare Answer](#1-beyond-the-bare-answer)
2. [What Context Fields Do](#2-what-context-fields-do)
3. [The Three Result Styles](#3-the-three-result-styles)
4. [Confidence Scores](#4-confidence-scores)
5. [Caveats and Warnings](#5-caveats-and-warnings)
6. [Provenance and Freshness](#6-provenance-and-freshness)
7. [Suggested Follow-ups](#7-suggested-follow-ups)
8. [Context Shapes Reasoning, Not Just Output](#8-context-shapes-reasoning-not-just-output)
9. [Key Principles](#9-key-principles)
10. [In the Real World](#10-in-the-real-world)
11. [Running the Experiment](#11-running-the-experiment)

---

## 1. Beyond the Bare Answer

A tool result is not just a data return — it's a reasoning artifact. The model reads every field you include, and those fields shape everything that follows: which caveats to mention, how confident to sound, whether to suggest a follow-up action.

Consider these two results for `get_population("Springfield")`:

**Bare result:**
```
Population: 114,230
```
Model response: *"Springfield has a population of 114,230."*

**Rich result:**
```
Population: 114,230 (city) — Note: multiple US cities named Springfield
source: US Census Bureau
data_freshness: 2020 census
confidence: 40% — interpret with caution
caveats:
  - Ambiguous: there are 34 Springfields in the US; data is for Springfield, IL
  - Which Springfield did you mean? MO (167k), OH (58k), MA (154k) also exist
suggested_followups:
  - Clarify which state's Springfield you mean
  - get_population('springfield, mo') — for Missouri
```
Model response: *"I found data for Springfield, IL (114,230), but I should mention there are 34 cities named Springfield in the US — this might not be the one you meant. Could you clarify which state? Missouri, Ohio, and Massachusetts each have their own Springfields with different populations."*

The underlying data is the same. The context produces a fundamentally better response.

---

## 2. What Context Fields Do

| Field | What the model does with it |
|-------|---------------------------|
| `value` | Reports the core answer |
| `confidence` | Calibrates how certain to sound; surfaces uncertainty when low |
| `source` | Attributes the data; mentions it when credibility matters |
| `freshness` | Warns when data may be stale |
| `caveats` | Surfaces limitations the model would otherwise omit |
| `suggested_followups` | Guides next tool calls or user questions |
| `data_type` | Helps the model format or interpret the value correctly |

None of these require special handling in the agentic loop. The model reads them because they're text — the same way it reads everything else.

---

## 3. The Three Result Styles

This experiment demonstrates three levels of context:

**Bare** — just the value:
```
Population: 13.96 million
```

**Simple** — value + source + confidence:
```
Population: 13.96 million
source: Statistics Bureau of Japan
confidence: 95%
```

**Rich** — full context:
```
Population: 13.96 million (city proper), 37.4 million (greater metro)
source: Statistics Bureau of Japan
data_freshness: 2023 census estimate
confidence: high
caveats:
  - City proper boundary excludes many effectively urban areas
  - Metro population definition varies by source (±5M)
suggested_followups:
  - get_population('osaka') — for comparison
  - get_urban_density('tokyo') — for density per km²
```

Run `demo.py --mock` to see all three styles side by side for the same query.

---

## 4. Confidence Scores

Confidence is one of the most impactful context fields. When the model sees a low-confidence result, it hedges appropriately:

```
confidence: 40% — interpret with caution
```

vs.

```
confidence: high
```

The model will use different language: "approximately", "the data suggests", "you may want to verify" for low confidence vs. direct statements for high confidence.

**How to set confidence:**
- **0.9–1.0** — verified data from authoritative sources
- **0.6–0.9** — reasonably reliable but may be outdated or have minor caveats
- **0.3–0.6** — unreliable, ambiguous, or from a low-quality source
- **0.0–0.3** — data is a guess or the query was ambiguous

Confidence is not a statistical probability — it's a signal to the model about how much weight to give the result. Even approximate values are useful.

---

## 5. Caveats and Warnings

Caveats are perhaps the most important context field. They allow your tool to surface known limitations that the model would otherwise omit — leading to overconfident or misleading answers.

```python
# Without caveats: the model just reports the answer
"Population: 7.10 per share"

# With caveats: the model gives a full picture
caveats = [
    "This stock has high volatility and speculative characteristics",
    "Fundamental analysis (P/E) is not applicable — company is unprofitable",
    "Price can be heavily influenced by social media sentiment",
]
```

Common caveat types:
- **Ambiguity caveats** — "Springfield refers to multiple cities"
- **Staleness caveats** — "Last updated 18 months ago"
- **Scope caveats** — "This is the US figure; global data differs"
- **Reliability caveats** — "This is modeled data, not measured"
- **Disclaimer caveats** — "Not financial advice"

---

## 6. Provenance and Freshness

Source and freshness let the model answer "how do you know that?" credibly:

```
source: Statistics Bureau of Japan
data_freshness: 2023 census estimate
```

Without these, the model sounds like it's making things up. With them, it can say:
*"According to the Statistics Bureau of Japan's 2023 census estimate..."*

**Freshness is especially important for:**
- Stock prices (minutes old matters)
- Population data (years old matters)
- Sports scores, elections, breaking news (seconds matter)
- Legal or regulatory data (may have changed)

When data is fresh: don't mention freshness unless asked.
When data is stale: the model should proactively warn the user.

---

## 7. Suggested Follow-ups

Follow-up suggestions are a lightweight way to make your agent more proactive:

```
suggested_followups:
  - get_population('osaka') — for comparison
  - get_urban_density('tokyo') — for density per km²
```

The model may:
- Call these tools automatically (if the user's goal seems to need them)
- Offer them as next-step suggestions in its response
- Use them to proactively enrich an incomplete answer

This is different from explicit chaining (experiment 02) — it's the tool communicating to the model what a thorough answer might require, giving the model the agency to decide whether to pursue it.

---

## 8. Context Shapes Reasoning, Not Just Output

The most subtle effect of rich tool results is on the model's *reasoning*, not just its output. When a result includes caveats, the model doesn't just quote them — it integrates them into its analysis.

**Example:** `get_stock_info("GME")` returns:
```
P/E ratio: N/A (company not profitable)
caveats:
  - Price can be heavily influenced by social media sentiment
```

The model's response might be:
> *"GME is currently trading at $14.20. It's worth noting that traditional valuation metrics like P/E ratio don't apply here since the company isn't profitable, and its price has historically been driven by retail investor sentiment rather than fundamentals. If you're researching this for investment purposes, you'd want to look at factors beyond the stock price."*

The model synthesized the caveats into an analytical statement — it didn't just list them. This is context-driven reasoning: the result's structure told the model what kind of answer was appropriate.

---

## 9. Key Principles

> **Principle 1 — A tool result is a reasoning artifact, not just a return value.**
> The model reads every field. Design results to communicate not just what the answer is, but how much to trust it, where it came from, and what to do with it.

> **Principle 2 — Confidence calibration is a first-class feature.**
> Without confidence scores, the model sounds equally certain about everything. A 40% confidence signal prevents overconfident answers on ambiguous data.

> **Principle 3 — Surface caveats your users would want to know.**
> If you know a limitation of the data — ambiguity, staleness, scope — include it in the result. The model will incorporate it. If you don't, the model won't know to mention it.

> **Principle 4 — Provenance answers "how do you know that?"**
> Source and freshness let the model speak with appropriate authority. Without them, the model sounds like it's fabricating.

> **Principle 5 — Rich results decouple knowledge from presentation.**
> Your tool returns what it knows about the data. The model decides how to present it. This is the right separation of concerns.

---

## 10. In the Real World

**Wolfram Alpha / Perplexity**
Search and compute tools return not just answers but confidence levels, source URLs, and "related queries." These context fields are fed directly to the language model to produce hedged, attributable answers.

**Medical Decision Support**
Clinical AI tools return diagnoses with confidence scores, differential diagnoses, evidence level (case report vs. RCT), and guideline references. The model uses all of these to produce appropriately cautious outputs.

**Financial Data APIs**
Bloomberg and Refinitiv data includes `as_of_date`, `data_quality_flag`, and `revision_count` on every field. These are standard context fields that downstream models use to decide how much weight to give a figure.

**RAG Systems**
When a RAG retriever returns chunks, it includes `score`, `source_document`, `section`, and sometimes `created_at`. The language model uses these to write attributable answers and hedge on low-score results.

**OpenAI Assistants — File Search**
When file search returns chunks, it includes the document name and page number in the result. This is tool result context — it gives the model the provenance it needs to cite sources correctly.

**Anthropic's Constitutional AI**
Claude is trained to respond differently based on the "confidence" signals in its training data — acknowledging uncertainty when the training signal was mixed. Tool result confidence is the runtime equivalent: a direct channel to calibrate model certainty on a specific fact.

---

## 11. Running the Experiment

```bash
# From the project root

# Mock mode — see bare vs. simple vs. rich results side by side
uv run python tools/05-tool-result-context/demo.py --mock

# Real mode — default (rich results)
ANTHROPIC_API_KEY=sk-... uv run python tools/05-tool-result-context/demo.py --real

# Real mode — compare result styles
ANTHROPIC_API_KEY=sk-... uv run python tools/05-tool-result-context/demo.py --real --style bare
ANTHROPIC_API_KEY=sk-... uv run python tools/05-tool-result-context/demo.py --real --style simple
ANTHROPIC_API_KEY=sk-... uv run python tools/05-tool-result-context/demo.py --real --style rich
```

**Suggested queries:**
- `"What's the population of Springfield?"` — ambiguity caveats drive a much better response
- `"Tell me about GME stock."` — speculative caveats lead to appropriate risk framing
- `"How do you say hello in Arabic?"` — cultural caveats produce a nuanced translation
- `"What's the population of Tokyo?"` — high confidence produces a direct confident answer

**Suggested exercises:**
1. Run the same query with `--style bare` and `--style rich` and compare the responses.
2. Lower the confidence on the Tokyo population to 0.3 and observe how the model's language changes.
3. Add a `followup_taken` field to the result that tracks whether the suggested follow-ups were used.
4. Add a new tool `get_urban_density(city)` and see whether the model calls it based on the follow-up suggestion in `get_population`.

---

*Previous: [Tool Error Handling](../04-tool-error-handling/) · Next: [Human-in-the-Loop](../06-human-in-the-loop/)*
