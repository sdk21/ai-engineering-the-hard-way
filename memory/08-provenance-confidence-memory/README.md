# Lesson: Provenance and Confidence Memory

**Vertical:** Memory | **Difficulty:** Intermediate–Advanced | **Status:** ✅ Ready

---

## Table of Contents

1. [The Blind Accumulation Problem](#1-the-blind-accumulation-problem)
2. [Provenance: Where Did This Come From?](#2-provenance-where-did-this-come-from)
3. [Confidence: How Much Should We Trust It?](#3-confidence-how-much-should-we-trust-it)
4. [Architecture](#4-architecture)
5. [The Data Model](#5-the-data-model)
6. [Confidence Dynamics](#6-confidence-dynamics)
7. [Contradiction Detection](#7-contradiction-detection)
8. [Exponential Decay](#8-exponential-decay)
9. [Threshold Injection](#9-threshold-injection)
10. [The Audit Trail](#10-the-audit-trail)
11. [Extractor Quality as Prior](#11-extractor-quality-as-prior)
12. [Failure Modes](#12-failure-modes)
13. [Key Principles](#13-key-principles)
14. [In the Real World](#14-in-the-real-world)
15. [Running the Experiment](#15-running-the-experiment)

---

## 1. The Blind Accumulation Problem

Every memory experiment so far accumulates facts without asking two questions that matter enormously in production:

**Where did this fact come from?**

When the model says "your name is Alice", which turn established that? Was it extracted by a reliable LLM extractor or a fragile regex? Was it stated once in passing or confirmed three times? If the model is wrong, can we trace the error to its source?

Without provenance, memory is a black box. You cannot audit it, debug it, or calibrate trust in it.

**How much should we trust it?**

Not all facts are equal:
- "My name is Alice" said once, extracted by regex: moderate confidence
- "My name is Alice" confirmed five times across sessions: high confidence
- "I live in Tokyo" then "Actually I moved to Osaka": contradiction — lower confidence
- "I live in Tokyo" stated months ago, never mentioned since: decayed confidence

Without confidence scores, a casual offhand comment carries the same weight as a repeatedly confirmed fact. Contradictions silently overwrite prior beliefs. Old stale facts inject with the same authority as freshly confirmed ones.

This experiment adds both layers to entity-style fact storage.

---

## 2. Provenance: Where Did This Come From?

Every fact in the store carries a list of `Provenance` records — one per assertion event:

```python
@dataclass
class Provenance:
    turn_index:  int    # which turn produced this assertion
    session_id:  str    # which session (for cross-session recall)
    timestamp:   str    # wall-clock time of assertion
    source_text: str    # the raw sentence the fact was extracted from
    extractor:   str    # "regex", "llm", or "manual"
```

When the same fact is asserted again, a new `Provenance` record is appended — not overwriting the first. The full assertion history is always recoverable:

```
user.name = 'Alice'
  provenance: turn 1 [regex] "My name is Alice."
  provenance: turn 4 [regex] "Just to confirm — my name is Alice."
  provenance: turn 9 [llm]   "Alice here, checking in again."
```

This makes the audit trail complete. You can always answer: "why does the model believe X?" by inspecting the provenance chain for fact X.

---

## 3. Confidence: How Much Should We Trust It?

Each fact has a `ConfidenceScore` that evolves dynamically:

```python
@dataclass
class ConfidenceScore:
    score:                float   # current confidence in [0.0, 1.0]
    reinforcement_count:  int     # number of confirming re-assertions
    contradiction_count:  int     # number of conflicting assertions
    last_seen_turn:       int     # turn of the most recent assertion
    last_decay_turn:      int     # turn of the most recent decay step
```

The score is updated by four events:

| Event | Effect | Amount |
|-------|--------|--------|
| New fact (LLM extractor) | Initial score | 0.85 |
| New fact (regex extractor) | Initial score | 0.65 |
| Reinforcement (same value asserted again) | Score ↑ | +0.10 |
| Contradiction (different value for same attribute) | Score ↓ | −0.30 |
| Decay (idle turns without reinforcement) | Score ↓ | exponential |

---

## 4. Architecture

```
Each user turn
    ↓
FactExtractorFn(role, content, turn_index) → list[Assertion]
    ↓
ProvenanceStore.assert_fact(assertion, turn_index)
    ├─ if new (entity, attribute): create FactRecord, initial score
    ├─ if same value:              reinforce — score ↑, append provenance
    └─ if different value:         contradict — score ↓, record conflict
    ↓
ProvenanceStore.apply_decay(current_turn)
    → for each fact: score × 0.5^(idle_turns / half_life)
    ↓
get_system_prompt()
    → inject only facts with score ≥ threshold
    → suppressed facts retained for auditing, excluded from LLM context
```

Two extractor modes, same interface (`FactExtractorFn`):
- `mock_extractor` — regex, 9 patterns, initial confidence 0.65
- `make_real_extractor(api_key)` — Claude returns JSON, initial confidence 0.85

---

## 5. The Data Model

### Assertion

The output of the extractor — a claimed fact before integration:

```python
@dataclass
class Assertion:
    entity:              str    # e.g. "user", "alice", "project_orion"
    attribute:           str    # e.g. "name", "role", "location"
    value:               str    # e.g. "Alice", "backend architect", "Tokyo"
    source_text:         str    # raw sentence
    extractor:           str    # "regex" or "llm"
    initial_confidence:  float  # override if needed
```

### FactRecord

A fact as stored — assertion plus living confidence plus full history:

```python
@dataclass
class FactRecord:
    entity:         str
    attribute:      str
    value:          str             # current believed value (last-write)
    confidence:     ConfidenceScore
    provenance:     list[Provenance]     # all assertion events
    contradictions: list[Contradiction]  # conflicting claims
```

### Contradiction

```python
@dataclass
class Contradiction:
    turn_index:        int
    conflicting_value: str
    timestamp:         str
```

When a contradiction occurs, the new value wins (last-write) but the old value is recorded as a contradiction. The confidence score is penalised. The contradicted value is visible in the audit trail.

---

## 6. Confidence Dynamics

### Reinforcement

When the same (entity, attribute, value) is asserted again:

```
score_new = min(1.0, score_old + 0.10)
```

Repeated confirmation narrows uncertainty. A fact stated once at 0.65 rises to 0.75 after one confirmation, 0.85 after two, and caps at 1.0 after four.

### Contradiction

When the same (entity, attribute) is asserted with a **different value**:

```
score_new = max(0.0, score_old − 0.30)
```

Two contradictions from an initial score of 0.65:
- After first contradiction: 0.65 − 0.30 = 0.35 (below 0.40 threshold → suppressed)
- After second contradiction: 0.35 − 0.30 = 0.05 (near zero)

The fact is still retained for auditing but stops being injected into the prompt.

---

## 7. Contradiction Detection

Contradiction detection is (entity, attribute)-level: same entity and attribute, different value.

```python
if assertion.value.lower() == record.value.lower():
    # Same value — reinforce
    record.confidence.reinforce(REINFORCE_DELTA)
else:
    # Different value — contradiction
    record.contradictions.append(Contradiction(...))
    record.confidence.contradict(CONTRADICT_DELTA)
    record.value = assertion.value   # last-write wins
```

**What counts as a contradiction:**
- "I live in Tokyo" then "I live in Osaka" — contradicted location
- "I'm a software engineer" then "I'm a backend architect" — contradicted role

**What does NOT count:**
- Same attribute, same value — reinforcement
- Different attribute — independent facts, no conflict

**Limitations of last-write semantics:** the system always accepts the most recently asserted value, even if it contradicts a highly confident prior. A more sophisticated system might refuse to update a high-confidence fact from a low-confidence extractor, or might ask the user to confirm before overwriting a repeatedly-confirmed value.

---

## 8. Exponential Decay

Facts that are not reinforced gradually lose confidence:

```
score_after_N_idle_turns = score × 0.5^(N / half_life)
```

At `half_life = 20` turns:
- After 20 turns idle: score × 0.5 (halved)
- After 40 turns idle: score × 0.25 (quartered)
- After 60 turns idle: score × 0.125

For a fact starting at 0.65:
- After 20 turns: 0.325 — below default threshold, suppressed from prompt
- After 40 turns: 0.163
- After 60 turns: 0.081

Decay is incremental — applied once per turn, based on turns elapsed since the **last decay application** (tracked separately from `last_seen_turn`):

```python
def decay(self, current_turn: int, half_life: float) -> None:
    if self.last_seen_turn == current_turn:
        self.last_decay_turn = current_turn   # reinforced this turn — reset clock
        return
    incremental = current_turn - self.last_decay_turn
    factor = math.pow(0.5, incremental / half_life)
    self.score *= factor
    self.last_decay_turn = current_turn
```

The `last_decay_turn` field is the key correctness detail. Without it, applying decay each turn would compound incorrectly — each call would apply the full `0.5^(total_idle/half_life)` factor to an already-decayed score, resulting in double-exponential decay and collapsing scores to zero far too quickly.

---

## 9. Threshold Injection

Only facts at or above the `injection_threshold` (default: 40%) appear in the system prompt:

```
## Known facts (confidence ≥ 40%)
user:
  name: Alice  [0.71]
  role: software engineer  [0.65]
```

Suppressed facts (below threshold) are retained in the store and visible via the `suppressed` command — they are not forgotten, just not injected. This prevents the LLM from acting on low-confidence or contradicted information.

**Why 40%?** It is a conservative default — roughly the point where the fact has been either reinforced at least once or has not yet suffered a contradiction. Adjust with `--threshold` to match your application's tolerance for stale or uncertain facts.

---

## 10. The Audit Trail

The `audit` command surfaces the full history of every fact:

```
user:
  user.name = 'Alice'
    confidence : 0.706 (medium)  reinforced×1  contradicted×0
    provenance : turn 1 [regex] "My name is Alice."
    provenance : turn 4 [regex] "Just to confirm — my name is Alice."

  user.role = 'backend architect'
    confidence : 0.306 (low)  reinforced×0  contradicted×1
    provenance : turn 2 [regex] "I work as a software engineer."
    provenance : turn 5 [regex] "Actually, I'm a backend architect."
    conflict   : turn 5 asserted role='backend architect'   ⚠ SUPPRESSED

  user.location = 'Kyoto'
    confidence : 0.006 (very_low)  reinforced×0  contradicted×2
    provenance : turn 3 [regex] "I live in Tokyo."
    provenance : turn 6 [regex] "I live in Osaka."
    provenance : turn 7 [regex] "I live in Kyoto."
    conflict   : turn 6 asserted location='Osaka'
    conflict   : turn 7 asserted location='Kyoto'   ⚠ SUPPRESSED
```

This is the memory's full chain of custody. For any fact the model asserts or fails to assert, you can answer:
- When was this fact first established?
- How many times was it confirmed?
- Were there conflicts? What were they?
- Which extractor produced each piece of evidence?
- Is this fact currently being injected into the prompt, or suppressed?

---

## 11. Extractor Quality as Prior

The initial confidence is set by extractor type:

| Extractor | Initial Confidence | Rationale |
|-----------|--------------------|-----------|
| `llm`     | 0.85 | LLMs handle paraphrases, context, negation; fewer false positives |
| `regex`   | 0.65 | Brittle; correct when patterns match but misses and false-positives more likely |
| `manual`  | 1.00 | Human-confirmed facts; no uncertainty |

This is a Bayesian prior: before a fact is reinforced or contradicted, the extractor quality sets the starting belief. An LLM extractor starts closer to the injection threshold; a regex-extracted fact needs one reinforcement to reach "medium" confidence.

In a production system, you might add additional extractors with calibrated priors:
- `llm_structured` (Claude with tool use, schema-validated): 0.90
- `inferred` (derived from other facts, not directly stated): 0.50
- `web_search` (retrieved from an external source): 0.70

---

## 12. Failure Modes

**Contradictions from paraphrases** — "I'm a software engineer" and "I'm an SWE" are the same fact expressed differently. The system treats them as a contradiction because string comparison fails. Mitigation: normalise values before comparison, or use semantic similarity to detect near-matches.

**Over-penalising legitimate updates** — "I used to work in Tokyo, now I'm in Osaka" is a valid update, not an error. The −0.30 penalty applies regardless of intent. Mitigation: detect temporal language ("used to", "now", "as of last week") and apply a softer update rather than a contradiction penalty.

**Decay unsuited to session cadence** — A `half_life` of 20 turns is calibrated for an active conversational assistant. For a weekly-use assistant, a turn-based half-life of 20 is far too aggressive — the user would need to re-confirm their name every session. Mitigation: calibrate half-life to session frequency, or use wall-clock time instead of turn counts.

**No cross-attribute reasoning** — The store treats `user.role = "engineer"` and `user.role = "architect"` as a contradiction. But `user.role = "engineer"` and `project.name = "Orion"` are independent facts even if they mention the same entity. The contradiction logic is (entity, attribute)-scoped only.

**Confidence inflation via repetition** — A user who compulsively restates facts will push every score to 1.0 regardless of actual veracity. The score reflects frequency of assertion, not truth. This is inherent to memory systems based on self-report.

---

## 13. Key Principles

> **Principle 1 — Every fact needs a chain of custody.**
> "The model believes X" is not enough information to act on. You need to know when X was asserted, by what extractor, from what text, and how many times it has been confirmed or contradicted. Without that chain of custody, you cannot debug, audit, or calibrate trust.

> **Principle 2 — Confidence is dynamic, not static.**
> A fact's trustworthiness changes over time: it rises with confirmation, falls with contradiction, and fades without reinforcement. A memory system that assigns confidence once at extraction time and never updates it treats all facts as equally trustworthy regardless of their history.

> **Principle 3 — Suppression is not deletion.**
> Low-confidence facts should not be injected into the prompt — the model should not act on uncertain information. But they should not be deleted either — they are evidence of past assertions and future contradictions. Keep everything in the store; control what reaches the LLM via the injection threshold.

> **Principle 4 — The extractor is the prior.**
> Initial confidence encodes what you believe about the extractor's reliability before any evidence accumulates. A well-calibrated prior reduces the number of reinforcements needed to reach the injection threshold and limits the damage from false-positive extractions.

> **Principle 5 — Decay prevents stale facts from dominating.**
> Without decay, a fact asserted once three months ago injects into every subsequent session forever. Decay ensures that facts which have not been reinforced fade naturally, preventing the model from acting on information that may no longer be true.

---

## 14. In the Real World

**Wikidata — statement ranks**
Wikidata's data model has a "rank" field on every statement: normal, preferred, or deprecated. When a fact changes (a person's role, a country's capital), the old statement is not deleted — it is marked "deprecated" and a new "preferred" statement is added. This is provenance without deletion: the complete history of every assertion is preserved, with a priority ranking that controls which value is used in responses. It is the same suppression-not-deletion principle at encyclopaedic scale.

**Google Knowledge Graph — confidence scores**
Google's Knowledge Graph assigns confidence scores to extracted facts from the web. A fact appearing on ten high-authority pages has higher confidence than one appearing on a single blog. Contradictory facts from different sources both persist in the store; the confidence score determines which is surfaced in the Knowledge Panel. The reinforcement pattern (more sources → higher confidence) is the same mechanism as this experiment.

**Palantir Gotham — evidence graph**
Palantir's intelligence analysis platform stores every piece of information alongside its source (document, analyst, date), confidence level, and links to corroborating or contradicting evidence. Analysts can query not just what the system believes but why — following the provenance chain back to primary sources. This is the audit trail from this experiment at national-security scale.

**LangChain / LlamaIndex memory modules**
Several open-source memory implementations track "memory strength" as a scalar that increases with retrieval frequency and decreases over time — directly implementing the reinforcement + decay dynamics from this experiment. CrewAI's long-term memory module stores a `last_accessed` timestamp and boosts scores when memories are retrieved, a variant of reinforcement.

**Mem0 — contradictions and updates**
Mem0's memory update logic detects when a new assertion conflicts with a stored one and handles the conflict explicitly: it logs the contradiction, updates the stored value, and notes the update reason in the memory record. This is the same last-write-wins-with-penalty strategy in this experiment, embedded in a widely-deployed production memory system.

**Medical records — provenance as a legal requirement**
In healthcare, every entry in a patient record must carry its author, timestamp, and amendment history. No fact is ever deleted — only superseded by a newer entry with a note that the prior entry was amended. Confidence is implicit: a finding from a specialist's report carries more clinical weight than a patient's self-report. This is provenance memory enforced by regulation.

**Cognitive science — source monitoring**
In human cognition, "source monitoring" is the process of remembering not just what you know but where you learned it. Failures of source monitoring — remembering the fact but not the source — lead to false memories and misattributed beliefs. AI memory systems that store facts without provenance replicate this failure mode at scale. Provenance-aware memory is the engineering equivalent of robust source monitoring.

---

## 15. Running the Experiment

```bash
# From the project root

# Mock mode — regex extraction, no API key needed
uv run python memory/08-provenance-confidence-memory/demo.py --mock

# Real mode — Claude extracts facts with higher initial confidence
ANTHROPIC_API_KEY=sk-... uv run python memory/08-provenance-confidence-memory/demo.py --real

# Custom threshold and decay
uv run python memory/08-provenance-confidence-memory/demo.py --mock \
    --threshold 0.5 \
    --half-life 10
```

**Suggested conversation sequence:**

```
# Establish facts
You: My name is Alice.
You: I work as a software engineer.
You: I live in Tokyo.
You: facts          ← 3 facts, all at 0.65

# Reinforce
You: Just to confirm — my name is Alice.
You: facts          ← name rose to 0.71

# Contradict
You: Actually, I'm a backend architect, not an engineer.
You: facts          ← role dropped to 0.35, suppressed from prompt

# Double-contradict
You: I live in Osaka.
You: I live in Kyoto.
You: facts          ← location at ~0.006, name is the only injectable fact
You: suppressed     ← see role and location with conflict counts
You: audit          ← full provenance chain for all three facts

# Decay
You: decay 30       ← simulate 30 idle turns
You: facts          ← name dropped below threshold, no injectable facts
You: suppressed     ← all three facts now suppressed
```

**Compare mock vs real:**

The regex extractor starts facts at 0.65; the LLM extractor starts at 0.85. Try the same conversation in both modes and use `audit` to compare:

- Regex: facts may need one reinforcement to reach "medium" confidence
- LLM: facts start at "high" confidence, closer to the injection threshold

This difference makes the extractor quality concrete — you can see in the audit trail which extractor was responsible for each provenance record.

---

*Previous: [Knowledge Graph Memory](../07-knowledge-graph-memory/) | Next: [Conflict Resolution Memory](../09-conflict-resolution-memory/)*
