# Lesson: Conflict Resolution Memory

**Vertical:** Memory | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## Table of Contents

1. [Why Conflict Detection Is Not Enough](#1-why-conflict-detection-is-not-enough)
2. [The Five Conflict Types](#2-the-five-conflict-types)
3. [Architecture](#3-architecture)
4. [Conflict Classification](#4-conflict-classification)
5. [Resolution Strategies](#5-resolution-strategies)
6. [Per-Attribute Strategy Overrides](#6-per-attribute-strategy-overrides)
7. [User Arbitration](#7-user-arbitration)
8. [Dependency Propagation](#8-dependency-propagation)
9. [The Audit Trail](#9-the-audit-trail)
10. [Reviving Retracted Facts](#10-reviving-retracted-facts)
11. [Failure Modes](#11-failure-modes)
12. [Key Principles](#12-key-principles)
13. [In the Real World](#13-in-the-real-world)
14. [Running the Experiment](#14-running-the-experiment)

---

## 1. Why Conflict Detection Is Not Enough

Experiment 08 detected contradictions and applied a uniform confidence penalty. That is the minimum viable response: better than ignoring conflicts, worse than understanding them.

The problem is that not all contradictions are the same. Consider these four sentences:

```
"I work at Acme Corp."
"I moved to Osaka."           ← "moved" signals a legitimate change
"My name is Alicia."          ← near-match of "Alice" — likely a correction
"I work at Beta Corp."        ← no temporal cue — genuine conflict
"I no longer work at Beta Corp."  ← explicit negation — retraction
```

Experiment 08 treats all of these identically: −0.30 confidence penalty, overwrite the stored value. This is wrong in most cases:

- The **move to Osaka** is a valid update — penalising it discourages the model from tracking current state
- The **Alicia correction** is likely a minor error (regex mis-heard "Alice") — a soft penalty is appropriate
- The **Beta Corp conflict** is genuinely uncertain — a hard penalty is right
- The **retraction** should mark the fact as deprecated, not just penalised

And uniform confidence penalties are only one part of the problem. You also need to decide: when two values conflict, which one wins? Last-write? The one with more evidence? The one from the more authoritative source? Or should the system pause and ask the user?

This experiment implements all five conflict types, five resolution strategies, per-attribute strategy overrides, dependency propagation, and user arbitration — a complete conflict resolution engine.

---

## 2. The Five Conflict Types

### UPDATE — temporal language present

```
"I moved to Osaka."
"I'm now working at Beta Corp."
"As of this month, I lead the infrastructure team."
```

Detected by scanning for temporal markers: `"now"`, `"moved"`, `"currently"`, `"as of"`, `"recently"`, `"just"`, etc. Confidence penalty: **0.00** — this is an expected state change, not an error.

### CORRECTION — near-match value

```
"My name is Alice."  →  "My name is Alicia."
"I work in backend."  →  "I work on backend."
```

Detected when the old and new values are within edit distance 2, or one is a substring of the other. Likely a typo or extractor error rather than a true contradiction. Confidence penalty: **−0.10** (soft).

### CONFLICT — genuine disagreement

```
"I work at Acme Corp."  →  "I work at Beta Corp."
(no temporal language, not a near-match)
```

Hard confidence penalty: **−0.30**. The system does not know which is true; the resolution strategy determines the winner.

### RETRACTION — explicit negation

```
"I no longer work at Acme."
"I left Beta Corp."
"I'm not a manager anymore."
```

Detected by scanning for retraction phrases: `"no longer"`, `"left"`, `"quit"`, `"not anymore"`, etc. The fact is marked `RETRACTED` and removed from injection. No confidence penalty — the retraction is informative, not erroneous.

### PROPAGATED STALE — dependency changed

```
"My role is engineer."  →  "My role is manager."
```

If a dependency map says `role → team`, then when role is contradicted, `team` is automatically marked `STALE` with a −0.20 confidence penalty, even though no direct assertion about team was made.

---

## 3. Architecture

```
Each user turn
    ↓
FactExtractorFn(role, content, turn_index) → list[Assertion]
    (Assertion.value = None signals a retraction)
    ↓
ConflictResolutionStore.assert_fact(assertion, turn_index)
    ├─ new fact:         create FactRecord, set initial confidence
    ├─ same value:       reinforce — score ↑
    ├─ retraction text:  _handle_retraction — status=RETRACTED, score=0
    └─ different value:  classify_conflict → ConflictType
                         select strategy (global or per-attribute override)
                         apply resolution → winner
                         penalise by ConflictType delta
                         _propagate_stale(changed_key)
    ↓
ConflictResolutionStore.apply_decay(current_turn)
    ↓
get_system_prompt()
    → inject only ACTIVE facts above threshold
    → note UNRESOLVED conflicts in system prompt
```

---

## 4. Conflict Classification

```python
def classify_conflict(old_value, new_value, source_text) -> ConflictType:
    if new_value is None:
        return ConflictType.RETRACTION
    if _is_temporal_update(source_text):
        return ConflictType.UPDATE
    if _edit_distance(old_value, new_value) <= 2 or (old in new or new in old):
        return ConflictType.CORRECTION
    return ConflictType.CONFLICT
```

Classification is intentionally simple. Real-world systems would use:
- Semantic similarity (not just edit distance) for correction detection
- NER-aware comparison (proper noun changes are rarely corrections)
- Domain-specific rules ("engineer" → "senior engineer" is always a promotion, not a conflict)

The confidence delta table:

| ConflictType | Delta | Rationale |
|-------------|-------|-----------|
| UPDATE      | 0.00  | Expected change — no uncertainty introduced |
| CORRECTION  | −0.10 | Minor uncertainty — probably an error, not a real change |
| CONFLICT    | −0.30 | High uncertainty — genuine disagreement |
| RETRACTION  | 0.00  | No delta — fact is deprecated, not uncertain |

---

## 5. Resolution Strategies

Five strategies determine which value wins when a `CONFLICT` or `CORRECTION` is detected.

### LAST_WRITE

The most recently asserted value always wins.

```
Acme Corp  →  Beta Corp  →  winner: Beta Corp
```

**Use when:** facts are expected to change frequently (location, employer, project name). Recency is the best proxy for truth.

**Risk:** a single mistaken assertion overwrites a high-confidence prior.

### HIGHEST_CONF

The value with the higher current confidence score wins.

```
Acme Corp (conf=0.85, reinforced×3)  vs  Beta Corp (incoming, conf=0.65 initial)
→  winner: Acme Corp  (existing is more confident)
```

**Use when:** facts are stable and corrections are rare (name, birthdate). Protects against single-assertion overwrites of well-established facts.

**Risk:** a legitimate change gets rejected if the existing fact is highly reinforced.

### MOST_FREQUENT

The value that has been asserted the most times wins.

```
Alice (asserted 4 times)  vs  Alicia (asserted 1 time)
→  winner: Alice
```

**Use when:** you have multiple sessions or multiple sources asserting the same entity. The majority view is more likely to be correct.

**Risk:** a systematic extraction error (the same wrong value extracted repeatedly) wins over a single correct assertion.

### SOURCE_PRIORITY

Values from higher-authority sources win regardless of recency or frequency.

```
Source authority: manual (100) > llm (70) > regex (40)

manual: employer = "Acme Corp"   (authority 100)
 regex: employer = "Beta Corp"   (authority 40)
→  winner: Acme Corp  (manual wins)
```

**Use when:** you have mixed-quality inputs — human-confirmed notes, LLM extractions, and regex patterns — and want to guarantee that manually verified facts are not overwritten by automated extraction.

**Risk:** an authoritative but outdated manual entry blocks legitimate updates from lower-authority sources.

### USER_ARBITRATION

No automatic winner. The conflict is flagged as `UNRESOLVED` in the system prompt and the store. The application must call `resolve_conflict(entity, attribute, value)` to supply the winner.

```
user.location = 'Tokyo'  vs  'Osaka'  →  UNRESOLVED
(system prompt: "## Unresolved conflicts (awaiting user input): user.location")
User: resolve location Tokyo
→  user.location = 'Tokyo'  [active, conf += 0.20]
```

**Use when:** accuracy is paramount and the cost of an automated wrong decision exceeds the cost of asking the user. Customer support systems, medical records, financial data.

---

## 6. Per-Attribute Strategy Overrides

Different attributes have different semantics and warrant different strategies:

```python
store = ConflictResolutionStore(
    default_strategy=Strategy.LAST_WRITE,
    attribute_strategies={
        "name":     Strategy.HIGHEST_CONF,    # names rarely change; trust confident value
        "employer": Strategy.SOURCE_PRIORITY, # manual notes override extractions
        "location": Strategy.LAST_WRITE,      # people move; accept the latest
        "salary":   Strategy.USER_ARBITRATION,# never guess on sensitive facts
    }
)
```

The per-attribute map is checked first; the default strategy is the fallback. This allows a single store to behave differently for stable facts (name, DOB) vs volatile facts (location, employer, role).

---

## 7. User Arbitration

When the `USER_ARBITRATION` strategy is active for an attribute:

1. The conflict is recorded in the `FactRecord` with `winner=None`
2. The fact's status is set to `FactStatus.UNRESOLVED`
3. The system prompt gains an "## Unresolved conflicts" section listing the pending attributes
4. The LLM can inform the user that clarification is needed
5. The application calls `store.resolve_conflict(entity, attribute, chosen_value)`
6. The winner is recorded in the conflict record, status reverts to `ACTIVE`, confidence is boosted +0.20

This pattern decouples the conflict detection (automatic, happens during extraction) from the conflict resolution (explicit, requires human input). The system continues operating — unresolved facts are excluded from injection but the conversation continues — while surfacing the ambiguity for eventual resolution.

---

## 8. Dependency Propagation

Real knowledge is not flat. Facts depend on other facts:

- If your **employer** changes, your **team** (which was defined in the context of that employer) becomes stale
- If your **role** changes, your **responsibilities** become stale
- If a **project** is renamed, all **project-specific facts** become stale

The `dependency_map` parameter lets you declare these dependencies:

```python
store = ConflictResolutionStore(
    dependency_map={
        ("user", "employer"): [("user", "team"), ("user", "office_location")],
        ("user", "role"):     [("user", "team"), ("user", "direct_reports")],
    }
)
```

When a fact changes (via conflict, update, or retraction), `_propagate_stale` walks the dependency map and marks all dependents as `STALE` with a −0.20 confidence penalty. Stale facts are excluded from injection (same as suppressed facts) and visible in the audit trail with a note explaining which dependency triggered the stale marking.

---

## 9. The Audit Trail

Every conflict resolution decision is recorded in `ConflictRecord`:

```python
@dataclass
class ConflictRecord:
    turn_index:     int
    conflict_type:  ConflictType        # UPDATE / CORRECTION / CONFLICT / RETRACTION
    old_value:      str
    new_value:      str | None
    strategy_used:  Strategy            # which strategy resolved this
    winner:         str | None          # winning value (None if unresolved)
    old_confidence: float               # score before resolution
    new_confidence: float               # score after penalty
    notes:          str                 # e.g. "resolved by user arbitration"
```

The `audit` command renders the full trail:

```
user.employer = None  [retracted]  ⚠ SUPPRESSED
  confidence   : 0.000 (very_low)  asserted×1
  provenance   : turn 2 [regex] "I work at Acme Corp."
  provenance   : turn 5 [regex] "I work at Beta Corp."
  provenance   : turn 6 [regex] "I no longer work at Beta Corp."
  conflict     : turn 5 [conflict] old='Acme Corp' new='Beta Corp' via SOURCE_PRIORITY → kept old 'Acme Corp'
  conflict     : turn 6 [retraction] old='Beta Corp' new=None via LAST_WRITE → retracted

user.team = 'backend'  [stale]  ⚠ SUPPRESSED
  confidence   : 0.450 (low)  asserted×1
  provenance   : turn 8 [regex] "My team is backend."
  conflict     : propagated [conflict] old='backend' new='backend' via LAST_WRITE → kept old 'backend'
                 (marked stale: dependency ('user', 'role') changed)
```

For every fact in the store you can answer:
- What is the current believed value, and is it injected or suppressed?
- What evidence supports it (provenance chain)?
- Was it ever contested? By what? Under which strategy? Who won?
- Was it retracted? When? By what source?
- Was it marked stale? Which dependency triggered it?

---

## 10. Reviving Retracted Facts

A fact that has been retracted has `status=RETRACTED` and `value=None`. If the user later asserts the same attribute again ("I joined Delta Corp"), the system sees this as a conflict against the retracted fact. Since the stored value is `None`, `classify_conflict("", "Delta Corp", source)` returns `CONFLICT` (not an edit-distance correction). Resolution proceeds normally under the configured strategy — the fact is revived with a penalised confidence score, reflecting the uncertainty introduced by the prior retraction.

This is intentional: a fact that was retracted and then re-asserted is inherently less certain than one that was never contradicted.

---

## 11. Failure Modes

**Temporal markers are surface-level** — `_is_temporal_update` searches for literal strings. A sentence like "I recently stopped working at Acme" contains "recently" (temporal) but is actually a retraction. Both classifiers would claim it. The retraction check in `_handle_retraction` fires first (via `_is_retraction`), so this particular example is handled correctly — but the interaction between the two classifiers is brittle. Mitigation: use an LLM to classify conflict type rather than heuristics.

**Edit distance as proxy for correction** — "Alice" → "Alicia" (edit distance 2) is correctly classified as a correction. But "Alice" → "Alison" (edit distance 3) would be classified as a full conflict even if they are the same person. Real names have no reliable edit-distance threshold for correction detection. Mitigation: phonetic similarity (Soundex, Metaphone) or LLM-based semantic similarity.

**HIGHEST_CONF can lock in errors** — a fact reinforced 10 times has conf ≈ 1.0. A single correct contradicting assertion starts at 0.65. HIGHEST_CONF will keep the wrong value even when the correction is from a more reliable source. Mitigation: use SOURCE_PRIORITY for high-stakes attributes, or add a "force update" mechanism that bypasses confidence comparison.

**Dependency maps must be maintained manually** — if the application developer forgets to add a dependency edge, stale propagation doesn't happen. Mitigation: derive dependency maps automatically from the knowledge graph (Experiment 07) — an entity's attributes are implicitly dependent on its existence in the graph.

**Unresolved conflicts block injection** — if `USER_ARBITRATION` is used for a frequently-updated attribute, the user may need to resolve a conflict every session. This can become friction that discourages engagement. Mitigation: fall back to `LAST_WRITE` after N turns if no resolution is received.

---

## 12. Key Principles

> **Principle 1 — Not all contradictions are errors.**
> A temporal update, a correction, a genuine conflict, and an explicit retraction all produce a different value for the same (entity, attribute) pair — but they have different meanings and warrant different responses. A memory system that treats them identically is wrong more often than necessary.

> **Principle 2 — Resolution strategy is a policy decision, not an implementation detail.**
> There is no universally correct answer to "which value wins?" It depends on the domain, the source quality, the volatility of the attribute, and the cost of being wrong. Resolution strategies make this policy explicit and swappable.

> **Principle 3 — Stability and volatility require different strategies.**
> Stable attributes (name, date of birth) are best guarded by `HIGHEST_CONF` or `SOURCE_PRIORITY` — they should resist casual overwrites. Volatile attributes (location, employer, project status) are best served by `LAST_WRITE` — recency is the best proxy for truth.

> **Principle 4 — Dependency propagation prevents silent staleness.**
> When a fact changes, related facts may become outdated without any explicit assertion. Declaring dependencies and propagating stale marks ensures the system knows what it no longer fully trusts — rather than injecting confidently wrong information.

> **Principle 5 — Unresolvable conflicts should surface, not disappear.**
> When no automated strategy can reliably determine the winner, the right answer is to say so. `USER_ARBITRATION` makes unresolved conflicts visible in the system prompt so the LLM can acknowledge the uncertainty and prompt for clarification — rather than silently picking a possibly wrong value.

---

## 13. In the Real World

**Salesforce / HubSpot — data deduplication and merge**
CRM systems constantly resolve conflicts between records from different sources: a contact's phone number from a web form, a business card scan, and a manual update. Salesforce's deduplication rules implement SOURCE_PRIORITY and MOST_FREQUENT strategies — a manually verified record wins over an automatically imported one; a value appearing in three sources wins over one appearing once. The merge audit log is the conflict record in this experiment.

**Electronic Health Records — reconciliation**
HL7 FHIR (the standard for healthcare data exchange) has a formal `AuditEvent` resource that records every change to a patient record, including who made it, when, and what the prior value was. Conflicting medication records from different providers are flagged for clinician review — `USER_ARBITRATION` enforced by regulation. The dependency propagation concept appears as "clinical alerts": if a patient's known allergy record is updated, any related prescriptions are flagged as potentially stale.

**Git — three-way merge**
Git's three-way merge is conflict resolution on code. When two branches modify the same line, Git classifies the conflict type: if one branch is a direct ancestor of the other, the descendant wins (`LAST_WRITE`). If neither is an ancestor, the conflict is marked `UNRESOLVED` and the developer must arbitrate. The merge commit is the conflict record — complete with both old values, the resolution, and the author of the resolution.

**Wikipedia — edit conflicts and talk pages**
Wikipedia's edit conflict resolution is explicit `USER_ARBITRATION`: when two editors contest a fact, neither automatically wins. The conflict is documented on the article's Talk page with both editors providing their evidence (provenance). Admins act as arbitrators, and the resolution is documented with a rationale — exactly the `ConflictRecord` in this experiment, at encyclopaedic scale.

**Palantir Gotham — intelligence conflict resolution**
In intelligence analysis, the same entity (a person, organisation, location) may appear in hundreds of reports with conflicting attributes. Palantir Gotham uses a source-reliability scoring system (similar to SOURCE_PRIORITY) and a conflict resolution workflow where analysts adjudicate disagreements and record their reasoning. Every resolution decision is attributed to an analyst and timestamped — the audit trail is both a quality control mechanism and a legal record.

**Wikidata — statement deprecation**
When a Wikidata fact becomes outdated, it is not deleted — it is marked "deprecated" with a qualifier explaining why (e.g., `reason for deprecation: superseded by newer value`). Both the current and deprecated values remain in the store, with the current value ranked "preferred". This is the `RETRACTION` type with full audit retention — identical to the `FactStatus.RETRACTED` + provenance pattern in this experiment.

---

## 14. Running the Experiment

```bash
# From the project root

# Default strategy (LAST_WRITE)
uv run python memory/09-conflict-resolution-memory/demo.py --mock

# Switch strategies
uv run python memory/09-conflict-resolution-memory/demo.py --mock --strategy highest_conf
uv run python memory/09-conflict-resolution-memory/demo.py --mock --strategy source_priority
uv run python memory/09-conflict-resolution-memory/demo.py --mock --strategy most_frequent
uv run python memory/09-conflict-resolution-memory/demo.py --mock --strategy user_arbitration

# Real mode — Claude extracts facts (higher initial confidence)
ANTHROPIC_API_KEY=sk-... uv run python memory/09-conflict-resolution-memory/demo.py --real
```

**Suggested conversation (exercises all five conflict types):**

```
# New facts
You: My name is Alice.
You: I work at Acme Corp.
You: I live in Tokyo.

# Temporal UPDATE — no penalty
You: I moved to Osaka.           → [→] update, conf unchanged

# CORRECTION — soft penalty
You: My name is Alicia.          → [~] correction, conf −0.10

# Genuine CONFLICT — hard penalty
You: I work at Beta Corp.        → [⚡] conflict, conf −0.30

# Explicit RETRACTION
You: I no longer work at Beta Corp.   → [✗] retracted

# Dependency propagation
You: My role is engineer.
You: My team is backend.         → team depends on role
You: My role is manager.         → [⚡] role conflict → team marked STALE

facts     ← see what's injectable
suppressed  ← see what's below threshold or non-active
audit     ← full trail with conflict types, strategies, and winners
```

**Test USER_ARBITRATION:**

```bash
uv run python memory/09-conflict-resolution-memory/demo.py --mock --strategy user_arbitration
You: I live in Tokyo.
You: I live in Osaka.            → [?] unresolved conflict
unresolved                       ← see pending conflicts
resolve location Tokyo           ← you pick the winner
audit                            ← see resolution recorded
```

**Compare strategies side by side:**

Run the same conflicting sequence (`"My name is Alice."` then `"My name is Alicia."`) under `last_write`, `highest_conf`, and `source_priority`. Use `audit` after each to see how the same conflict is resolved differently — and why each strategy makes sense in different contexts.

---

*Previous: [Provenance & Confidence Memory](../08-provenance-confidence-memory/) | Next: Layered Memory (coming soon)*
