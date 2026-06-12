"""
Provenance and Confidence Memory
---------------------------------
All prior memory experiments accumulate facts without asking two critical
questions:

    1. Where did this fact come from?   — Provenance
    2. How much should we trust it?     — Confidence

Without provenance, a memory store is a black box: you cannot audit why
the model believes something, cannot trace an error back to its source,
and cannot distinguish a fact the user stated once from one they have
repeated across many sessions.

Without confidence, every fact is treated equally — a casual offhand
comment carries the same weight as a fact the user has confirmed five
times. Contradictions silently overwrite prior beliefs with no record
that a conflict occurred.

This experiment adds both layers to entity-style fact storage:

    Provenance — every fact record carries a list of Provenance objects,
    one per assertion. Each Provenance records which turn, which session,
    what raw text the fact was extracted from, and which extractor produced
    it (regex or LLM). Multiple assertions of the same fact are appended
    as additional provenance entries rather than overwriting the first.

    Confidence — each fact has a ConfidenceScore that evolves:
      - Initial score: set by extractor quality (LLM > regex > default)
      - Reinforcement: score rises when the same fact is asserted again
      - Contradiction: score falls when a conflicting value is asserted
        for the same (entity, attribute) pair; the conflict is recorded
      - Decay: score falls gradually for facts that have not been seen
        recently (configurable half-life in turns)

    Threshold injection: only facts whose confidence score meets the
    injection threshold (default: 0.4) appear in the system prompt.
    Low-confidence and contradicted facts are suppressed but retained
    in the store for auditing.

Architecture:

    Each user turn
        ↓
    FactExtractorFn(role, content, turn_index) → list[Assertion]
        ↓
    ProvenanceStore.assert_fact(assertion)
        → if new fact: create FactRecord with initial confidence
        → if same (entity, attr, value): reinforce — score ↑
        → if same (entity, attr) but different value: contradict — score ↓
           record contradiction in the existing record
        ↓
    ProvenanceStore.apply_decay(current_turn)
        → score ↓ for every fact not seen recently
        ↓
    get_system_prompt()
        → inject only facts with score ≥ threshold
        → suppressed facts are visible via the 'facts' command
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Provenance:
    """A single assertion event — one piece of evidence for a fact."""
    turn_index: int
    session_id: str
    timestamp: str
    source_text: str        # the raw sentence the fact was extracted from
    extractor: str          # "regex", "llm", or "manual"


@dataclass
class Contradiction:
    """A record of a conflicting assertion."""
    turn_index: int
    conflicting_value: str
    timestamp: str


@dataclass
class ConfidenceScore:
    """
    Mutable confidence score for a single fact.

    score:               current confidence in [0.0, 1.0]
    reinforcement_count: how many additional times this fact was asserted
    contradiction_count: how many times a conflicting value was asserted
    last_seen_turn:      turn index of the most recent assertion
    last_decay_turn:     turn index of the last decay application
                         (tracked separately from last_seen so incremental
                          decay is applied correctly each turn)
    """
    score: float
    reinforcement_count: int = 0
    contradiction_count: int = 0
    last_seen_turn: int = 0
    last_decay_turn: int = 0

    def reinforce(self, amount: float = 0.1) -> None:
        self.score = min(1.0, self.score + amount)
        self.reinforcement_count += 1

    def contradict(self, amount: float = 0.3) -> None:
        self.score = max(0.0, self.score - amount)
        self.contradiction_count += 1

    def decay(self, current_turn: int, half_life: float) -> None:
        """
        Incremental exponential decay.

        Called once per turn. Applies decay only for the turns elapsed
        since the LAST decay call (not since last_seen), so repeated
        calls compound correctly:

            score_after_N_idle_turns = score_initial × 0.5^(N / half_life)

        If the fact was reinforced this turn (last_seen_turn == current_turn),
        decay is skipped — reinforcement resets the idle clock.
        """
        if self.last_seen_turn == current_turn:
            # Reinforced this very turn — skip decay, reset clock
            self.last_decay_turn = current_turn
            return
        incremental = current_turn - self.last_decay_turn
        if incremental <= 0:
            return
        factor = math.pow(0.5, incremental / half_life)
        self.score *= factor
        self.last_decay_turn = current_turn

    @property
    def label(self) -> str:
        if self.score >= 0.8:
            return "high"
        if self.score >= 0.5:
            return "medium"
        if self.score >= 0.2:
            return "low"
        return "very_low"


@dataclass
class FactRecord:
    """
    A single (entity, attribute, value) fact with full provenance and
    a living confidence score.
    """
    entity: str
    attribute: str
    value: str
    confidence: ConfidenceScore
    provenance: list[Provenance] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.entity, self.attribute)

    def format_for_prompt(self) -> str:
        conf = f"{self.confidence.score:.2f} ({self.confidence.label})"
        return f"{self.entity}.{self.attribute} = {self.value!r}  [conf={conf}]"

    def format_for_audit(self) -> str:
        lines = [
            f"  {self.entity}.{self.attribute} = {self.value!r}",
            f"    confidence : {self.confidence.score:.3f} ({self.confidence.label})"
            f"  reinforced×{self.confidence.reinforcement_count}"
            f"  contradicted×{self.confidence.contradiction_count}",
        ]
        for p in self.provenance:
            lines.append(
                f"    provenance : turn {p.turn_index} [{p.extractor}]"
                f" \"{p.source_text[:60].rstrip()}\""
            )
        for c in self.contradictions:
            lines.append(
                f"    conflict   : turn {c.turn_index} asserted"
                f" {self.attribute}={c.conflicting_value!r}"
            )
        return "\n".join(lines)


# An Assertion is the raw output of the extractor — what was claimed.
@dataclass
class Assertion:
    entity: str
    attribute: str
    value: str
    source_text: str
    extractor: str          # "regex" | "llm"
    initial_confidence: float = 0.7


FactExtractorFn = Callable[[str, str, int], list[Assertion]]


# ---------------------------------------------------------------------------
# Provenance store
# ---------------------------------------------------------------------------

# Initial confidence by extractor quality
INITIAL_CONFIDENCE = {
    "llm":   0.85,
    "regex": 0.65,
    "manual": 1.0,
}

# Reinforcement / contradiction deltas
REINFORCE_DELTA   = 0.10
CONTRADICT_DELTA  = 0.30


class ProvenanceStore:
    """
    Stores (entity, attribute) → FactRecord with confidence tracking.

    A key maps to ONE fact record — the current believed value.
    When a contradicting value is asserted the conflict is recorded
    and the score drops, but the most-recently-asserted value wins
    (last-write with confidence penalty).
    """

    def __init__(
        self,
        decay_half_life: float = 20.0,
        injection_threshold: float = 0.40,
    ) -> None:
        self._facts: dict[tuple[str, str], FactRecord] = {}
        self.decay_half_life = decay_half_life
        self.injection_threshold = injection_threshold

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def assert_fact(
        self,
        assertion: Assertion,
        turn_index: int,
        session_id: str = "default",
    ) -> str:
        """
        Integrate an assertion into the store.

        Returns one of: "new", "reinforced", "contradicted"
        """
        key = (assertion.entity.lower(), assertion.attribute.lower())
        prov = Provenance(
            turn_index=turn_index,
            session_id=session_id,
            timestamp=_now(),
            source_text=assertion.source_text,
            extractor=assertion.extractor,
        )

        if key not in self._facts:
            # Brand-new fact
            initial = INITIAL_CONFIDENCE.get(assertion.extractor, assertion.initial_confidence)
            record = FactRecord(
                entity=assertion.entity.lower(),
                attribute=assertion.attribute.lower(),
                value=assertion.value,
                confidence=ConfidenceScore(
                    score=initial,
                    last_seen_turn=turn_index,
                    last_decay_turn=turn_index,
                ),
                provenance=[prov],
            )
            self._facts[key] = record
            return "new"

        record = self._facts[key]
        record.confidence.last_seen_turn = turn_index

        if assertion.value.lower() == record.value.lower():
            # Same value — reinforce
            record.confidence.reinforce(REINFORCE_DELTA)
            record.provenance.append(prov)
            return "reinforced"
        else:
            # Different value — contradiction
            record.contradictions.append(Contradiction(
                turn_index=turn_index,
                conflicting_value=assertion.value,
                timestamp=_now(),
            ))
            record.confidence.contradict(CONTRADICT_DELTA)
            # Update to new value (last-write wins) but score is penalised
            record.value = assertion.value
            record.provenance.append(prov)
            return "contradicted"

    def apply_decay(self, current_turn: int) -> None:
        """Apply decay to all facts not recently reinforced."""
        for record in self._facts.values():
            record.confidence.decay(current_turn, self.decay_half_life)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def injectable_facts(self) -> list[FactRecord]:
        """Facts at or above the injection threshold, sorted by entity."""
        return sorted(
            [r for r in self._facts.values()
             if r.confidence.score >= self.injection_threshold],
            key=lambda r: (r.entity, r.attribute),
        )

    def all_facts(self) -> list[FactRecord]:
        return sorted(self._facts.values(), key=lambda r: (r.entity, r.attribute))

    def suppressed_facts(self) -> list[FactRecord]:
        return [r for r in self._facts.values()
                if r.confidence.score < self.injection_threshold]

    def get(self, entity: str, attribute: str) -> FactRecord | None:
        return self._facts.get((entity.lower(), attribute.lower()))

    def __len__(self) -> int:
        return len(self._facts)

    def format_for_prompt(self) -> str:
        facts = self.injectable_facts()
        if not facts:
            return "(no facts above confidence threshold)"
        lines: list[str] = []
        current_entity: str = ""
        for r in facts:
            if r.entity != current_entity:
                current_entity = r.entity
                lines.append(f"{r.entity}:")
            lines.append(f"  {r.attribute}: {r.value}  [{r.confidence.score:.2f}]")
        return "\n".join(lines)

    def format_for_audit(self) -> str:
        facts = self.all_facts()
        if not facts:
            return "(no facts stored)"
        sections: list[str] = []
        current_entity: str = ""
        for r in facts:
            suppressed = "⚠ SUPPRESSED" if r.confidence.score < self.injection_threshold else ""
            if r.entity != current_entity:
                current_entity = r.entity
                sections.append(f"\n{r.entity}:")
            sections.append(r.format_for_audit() + (f"  {suppressed}" if suppressed else ""))
        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Provenance Memory
# ---------------------------------------------------------------------------

class ProvenanceMemory:
    """
    Full memory system with provenance tracking and confidence scoring.

    Lifecycle is the same as other memory experiments:
        memory = ProvenanceMemory(extractor=..., session_id="abc")
        memory.add_user_message("My name is Alice.")
        system = memory.get_system_prompt()
        ...

    Args:
        extractor:          FactExtractorFn — produces Assertions from turns
        session_id:         Identifier for this session (for provenance records)
        system_prompt:      Base system instructions
        buffer_size:        Recent message buffer (for LLM context)
        decay_half_life:    Turns after which an unreinforced fact halves in confidence
        injection_threshold: Minimum confidence score for injection into prompt
    """

    def __init__(
        self,
        extractor: FactExtractorFn,
        session_id: str = "default",
        system_prompt: str = "You are a helpful assistant.",
        buffer_size: int = 8,
        decay_half_life: float = 20.0,
        injection_threshold: float = 0.40,
    ) -> None:
        self.extractor = extractor
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.buffer_size = buffer_size
        self.store = ProvenanceStore(
            decay_half_life=decay_half_life,
            injection_threshold=injection_threshold,
        )
        self._buffer: list[dict[str, str]] = []
        self._turn_index: int = 0
        self._assertion_log: list[tuple[Assertion, str]] = []   # (assertion, outcome)

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> list[tuple[Assertion, str]]:
        """
        Process a user turn: extract facts, integrate into store, apply decay.
        Returns list of (assertion, outcome) pairs for the demo to display.
        """
        self._turn_index += 1
        assertions = self.extractor("user", content, self._turn_index)
        results: list[tuple[Assertion, str]] = []
        for a in assertions:
            outcome = self.store.assert_fact(a, self._turn_index, self.session_id)
            results.append((a, outcome))
            self._assertion_log.append((a, outcome))

        self.store.apply_decay(self._turn_index)
        self._push({"role": "user", "content": content})
        return results

    def add_assistant_message(self, content: str) -> None:
        self._push({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._buffer)

    def get_system_prompt(self) -> str:
        facts_section = self.store.format_for_prompt()
        if "no facts" in facts_section:
            return self.system_prompt
        return (
            f"{self.system_prompt}\n\n"
            f"## Known facts (confidence ≥ {self.store.injection_threshold:.0%})\n"
            f"{facts_section}"
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def turn_index(self) -> int:
        return self._turn_index

    @property
    def total_facts(self) -> int:
        return len(self.store)

    @property
    def injectable_count(self) -> int:
        return len(self.store.injectable_facts())

    @property
    def suppressed_count(self) -> int:
        return len(self.store.suppressed_facts())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _push(self, msg: dict[str, str]) -> None:
        self._buffer.append(msg)
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)
