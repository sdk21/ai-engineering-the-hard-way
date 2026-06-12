"""
Conflict Resolution Memory
--------------------------
Experiment 08 detected conflicts and applied a confidence penalty.
That is the minimum viable response to a contradiction. In practice,
conflicts in a memory store are not all the same thing:

    "I live in Tokyo."   →   "I moved to Osaka."
        — legitimate update; the old value is no longer true

    "My name is Alice."  →   "My name is Alicia."
        — possible correction of a prior error (the extractor mis-heard)
        — or a genuine name change; rare but valid

    "I'm a software engineer."  →  "I'm a senior software engineer."
        — partial match / refinement; not a true contradiction

    "I no longer work at Acme."
        — explicit retraction; the fact should be removed or deprecated

    Analyst note: employer = "Acme"   →   user says: employer = "Beta Corp"
        — source authority conflict: manual (high-trust) vs LLM extraction

Each case deserves a different resolution strategy. This experiment
implements a configurable resolution engine with five strategies, plus
a `ConflictRecord` that stores the full decision trail so every
resolution is auditable.

Resolution strategies
---------------------

LAST_WRITE      — most recently asserted value wins (Experiment 08 default)
HIGHEST_CONF    — whichever value currently has a higher confidence score wins
MOST_FREQUENT   — whichever value has been asserted more times wins
SOURCE_PRIORITY — values from higher-authority sources win regardless of
                  recency (manual > llm > regex)
USER_ARBITRATION — pause and record an UNRESOLVED conflict; the application
                   must call resolve_conflict() to supply the winner

Each strategy can be set globally on the store or overridden per-attribute,
allowing mixed behaviour:

    store = ConflictResolutionStore(
        default_strategy=Strategy.LAST_WRITE,
        attribute_strategies={
            "name":     Strategy.HIGHEST_CONF,   # corrections are rare
            "employer": Strategy.SOURCE_PRIORITY, # manual notes trump extractions
            "location": Strategy.LAST_WRITE,      # users move; accept updates
        }
    )

Special cases
-------------

Temporal updates — if the incoming text contains temporal language
("now", "moved to", "as of", "no longer", "used to") the conflict is
classified as an UPDATE rather than a CONFLICT, and no confidence
penalty is applied to the incoming value.

Explicit retractions — "I no longer work at Acme" emits a Retraction
assertion (value=None). The store marks the fact as RETRACTED, removes
it from injectable facts, and records the retraction in the audit trail.

Partial matches — values within edit distance 1 of each other, or where
one is a substring of the other, are classified as REFINEMENTS rather
than full contradictions. A softer penalty applies.

Conflict propagation — when a fact is retracted or contradicted, any
dependent facts (those that reference the same entity via a known
dependency map) are marked STALE and their confidence is halved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, capped at 5 for performance."""
    a, b = a.lower()[:40], b.lower()[:40]
    if abs(len(a) - len(b)) > 5:
        return 6
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        ndp = [i + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (ca != cb), dp[j + 1] + 1, ndp[-1] + 1))
        dp = ndp
    return dp[-1]


# Temporal update markers — presence of these words signals a legitimate
# update rather than an error contradiction.
_TEMPORAL_MARKERS = frozenset([
    "now", "moved", "moving", "left", "leaving", "joined", "starting",
    "currently", "as of", "lately", "recently", "used to", "no longer",
    "not anymore", "just", "today", "this week", "this month",
])


def _is_temporal_update(source_text: str) -> bool:
    text = source_text.lower()
    return any(m in text for m in _TEMPORAL_MARKERS)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class Strategy(Enum):
    LAST_WRITE       = auto()   # most recent value wins
    HIGHEST_CONF     = auto()   # higher current confidence wins
    MOST_FREQUENT    = auto()   # more-asserted value wins
    SOURCE_PRIORITY  = auto()   # extractor authority wins
    USER_ARBITRATION = auto()   # leave unresolved until explicit call


# Source authority ranking (higher = more trusted)
SOURCE_AUTHORITY: dict[str, int] = {
    "manual": 100,
    "llm":     70,
    "regex":   40,
}


# ---------------------------------------------------------------------------
# Conflict classification
# ---------------------------------------------------------------------------

class ConflictType(Enum):
    UPDATE     = "update"       # temporal language present — legitimate change
    CORRECTION = "correction"   # near-match or same-entity refinement
    CONFLICT   = "conflict"     # genuine contradiction, no cue either way
    RETRACTION = "retraction"   # explicit "no longer" / None value


def classify_conflict(
    old_value: str,
    new_value: str | None,
    source_text: str,
) -> ConflictType:
    if new_value is None:
        return ConflictType.RETRACTION
    if _is_temporal_update(source_text):
        return ConflictType.UPDATE
    # Near-match: edit distance ≤ 2 or substring
    ov, nv = old_value.lower(), new_value.lower()
    if _edit_distance(ov, nv) <= 2 or ov in nv or nv in ov:
        return ConflictType.CORRECTION
    return ConflictType.CONFLICT


# Confidence deltas by conflict type
CONFIDENCE_DELTA: dict[ConflictType, float] = {
    ConflictType.UPDATE:     0.00,   # no penalty — it's just a change
    ConflictType.CORRECTION: -0.10,  # soft penalty — possible extractor error
    ConflictType.CONFLICT:   -0.30,  # hard penalty — genuine disagreement
    ConflictType.RETRACTION:  0.00,  # no penalty — fact is simply deprecated
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Provenance:
    turn_index: int
    session_id: str
    timestamp: str
    source_text: str
    extractor: str      # "regex" | "llm" | "manual"
    assertion_count: int = 1   # how many times this exact value was seen


@dataclass
class ConflictRecord:
    """Full record of a conflict event and how it was resolved."""
    turn_index: int
    conflict_type: ConflictType
    old_value: str
    new_value: str | None
    strategy_used: Strategy
    winner: str | None          # None if USER_ARBITRATION and unresolved
    old_confidence: float
    new_confidence: float
    timestamp: str = field(default_factory=_now)
    notes: str = ""


class FactStatus(Enum):
    ACTIVE    = "active"
    RETRACTED = "retracted"
    STALE     = "stale"       # a dependency was contradicted
    UNRESOLVED = "unresolved" # USER_ARBITRATION pending


@dataclass
class ConfidenceScore:
    score: float
    assertion_count: int = 1
    last_seen_turn: int = 0
    last_decay_turn: int = 0

    def reinforce(self, delta: float = 0.10) -> None:
        self.score = min(1.0, self.score + delta)
        self.assertion_count += 1

    def penalise(self, delta: float) -> None:
        self.score = max(0.0, self.score - delta)

    def decay(self, current_turn: int, half_life: float) -> None:
        if self.last_seen_turn == current_turn:
            self.last_decay_turn = current_turn
            return
        incremental = current_turn - self.last_decay_turn
        if incremental <= 0:
            return
        self.score *= math.pow(0.5, incremental / half_life)
        self.last_decay_turn = current_turn

    @property
    def label(self) -> str:
        if self.score >= 0.8:   return "high"
        if self.score >= 0.5:   return "medium"
        if self.score >= 0.2:   return "low"
        return "very_low"


@dataclass
class FactRecord:
    entity: str
    attribute: str
    value: str | None           # None = retracted
    status: FactStatus
    confidence: ConfidenceScore
    provenance: list[Provenance] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.entity, self.attribute)

    def format_for_prompt(self) -> str:
        return f"{self.entity}.{self.attribute} = {self.value!r}  [{self.confidence.score:.2f}]"

    def format_for_audit(self, injection_threshold: float = 0.4) -> str:
        suppressed = "  ⚠ SUPPRESSED" if (
            self.confidence.score < injection_threshold
            or self.status != FactStatus.ACTIVE
        ) else ""
        lines = [
            f"  {self.entity}.{self.attribute} = {self.value!r}  "
            f"[{self.status.value}]{suppressed}",
            f"    confidence   : {self.confidence.score:.3f} ({self.confidence.label})"
            f"  asserted×{self.confidence.assertion_count}",
        ]
        for p in self.provenance:
            lines.append(
                f"    provenance   : turn {p.turn_index} [{p.extractor}]"
                f" \"{p.source_text[:60].rstrip()}\""
            )
        for c in self.conflicts:
            if c.conflict_type == ConflictType.RETRACTION:
                winner_note = "→ retracted"
            elif c.winner is None:
                winner_note = "→ UNRESOLVED (awaiting arbitration)"
            elif c.winner == c.old_value:
                winner_note = f"→ kept old {c.winner!r}"
            else:
                winner_note = f"→ updated to {c.winner!r}"
            turn_label = f"turn {c.turn_index}" if c.turn_index >= 0 else "propagated"
            lines.append(
                f"    conflict     : {turn_label} [{c.conflict_type.value}]"
                f" old={c.old_value!r} new={c.new_value!r}"
                f" via {c.strategy_used.name} {winner_note}"
            )
        return "\n".join(lines)


# An Assertion is what the extractor emits. value=None means retraction.
@dataclass
class Assertion:
    entity: str
    attribute: str
    value: str | None
    source_text: str
    extractor: str
    initial_confidence: float = 0.70


FactExtractorFn = Callable[[str, str, int], list["Assertion"]]

INITIAL_CONFIDENCE = {"manual": 1.0, "llm": 0.85, "regex": 0.65}


# ---------------------------------------------------------------------------
# Conflict resolution store
# ---------------------------------------------------------------------------

class ConflictResolutionStore:
    """
    Stores (entity, attribute) → FactRecord with configurable conflict
    resolution strategies.

    Args:
        default_strategy:    Strategy applied when no attribute-specific
                             override is set.
        attribute_strategies: Per-attribute strategy overrides.
        dependency_map:       {(entity, attribute) → list of (entity, attribute)}
                             When a fact is contradicted or retracted, all
                             dependents are marked STALE.
        injection_threshold: Minimum score for injection into the system prompt.
        decay_half_life:     Turn-based half-life for confidence decay.
    """

    def __init__(
        self,
        default_strategy: Strategy = Strategy.LAST_WRITE,
        attribute_strategies: dict[str, Strategy] | None = None,
        dependency_map: dict[tuple[str, str], list[tuple[str, str]]] | None = None,
        injection_threshold: float = 0.40,
        decay_half_life: float = 20.0,
    ) -> None:
        self.default_strategy = default_strategy
        self.attribute_strategies = attribute_strategies or {}
        self.dependency_map = dependency_map or {}
        self.injection_threshold = injection_threshold
        self.decay_half_life = decay_half_life
        self._facts: dict[tuple[str, str], FactRecord] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def assert_fact(
        self,
        assertion: Assertion,
        turn_index: int,
        session_id: str = "default",
    ) -> tuple[str, ConflictRecord | None]:
        """
        Integrate an assertion into the store using the configured strategy.

        Returns (outcome, conflict_record_or_None).
        outcome is one of: "new", "reinforced", "update", "correction",
                           "conflict", "retraction", "unresolved"
        """
        key = (assertion.entity.lower(), assertion.attribute.lower())
        prov = Provenance(
            turn_index=turn_index,
            session_id=session_id,
            timestamp=_now(),
            source_text=assertion.source_text,
            extractor=assertion.extractor,
        )

        # --- New fact ---
        if key not in self._facts:
            initial = INITIAL_CONFIDENCE.get(assertion.extractor, assertion.initial_confidence)
            record = FactRecord(
                entity=assertion.entity.lower(),
                attribute=assertion.attribute.lower(),
                value=assertion.value,
                status=FactStatus.ACTIVE if assertion.value is not None else FactStatus.RETRACTED,
                confidence=ConfidenceScore(
                    score=initial,
                    last_seen_turn=turn_index,
                    last_decay_turn=turn_index,
                ),
                provenance=[prov],
            )
            self._facts[key] = record
            return "new", None

        record = self._facts[key]

        # --- Retraction ---
        if assertion.value is None or _is_retraction(assertion.source_text):
            return self._handle_retraction(record, prov, turn_index)

        # --- Reinforcement (same value) ---
        if record.value is not None and assertion.value.lower() == record.value.lower():
            record.confidence.reinforce(0.10)
            record.confidence.last_seen_turn = turn_index
            record.provenance.append(prov)
            return "reinforced", None

        # --- Conflict: different value ---
        return self._handle_conflict(record, assertion, prov, turn_index)

    # ------------------------------------------------------------------
    # Conflict handler
    # ------------------------------------------------------------------

    def _handle_conflict(
        self,
        record: FactRecord,
        assertion: Assertion,
        prov: Provenance,
        turn_index: int,
    ) -> tuple[str, ConflictRecord]:
        conflict_type = classify_conflict(
            old_value=record.value or "",
            new_value=assertion.value,
            source_text=assertion.source_text,
        )
        strategy = self.attribute_strategies.get(
            record.attribute, self.default_strategy
        )
        old_score = record.confidence.score
        penalty = CONFIDENCE_DELTA[conflict_type]

        # Determine winner
        if strategy == Strategy.LAST_WRITE:
            winner = assertion.value

        elif strategy == Strategy.HIGHEST_CONF:
            incoming_initial = INITIAL_CONFIDENCE.get(assertion.extractor, 0.65)
            # incoming fact hasn't been reinforced yet — compare its initial score
            winner = assertion.value if incoming_initial >= record.confidence.score else record.value

        elif strategy == Strategy.MOST_FREQUENT:
            incoming_count = 1
            winner = assertion.value if incoming_count >= record.confidence.assertion_count else record.value

        elif strategy == Strategy.SOURCE_PRIORITY:
            old_auth = SOURCE_AUTHORITY.get(record.provenance[-1].extractor, 0)
            new_auth = SOURCE_AUTHORITY.get(assertion.extractor, 0)
            winner = assertion.value if new_auth >= old_auth else record.value

        elif strategy == Strategy.USER_ARBITRATION:
            winner = None   # unresolved — caller must invoke resolve_conflict()
        else:
            winner = assertion.value

        conflict = ConflictRecord(
            turn_index=turn_index,
            conflict_type=conflict_type,
            old_value=record.value or "",
            new_value=assertion.value,
            strategy_used=strategy,
            winner=winner,
            old_confidence=old_score,
            new_confidence=max(0.0, old_score + penalty),
        )

        # Apply the outcome
        record.provenance.append(prov)
        record.conflicts.append(conflict)
        record.confidence.penalise(abs(penalty))
        record.confidence.last_seen_turn = turn_index

        if strategy == Strategy.USER_ARBITRATION:
            record.status = FactStatus.UNRESOLVED
            # Store the candidate new value in provenance but don't update value yet
        else:
            if winner != record.value:
                record.value = winner
                self._propagate_stale(record.key)

        if conflict_type == ConflictType.UPDATE:
            record.status = FactStatus.ACTIVE
            return "update", conflict
        elif strategy == Strategy.USER_ARBITRATION:
            return "unresolved", conflict
        else:
            return conflict_type.value, conflict

    def _handle_retraction(
        self,
        record: FactRecord,
        prov: Provenance,
        turn_index: int,
    ) -> tuple[str, ConflictRecord]:
        conflict = ConflictRecord(
            turn_index=turn_index,
            conflict_type=ConflictType.RETRACTION,
            old_value=record.value or "",
            new_value=None,
            strategy_used=self.default_strategy,
            winner=None,
            old_confidence=record.confidence.score,
            new_confidence=0.0,
            notes="explicit retraction",
        )
        record.value = None
        record.status = FactStatus.RETRACTED
        record.confidence.score = 0.0
        record.provenance.append(prov)
        record.conflicts.append(conflict)
        self._propagate_stale(record.key)
        return "retraction", conflict

    def _propagate_stale(self, changed_key: tuple[str, str]) -> None:
        """Mark all facts that depend on the changed fact as STALE."""
        for dep_key in self.dependency_map.get(changed_key, []):
            dep = self._facts.get(dep_key)
            if dep and dep.status == FactStatus.ACTIVE:
                dep.status = FactStatus.STALE
                dep.confidence.penalise(0.20)
                dep.conflicts.append(ConflictRecord(
                    turn_index=-1,
                    conflict_type=ConflictType.CONFLICT,
                    old_value=dep.value or "",
                    new_value=dep.value,
                    strategy_used=self.default_strategy,
                    winner=dep.value,
                    old_confidence=dep.confidence.score + 0.20,
                    new_confidence=dep.confidence.score,
                    notes=f"marked stale: dependency {changed_key} changed",
                ))

    # ------------------------------------------------------------------
    # User arbitration
    # ------------------------------------------------------------------

    def resolve_conflict(
        self,
        entity: str,
        attribute: str,
        chosen_value: str,
    ) -> bool:
        """
        Resolve a USER_ARBITRATION conflict by supplying the winning value.
        Returns True if a pending conflict was found and resolved.
        """
        key = (entity.lower(), attribute.lower())
        record = self._facts.get(key)
        if not record or record.status != FactStatus.UNRESOLVED:
            return False

        # Find the most recent unresolved conflict
        for conflict in reversed(record.conflicts):
            if conflict.winner is None:
                conflict.winner = chosen_value
                conflict.notes = "resolved by user arbitration"
                break

        record.value = chosen_value
        record.status = FactStatus.ACTIVE
        record.confidence.score = min(1.0, record.confidence.score + 0.20)
        return True

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def apply_decay(self, current_turn: int) -> None:
        for record in self._facts.values():
            if record.status == FactStatus.ACTIVE:
                record.confidence.decay(current_turn, self.decay_half_life)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def injectable_facts(self) -> list[FactRecord]:
        return sorted(
            [r for r in self._facts.values()
             if r.status == FactStatus.ACTIVE
             and r.value is not None
             and r.confidence.score >= self.injection_threshold],
            key=lambda r: (r.entity, r.attribute),
        )

    def all_facts(self) -> list[FactRecord]:
        return sorted(self._facts.values(), key=lambda r: (r.entity, r.attribute))

    def unresolved_conflicts(self) -> list[FactRecord]:
        return [r for r in self._facts.values() if r.status == FactStatus.UNRESOLVED]

    def get(self, entity: str, attribute: str) -> FactRecord | None:
        return self._facts.get((entity.lower(), attribute.lower()))

    def __len__(self) -> int:
        return len(self._facts)

    def format_for_prompt(self) -> str:
        facts = self.injectable_facts()
        if not facts:
            return "(no facts above confidence threshold)"
        lines: list[str] = []
        current_entity = ""
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
        lines: list[str] = []
        current_entity = ""
        for r in facts:
            if r.entity != current_entity:
                current_entity = r.entity
                lines.append(f"\n{r.entity}:")
            lines.append(r.format_for_audit(self.injection_threshold))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retraction detection
# ---------------------------------------------------------------------------

_RETRACTION_PHRASES = frozenset([
    "no longer", "not anymore", "don't work", "doesn't work", "left ",
    "quit ", "stopped ", "retired", "resigned", "moved away", "gave up",
])


def _is_retraction(source_text: str) -> bool:
    text = source_text.lower()
    return any(p in text for p in _RETRACTION_PHRASES)


# ---------------------------------------------------------------------------
# ConflictResolutionMemory
# ---------------------------------------------------------------------------

class ConflictResolutionMemory:
    """
    Full memory system with pluggable conflict resolution strategies.

    Args:
        extractor:           FactExtractorFn
        store:               ConflictResolutionStore (pre-configured with strategies)
        session_id:          For provenance records
        system_prompt:       Base instructions
        buffer_size:         Recent message buffer size
    """

    def __init__(
        self,
        extractor: FactExtractorFn,
        store: ConflictResolutionStore | None = None,
        session_id: str = "default",
        system_prompt: str = "You are a helpful assistant.",
        buffer_size: int = 8,
    ) -> None:
        self.extractor = extractor
        self.store = store if store is not None else ConflictResolutionStore()
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.buffer_size = buffer_size
        self._buffer: list[dict[str, str]] = []
        self._turn_index: int = 0

    def add_user_message(self, content: str) -> list[tuple[Assertion, str, ConflictRecord | None]]:
        self._turn_index += 1
        assertions = self.extractor("user", content, self._turn_index)
        results = []
        for a in assertions:
            outcome, conflict = self.store.assert_fact(a, self._turn_index, self.session_id)
            results.append((a, outcome, conflict))
        self.store.apply_decay(self._turn_index)
        self._push({"role": "user", "content": content})
        return results

    def add_assistant_message(self, content: str) -> None:
        self._push({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._buffer)

    def get_system_prompt(self) -> str:
        facts_section = self.store.format_for_prompt()
        unresolved = self.store.unresolved_conflicts()
        parts = [self.system_prompt]
        if "no facts" not in facts_section:
            parts.append(
                f"## Known facts (confidence ≥ {self.store.injection_threshold:.0%})\n"
                + facts_section
            )
        if unresolved:
            attrs = ", ".join(f"{r.entity}.{r.attribute}" for r in unresolved)
            parts.append(
                f"## Unresolved conflicts (awaiting user input)\n  {attrs}"
            )
        return "\n\n".join(parts)

    @property
    def turn_index(self) -> int:
        return self._turn_index

    def _push(self, msg: dict[str, str]) -> None:
        self._buffer.append(msg)
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)
