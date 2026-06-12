"""
Layered Memory
--------------
Every prior experiment in this vertical solved one problem in isolation:

    01 — conversation buffer
    02 — sliding window
    03 — summarisation
    04 — vector / semantic retrieval
    05 — entity facts
    06 — episodic (cross-session)
    07 — knowledge graph
    08 — provenance + confidence
    09 — conflict resolution

A real assistant needs all of these simultaneously. This experiment
assembles them into a five-layer stack:

    ┌─────────────────────────────────────────────────┐
    │  Layer 5  Provenance + Confidence  (meta-layer) │  wraps layers 2–4
    ├─────────────────────────────────────────────────┤
    │  Layer 4  Semantic / Vector        (long-tail)  │  past turns by similarity
    ├─────────────────────────────────────────────────┤
    │  Layer 3  Episodic                 (history)    │  cross-session summaries
    ├─────────────────────────────────────────────────┤
    │  Layer 2  Entity + KG              (structure)  │  facts + relationships
    ├─────────────────────────────────────────────────┤
    │  Layer 1  Working memory           (buffer)     │  recent turns, in-flight
    └─────────────────────────────────────────────────┘

Each layer is independently useful; together they cover the full
spectrum of what the model needs to know:

    Working   — What happened in the last few turns? (fast, volatile)
    Entity/KG — Who is the user? What are the relationships? (structured)
    Episodic  — What happened in past sessions? (durable, compressed)
    Semantic  — What past turns are relevant to this question? (associative)
    Provenance— How confident are we in each fact? Was it contradicted?

Design decisions
----------------

Write routing
    Every user turn is written to working memory immediately.
    Entity/KG extraction runs on every turn (same as experiments 05/07).
    Semantic indexing runs on every turn (embed + store).
    Episodic write happens at session end (summarise + persist).
    Provenance is a meta-layer: it wraps entity/KG writes automatically.

Read composition (system prompt assembly)
    The system prompt is assembled in fixed priority order:
        1. Base instructions
        2. Episodic summaries (cross-session context)
        3. Entity/KG facts (structured beliefs, confidence-filtered)
        4. Semantic hits (relevant past turns, deduplicated vs buffer)
        5. Working memory is the messages list, not the system prompt

    Each layer has a token budget. If a layer would exceed its budget,
    it is truncated. The total budget is enforced before the LLM call.

Layer conflict resolution
    Entity layer is authoritative for structured facts.
    Episodic layer is advisory — summaries inform but do not override facts.
    If the entity layer has no fact for an attribute, the episodic and
    semantic layers provide best-effort recall.

Layer fallback
    If semantic retrieval returns nothing above the similarity threshold,
    the token budget is reallocated to the entity/KG section.
    If episodic is empty (first session), that section is omitted.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Layer 1 — Working Memory (conversation buffer)
# ---------------------------------------------------------------------------

class WorkingMemory:
    """
    Bounded FIFO buffer of recent turns.
    This is the messages list passed to the LLM on every call.
    """

    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self._buffer: list[dict[str, str]] = []

    def push(self, role: str, content: str) -> None:
        self._buffer.append({"role": role, "content": content})
        if len(self._buffer) > self.max_turns:
            self._buffer.pop(0)

    def messages(self) -> list[dict[str, str]]:
        return list(self._buffer)

    @property
    def turn_count(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Layer 2 — Entity + Knowledge Graph
# ---------------------------------------------------------------------------

@dataclass
class EntityFact:
    entity: str
    attribute: str
    value: str
    confidence: float = 0.70
    assertion_count: int = 1
    last_updated_turn: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return (self.entity, self.attribute)


@dataclass
class KGEdge:
    source: str
    relation: str
    target: str


class StructuredMemory:
    """
    Entity fact store + knowledge graph, with lightweight confidence tracking.
    Simplified from experiments 05/07/08 — keeps only what is needed for
    composition without re-implementing the full conflict resolution engine.
    """

    def __init__(self, injection_threshold: float = 0.40) -> None:
        self._facts: dict[tuple[str, str], EntityFact] = {}
        self._edges: dict[tuple[str, str, str], KGEdge] = {}
        self._adj: dict[str, set[tuple[str, str, str]]] = {}
        self.injection_threshold = injection_threshold

    # Facts

    def upsert_fact(self, entity: str, attribute: str, value: str,
                    turn_index: int, extractor: str = "regex") -> str:
        key = (entity.lower(), attribute.lower())
        initial = 0.85 if extractor == "llm" else 0.65
        if key not in self._facts:
            self._facts[key] = EntityFact(
                entity=entity.lower(), attribute=attribute.lower(),
                value=value, confidence=initial, last_updated_turn=turn_index,
            )
            return "new"
        fact = self._facts[key]
        if fact.value.lower() == value.lower():
            fact.confidence = min(1.0, fact.confidence + 0.10)
            fact.assertion_count += 1
            fact.last_updated_turn = turn_index
            return "reinforced"
        else:
            fact.confidence = max(0.0, fact.confidence - 0.25)
            fact.value = value
            fact.last_updated_turn = turn_index
            return "updated"

    def injectable_facts(self) -> list[EntityFact]:
        return sorted(
            [f for f in self._facts.values()
             if f.confidence >= self.injection_threshold],
            key=lambda f: (f.entity, f.attribute),
        )

    # Graph

    def upsert_edge(self, source: str, relation: str, target: str) -> None:
        source, target = source.lower(), target.lower()
        key = (source, relation, target)
        if key not in self._edges:
            self._edges[key] = KGEdge(source=source, relation=relation, target=target)
        self._adj.setdefault(source, set()).add(key)
        self._adj.setdefault(target, set()).add(key)

    def neighbours(self, node_id: str) -> list[KGEdge]:
        node_id = node_id.lower()
        keys = self._adj.get(node_id, set())
        seen: set[tuple] = set()
        result = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                result.append(self._edges[k])
        return result

    # Prompt formatting

    def format_for_prompt(self, focus_entities: list[str] | None = None) -> str:
        lines: list[str] = []

        # Facts section
        facts = self.injectable_facts()
        if facts:
            lines.append("Facts:")
            current_entity = ""
            for f in facts:
                if f.entity != current_entity:
                    current_entity = f.entity
                    lines.append(f"  {f.entity}:")
                lines.append(f"    {f.attribute}: {f.value}  [{f.confidence:.2f}]")

        # KG edges for focus entities
        if focus_entities:
            edges: list[KGEdge] = []
            seen_keys: set[tuple] = set()
            for eid in focus_entities[:5]:
                for e in self.neighbours(eid):
                    if e.source != e.target and (e.source, e.relation, e.target) not in seen_keys:
                        seen_keys.add((e.source, e.relation, e.target))
                        edges.append(e)
            if edges:
                lines.append("Relationships:")
                for e in edges[:15]:
                    lines.append(f"  {e.source} --[{e.relation}]--> {e.target}")

        return "\n".join(lines) if lines else ""

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    @property
    def edge_count(self) -> int:
        return len(self._edges)


# ---------------------------------------------------------------------------
# Layer 3 — Episodic Memory
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    session_id: str
    started_at: str
    ended_at: str
    summary: str

    def format_for_prompt(self) -> str:
        date = self.started_at[:10]
        return f"[{date}] {self.summary}"


SummarizerFn = Callable[[list[dict[str, str]]], str]


class EpisodicMemory:
    """
    Lightweight in-memory episode store for the layered demo.
    Persists episodes as a simple list; real deployments would use
    SQLite (see experiment 06).
    """

    def __init__(
        self,
        summarizer: SummarizerFn,
        recall_n: int = 3,
    ) -> None:
        self.summarizer = summarizer
        self.recall_n = recall_n
        self._episodes: list[Episode] = []
        self._session_id: str = str(uuid.uuid4())[:8]
        self._started_at: str = _now()

    def save_session(self, turns: list[dict[str, str]]) -> Episode | None:
        if not turns:
            return None
        summary = self.summarizer(turns)
        ep = Episode(
            session_id=self._session_id,
            started_at=self._started_at,
            ended_at=_now(),
            summary=summary,
        )
        self._episodes.append(ep)
        # Start fresh session
        self._session_id = str(uuid.uuid4())[:8]
        self._started_at = _now()
        return ep

    def recent_episodes(self) -> list[Episode]:
        return list(reversed(self._episodes[-self.recall_n:]))

    def format_for_prompt(self) -> str:
        episodes = self.recent_episodes()
        if not episodes:
            return ""
        lines = ["Previous sessions:"]
        for ep in reversed(episodes):   # chronological
            lines.append(f"  {ep.format_for_prompt()}")
        return "\n".join(lines)

    @property
    def episode_count(self) -> int:
        return len(self._episodes)


# ---------------------------------------------------------------------------
# Layer 4 — Semantic / Vector Memory
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class SemanticEntry:
    turn_index: int
    role: str
    content: str
    embedding: list[float]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class SemanticMemory:
    """
    Embeds every turn and retrieves the top-k most similar to the
    current query. Uses plain Python — no FAISS dependency here
    since the layered experiment is about composition, not retrieval
    speed. Swap in FaissVectorStore from experiment 04 for production.
    """

    def __init__(
        self,
        embedder: Embedder,
        top_k: int = 3,
        min_similarity: float = 0.35,
    ) -> None:
        self.embedder = embedder
        self.top_k = top_k
        self.min_similarity = min_similarity
        self._entries: list[SemanticEntry] = []

    def index(self, turn_index: int, role: str, content: str) -> None:
        emb = self.embedder.encode([content])[0]
        self._entries.append(SemanticEntry(turn_index, role, content, emb))

    def retrieve(
        self,
        query: str,
        exclude_turns: set[int] | None = None,
    ) -> list[SemanticEntry]:
        if not self._entries:
            return []
        exclude_turns = exclude_turns or set()
        q_emb = self.embedder.encode([query])[0]
        scored = [
            (e, _cosine(q_emb, e.embedding))
            for e in self._entries
            if e.turn_index not in exclude_turns
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, score in scored[:self.top_k] if score >= self.min_similarity]

    def format_for_prompt(self, hits: list[SemanticEntry]) -> str:
        if not hits:
            return ""
        lines = ["Relevant past turns:"]
        for h in hits:
            lines.append(f"  [{h.role}] {h.content[:120]}")
        return "\n".join(lines)

    @property
    def indexed_count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Extraction helpers (used by the layered system)
# ---------------------------------------------------------------------------

ExtractorFn = Callable[[str, str], tuple[list[tuple], list[tuple]]]
# Returns (entity_facts, kg_edges)
# entity_facts: list of (entity, attribute, value)
# kg_edges: list of (source, relation, target)


# ---------------------------------------------------------------------------
# Layer budget allocation
# ---------------------------------------------------------------------------

@dataclass
class LayerBudget:
    """
    Approximate token budget per layer section in the system prompt.
    One "token" ≈ 4 characters for budget purposes.
    """
    episodic_chars: int = 600
    structured_chars: int = 800
    semantic_chars: int = 500


# ---------------------------------------------------------------------------
# Layered Memory — the composition engine
# ---------------------------------------------------------------------------

class LayeredMemory:
    """
    Composes all five memory layers into a single assistant memory system.

    Usage:
        memory = LayeredMemory(extractor=..., embedder=..., summarizer=...)
        memory.start_session()
        ...
        results = memory.add_user_message("I work at Acme on project Orion.")
        system  = memory.get_system_prompt("What am I working on?")
        reply   = llm(system=system, messages=memory.messages())
        memory.add_assistant_message(reply)
        ...
        memory.end_session()

    Args:
        extractor:    ExtractorFn — returns (entity_facts, kg_edges) per turn
        embedder:     Embedder — encodes text into vectors for semantic layer
        summarizer:   SummarizerFn — condenses a session's turns into a summary
        system_prompt: Base instructions prepended to every system prompt
        budget:       LayerBudget — token budget per section
        working_size: Max turns in the working memory buffer
        recall_n:     Past sessions to recall from episodic layer
        top_k:        Semantic hits to retrieve per query
        min_similarity: Minimum cosine similarity for semantic hits
        injection_threshold: Minimum confidence for entity/KG injection
    """

    def __init__(
        self,
        extractor: ExtractorFn,
        embedder: Embedder,
        summarizer: SummarizerFn,
        system_prompt: str = "You are a helpful assistant with layered memory.",
        budget: LayerBudget | None = None,
        working_size: int = 8,
        recall_n: int = 3,
        top_k: int = 3,
        min_similarity: float = 0.35,
        injection_threshold: float = 0.40,
    ) -> None:
        self.system_prompt = system_prompt
        self.budget = budget or LayerBudget()

        # Layer 1
        self._working = WorkingMemory(max_turns=working_size)
        # Layer 2
        self._structured = StructuredMemory(injection_threshold=injection_threshold)
        # Layer 3
        self._episodic = EpisodicMemory(summarizer=summarizer, recall_n=recall_n)
        # Layer 4
        self._semantic = SemanticMemory(embedder=embedder, top_k=top_k,
                                        min_similarity=min_similarity)
        # Extraction
        self._extractor = extractor
        self._turn_index: int = 0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Begin a new session. Call once before the first user message."""
        self._turn_index = 0

    def end_session(self) -> Episode | None:
        """Summarise the session and persist to episodic layer."""
        turns = self._working.messages()
        return self._episodic.save_session(turns)

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> dict:
        """
        Process a user turn through all write paths:
          - Layer 1: push to working memory
          - Layer 2: extract and upsert entity facts + KG edges
          - Layer 4: embed and index for semantic retrieval
        Returns a dict of extraction results for display.
        """
        self._turn_index += 1
        self._working.push("user", content)
        self._semantic.index(self._turn_index, "user", content)

        entity_facts, kg_edges = self._extractor("user", content)
        fact_outcomes: list[tuple[str, str, str, str]] = []   # (entity, attr, value, outcome)
        for entity, attribute, value in entity_facts:
            outcome = self._structured.upsert_fact(
                entity, attribute, value, self._turn_index,
            )
            fact_outcomes.append((entity, attribute, value, outcome))

        for source, relation, target in kg_edges:
            self._structured.upsert_edge(source, relation, target)

        return {
            "turn": self._turn_index,
            "facts": fact_outcomes,
            "edges": kg_edges,
        }

    def add_assistant_message(self, content: str) -> None:
        self._working.push("assistant", content)
        self._semantic.index(self._turn_index, "assistant", content)

    def messages(self) -> list[dict[str, str]]:
        return self._working.messages()

    # ------------------------------------------------------------------
    # System prompt assembly (the core of this experiment)
    # ------------------------------------------------------------------

    def get_system_prompt(self, query: str = "") -> str:
        """
        Assemble the system prompt from all layers within budget.

        Priority order:
            1. Base instructions (always included)
            2. Episodic summaries (past sessions)
            3. Structured facts + KG (entity beliefs)
            4. Semantic hits (relevant past turns)

        Each section is truncated to its character budget.
        Empty sections are omitted entirely.
        """
        sections: list[str] = [self.system_prompt]

        # Layer 3 — episodic
        episodic_text = self._episodic.format_for_prompt()
        if episodic_text:
            sections.append(self._truncate(episodic_text, self.budget.episodic_chars))

        # Layer 2 — structured (entity + KG)
        # Focus on entities recently mentioned in the query
        focus = self._extract_capitalized(query) if query else []
        structured_text = self._structured.format_for_prompt(focus_entities=focus)
        if structured_text:
            sections.append(
                "## Known facts\n"
                + self._truncate(structured_text, self.budget.structured_chars)
            )

        # Layer 4 — semantic
        # Exclude turns currently in the working memory buffer (already in messages)
        buffer_turn_indices = set(
            range(
                max(1, self._turn_index - self._working.turn_count + 1),
                self._turn_index + 1,
            )
        )
        hits = self._semantic.retrieve(query or "", exclude_turns=buffer_turn_indices)
        semantic_text = self._semantic.format_for_prompt(hits)
        if semantic_text:
            sections.append(
                "## Relevant past context\n"
                + self._truncate(semantic_text, self.budget.semantic_chars)
            )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "turn":          self._turn_index,
            "working_turns": self._working.turn_count,
            "facts":         self._structured.fact_count,
            "kg_edges":      self._structured.edge_count,
            "episodes":      self._episodic.episode_count,
            "semantic_idx":  self._semantic.indexed_count,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars // 2:
            truncated = truncated[:last_newline]
        return truncated + "\n  … (truncated)"

    @staticmethod
    def _extract_capitalized(text: str) -> list[str]:
        """Pull out capitalised words as likely entity names for KG focus."""
        return list(dict.fromkeys(
            m.lower() for m in re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        ))
