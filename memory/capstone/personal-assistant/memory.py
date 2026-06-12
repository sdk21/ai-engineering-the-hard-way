"""
Persistent Layered Memory — production-grade memory system.

Implements all five layers from experiment 10 with production concerns:

    Persistence     — SQLite for facts + episodes; JSON for vector index.
                      Memory survives process restarts, reboots, deployments.

    Prompt caching  — System prompt split into a STABLE PREFIX (base
                      instructions + episodic + structured facts) and a
                      DYNAMIC SUFFIX (semantic hits). The stable prefix
                      is suitable for Anthropic's prompt caching
                      (cache_control: {"type": "ephemeral"}) — you pay
                      full price once per cache TTL (~5 min), then ~10%
                      on subsequent calls. Only the dynamic suffix is
                      rebuilt on every turn.

    TTL cleanup     — Facts with confidence below a floor and last-seen
                      older than N days are pruned on startup. Episodes
                      older than the retention window are archived.

    Session resume  — Sessions are identified by UUID and persisted.
                      On startup the last open session is resumed if
                      it is within the resume window; otherwise a new
                      session is started.

    Async writes    — Extraction and indexing are designed to be called
                      after the LLM response is returned, not before.
                      The LLM call uses the system prompt built from the
                      state BEFORE the current turn — consistent with
                      how production systems handle latency.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Protocol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _now_dt() -> datetime:
    return datetime.now(timezone.utc)

def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...

SummarizerFn = Callable[[list[dict[str, str]]], str]
ExtractorFn  = Callable[[str, str], tuple[list[tuple], list[tuple]]]


# ---------------------------------------------------------------------------
# Persistent fact store — SQLite
# ---------------------------------------------------------------------------

class PersistentFactStore:
    """
    SQLite-backed entity fact store with confidence tracking.

    Schema:
        facts(entity, attribute, value, confidence, assertion_count,
              last_seen, created_at, source_text, extractor)

        episodes(session_id PK, started_at, ended_at, summary)
        turns(id, session_id FK, role, content, timestamp)
    """

    INITIAL_CONFIDENCE = {"llm": 0.85, "regex": 0.65, "manual": 1.0}

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    entity          TEXT NOT NULL,
                    attribute       TEXT NOT NULL,
                    value           TEXT NOT NULL,
                    confidence      REAL NOT NULL DEFAULT 0.65,
                    assertion_count INTEGER NOT NULL DEFAULT 1,
                    last_seen       TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    source_text     TEXT NOT NULL DEFAULT '',
                    extractor       TEXT NOT NULL DEFAULT 'regex',
                    PRIMARY KEY (entity, attribute)
                );
                CREATE INDEX IF NOT EXISTS idx_facts_confidence
                    ON facts(confidence DESC);
                CREATE INDEX IF NOT EXISTS idx_facts_last_seen
                    ON facts(last_seen DESC);

                CREATE TABLE IF NOT EXISTS episodes (
                    session_id  TEXT PRIMARY KEY,
                    started_at  TEXT NOT NULL,
                    ended_at    TEXT NOT NULL,
                    summary     TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_started
                    ON episodes(started_at DESC);

                CREATE TABLE IF NOT EXISTS turns (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL REFERENCES episodes(session_id),
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    timestamp   TEXT NOT NULL
                );
            """)

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def upsert_fact(
        self,
        entity: str,
        attribute: str,
        value: str,
        source_text: str = "",
        extractor: str = "regex",
    ) -> str:
        entity, attribute = entity.lower(), attribute.lower()
        initial = self.INITIAL_CONFIDENCE.get(extractor, 0.65)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, confidence, assertion_count FROM facts WHERE entity=? AND attribute=?",
                (entity, attribute),
            ).fetchone()

            if row is None:
                conn.execute(
                    "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?)",
                    (entity, attribute, value, initial, 1, _now(), _now(), source_text, extractor),
                )
                return "new"

            if row["value"].lower() == value.lower():
                new_conf = min(1.0, row["confidence"] + 0.10)
                conn.execute(
                    "UPDATE facts SET confidence=?, assertion_count=?, last_seen=?, source_text=? "
                    "WHERE entity=? AND attribute=?",
                    (new_conf, row["assertion_count"] + 1, _now(), source_text, entity, attribute),
                )
                return "reinforced"
            else:
                new_conf = max(0.0, row["confidence"] - 0.25)
                conn.execute(
                    "UPDATE facts SET value=?, confidence=?, last_seen=?, source_text=?, extractor=? "
                    "WHERE entity=? AND attribute=?",
                    (value, new_conf, _now(), source_text, extractor, entity, attribute),
                )
                return "updated"

    def injectable_facts(self, threshold: float = 0.40) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT entity, attribute, value, confidence, assertion_count "
                "FROM facts WHERE confidence >= ? ORDER BY entity, attribute",
                (threshold,),
            ).fetchall()
        return [dict(r) for r in rows]

    def all_facts(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT entity, attribute, value, confidence, assertion_count, "
                "last_seen, source_text, extractor FROM facts ORDER BY entity, attribute",
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_fact(self, entity: str, attribute: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM facts WHERE entity=? AND attribute=?",
                (entity.lower(), attribute.lower()),
            )
            return cur.rowcount > 0

    def prune_stale(self, min_confidence: float = 0.15, max_age_days: int = 90) -> int:
        """Remove facts that are low-confidence AND old. Returns count deleted."""
        cutoff = (_now_dt() - timedelta(days=max_age_days)).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM facts WHERE confidence < ? AND last_seen < ?",
                (min_confidence, cutoff),
            )
            return cur.rowcount

    def fact_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def save_episode(
        self,
        session_id: str,
        started_at: str,
        ended_at: str,
        summary: str,
        turns: list[dict[str, str]],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?)",
                (session_id, started_at, ended_at, summary),
            )
            conn.executemany(
                "INSERT INTO turns(session_id, role, content, timestamp) VALUES (?,?,?,?)",
                [(session_id, t["role"], t["content"], _now()) for t in turns],
            )

    def load_recent_episodes(self, n: int = 3) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, started_at, ended_at, summary "
                "FROM episodes ORDER BY started_at DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]   # chronological

    def episode_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def prune_old_episodes(self, max_age_days: int = 365) -> int:
        cutoff = (_now_dt() - timedelta(days=max_age_days)).isoformat()
        with self._connect() as conn:
            # Delete turns first (FK)
            conn.execute(
                "DELETE FROM turns WHERE session_id IN "
                "(SELECT session_id FROM episodes WHERE started_at < ?)",
                (cutoff,),
            )
            cur = conn.execute("DELETE FROM episodes WHERE started_at < ?", (cutoff,))
            return cur.rowcount


# ---------------------------------------------------------------------------
# Persistent semantic index — JSON (swap for FAISS in high-volume deployments)
# ---------------------------------------------------------------------------

@dataclass
class SemanticEntry:
    session_id: str
    turn_index: int
    role: str
    content: str
    embedding: list[float]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class PersistentSemanticIndex:
    """
    JSON-backed semantic index. Loads all entries into RAM on startup;
    persists to disk after each index() call.

    For deployments > ~5,000 turns, replace with faiss.IndexFlatIP
    (see experiment 04) for O(log N) retrieval.
    """

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[SemanticEntry] = self._load()

    def _load(self) -> list[SemanticEntry]:
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text())
        return [SemanticEntry(**e) for e in data]

    def _save(self) -> None:
        data = [
            {"session_id": e.session_id, "turn_index": e.turn_index,
             "role": e.role, "content": e.content, "embedding": e.embedding}
            for e in self._entries
        ]
        self.index_path.write_text(json.dumps(data))

    def index(
        self,
        session_id: str,
        turn_index: int,
        role: str,
        content: str,
        embedder: Embedder,
    ) -> None:
        emb = embedder.encode([content])[0]
        self._entries.append(SemanticEntry(session_id, turn_index, role, content, emb))
        self._save()

    def retrieve(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 3,
        min_similarity: float = 0.35,
        exclude: set[tuple[str, int]] | None = None,   # (session_id, turn_index)
    ) -> list[SemanticEntry]:
        if not self._entries:
            return []
        exclude = exclude or set()
        q_emb = embedder.encode([query])[0]
        scored = [
            (e, _cosine(q_emb, e.embedding))
            for e in self._entries
            if (e.session_id, e.turn_index) not in exclude
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, s in scored[:top_k] if s >= min_similarity]

    def prune_old_sessions(self, keep_session_ids: set[str]) -> int:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.session_id in keep_session_ids]
        removed = before - len(self._entries)
        if removed:
            self._save()
        return removed

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    session_id: str
    started_at: str
    turns: list[dict[str, str]] = field(default_factory=list)
    turn_index: int = 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "turns": self.turns,
            "turn_index": self.turn_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        return cls(**d)


class SessionManager:
    """
    Persists the current session to a JSON sidecar file so it can be
    resumed after a process restart.
    """

    def __init__(self, session_path: Path, resume_window_hours: int = 4) -> None:
        self.session_path = session_path
        self.resume_window = timedelta(hours=resume_window_hours)
        session_path.parent.mkdir(parents=True, exist_ok=True)

    def load_or_new(self) -> tuple[SessionState, bool]:
        """
        Returns (session, was_resumed).
        Resumes if a session file exists and was started within the resume window.
        """
        if self.session_path.exists():
            try:
                state = SessionState.from_dict(json.loads(self.session_path.read_text()))
                age = _now_dt() - _parse_dt(state.started_at)
                if age <= self.resume_window:
                    return state, True
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        return self._new_session(), False

    def _new_session(self) -> SessionState:
        return SessionState(
            session_id=str(uuid.uuid4())[:8],
            started_at=_now(),
        )

    def save(self, state: SessionState) -> None:
        self.session_path.write_text(json.dumps(state.to_dict(), indent=2))

    def clear(self) -> None:
        if self.session_path.exists():
            self.session_path.unlink()


# ---------------------------------------------------------------------------
# Production layered memory
# ---------------------------------------------------------------------------

class ProductionMemory:
    """
    Production-grade layered memory with persistence, prompt caching
    structure, and TTL cleanup.

    The system prompt is split into two parts for caching efficiency:

        stable_prefix()   — base instructions + episodic + structured facts
                            Changes infrequently (only when facts/episodes update).
                            Pass this to cache_control: {"type": "ephemeral"}
                            on the Anthropic API for ~90% token cost reduction.

        dynamic_suffix()  — semantic hits for the current query
                            Changes every turn. Not worth caching.

        get_system_prompt(query) — concatenates both (for simple use cases)

    Args:
        data_dir:             Directory for all persistent files
        embedder:             Embedder for semantic indexing
        extractor:            ExtractorFn → (entity_facts, kg_edges)
        summarizer:           SummarizerFn → session summary string
        injection_threshold:  Min confidence for fact injection
        recall_n:             Past episodes to inject
        top_k:                Semantic hits per query
        min_similarity:       Min cosine similarity for semantic hits
        resume_window_hours:  Resume last session if started within this window
        prune_on_startup:     Run TTL cleanup when memory is initialised
    """

    STABLE_BUDGET  = 1200   # chars for episodic + structured
    SEMANTIC_BUDGET = 500   # chars for semantic hits

    def __init__(
        self,
        data_dir: Path,
        embedder: Embedder,
        extractor: ExtractorFn,
        summarizer: SummarizerFn,
        base_prompt: str = "You are a helpful personal assistant with persistent memory.",
        injection_threshold: float = 0.40,
        recall_n: int = 3,
        top_k: int = 3,
        min_similarity: float = 0.35,
        resume_window_hours: int = 4,
        prune_on_startup: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.embedder = embedder
        self.extractor = extractor
        self.summarizer = summarizer
        self.base_prompt = base_prompt
        self.injection_threshold = injection_threshold
        self.recall_n = recall_n
        self.top_k = top_k
        self.min_similarity = min_similarity

        self._store     = PersistentFactStore(data_dir / "memory.db")
        self._semantic  = PersistentSemanticIndex(data_dir / "semantic.json")
        self._sessions  = SessionManager(data_dir / "session.json", resume_window_hours)

        if prune_on_startup:
            self._run_ttl_cleanup()

        self._session, self._resumed = self._sessions.load_or_new()
        self._buffer: list[dict[str, str]] = list(self._session.turns)

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def was_resumed(self) -> bool:
        return self._resumed

    @property
    def turn_index(self) -> int:
        return self._session.turn_index

    def end_session(self) -> str | None:
        """Summarise and persist the current session. Returns the summary."""
        if not self._buffer:
            return None
        summary = self.summarizer(self._buffer)
        self._store.save_episode(
            session_id=self._session.session_id,
            started_at=self._session.started_at,
            ended_at=_now(),
            summary=summary,
            turns=self._buffer,
        )
        self._sessions.clear()
        return summary

    def new_session(self) -> None:
        """Start a fresh session without saving the current one."""
        self._session = SessionManager(
            self._sessions.session_path,
            int(self._sessions.resume_window.total_seconds() // 3600),
        )._new_session()
        self._buffer = []
        self._sessions.save(self._session)

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> dict:
        """
        Process a user turn.

        Write order:
            1. Push to buffer (layer 1)
            2. Semantic index (layer 4) — fast, runs before LLM call
            3. Extraction + structured upsert (layer 2) — runs AFTER
               the LLM response to keep the latency off the hot path.
               In this implementation both happen synchronously; a
               production system would run step 3 in a background thread.
        """
        self._session.turn_index += 1
        self._buffer.append({"role": "user", "content": content})

        # Layer 4: index immediately
        self._semantic.index(
            session_id=self._session.session_id,
            turn_index=self._session.turn_index,
            role="user",
            content=content,
            embedder=self.embedder,
        )

        # Layer 2: extract and upsert
        entity_facts, kg_edges = self.extractor("user", content)
        outcomes = []
        for entity, attribute, value in entity_facts:
            outcome = self._store.upsert_fact(entity, attribute, value,
                                              source_text=content)
            outcomes.append((entity, attribute, value, outcome))

        self._sessions.save(self._session)
        return {"facts": outcomes, "edges": kg_edges, "turn": self._session.turn_index}

    def add_assistant_message(self, content: str) -> None:
        self._buffer.append({"role": "assistant", "content": content})
        self._semantic.index(
            session_id=self._session.session_id,
            turn_index=self._session.turn_index,
            role="assistant",
            content=content,
            embedder=self.embedder,
        )
        self._sessions.save(self._session)

    def messages(self) -> list[dict[str, str]]:
        return list(self._buffer)

    # ------------------------------------------------------------------
    # System prompt (split for caching)
    # ------------------------------------------------------------------

    def stable_prefix(self) -> str:
        """
        The cacheable part of the system prompt.
        Contents: base instructions + episodic summaries + structured facts.

        This changes only when facts or episodes are updated — typically
        at most once per turn. Pass to Anthropic's API with:
            {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}
        """
        sections = [self.base_prompt]

        # Layer 3: episodic
        episodes = self._store.load_recent_episodes(self.recall_n)
        if episodes:
            lines = ["## Previous sessions"]
            for ep in episodes:
                date = ep["started_at"][:10]
                lines.append(f"  [{date}] {ep['summary']}")
            ep_text = "\n".join(lines)
            sections.append(self._truncate(ep_text, self.STABLE_BUDGET // 2))

        # Layer 2: structured facts
        facts = self._store.injectable_facts(self.injection_threshold)
        if facts:
            lines = ["## Known facts"]
            current_entity = ""
            for f in facts:
                if f["entity"] != current_entity:
                    current_entity = f["entity"]
                    lines.append(f"  {f['entity']}:")
                lines.append(
                    f"    {f['attribute']}: {f['value']}  [{f['confidence']:.2f}]"
                )
            fact_text = "\n".join(lines)
            sections.append(self._truncate(fact_text, self.STABLE_BUDGET // 2))

        return "\n\n".join(sections)

    def dynamic_suffix(self, query: str) -> str:
        """
        The per-query part of the system prompt: semantic hits.
        Rebuilt on every turn. Not suitable for prompt caching.
        """
        if not query:
            return ""

        # Exclude turns from the current buffer (already in messages)
        buffer_pairs = {
            (self._session.session_id, self._session.turn_index - i)
            for i in range(len(self._buffer))
        }
        hits = self._semantic.retrieve(
            query=query,
            embedder=self.embedder,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
            exclude=buffer_pairs,
        )
        if not hits:
            return ""

        lines = ["## Relevant past context"]
        for h in hits:
            lines.append(f"  [{h.role}] {h.content[:120]}")
        return self._truncate("\n".join(lines), self.SEMANTIC_BUDGET)

    def get_system_prompt(self, query: str = "") -> str:
        """Concatenate stable prefix + dynamic suffix (simple API)."""
        prefix = self.stable_prefix()
        suffix = self.dynamic_suffix(query)
        return f"{prefix}\n\n{suffix}".strip() if suffix else prefix

    # ------------------------------------------------------------------
    # Memory inspection
    # ------------------------------------------------------------------

    def inspect(self) -> str:
        facts = self._store.all_facts()
        episodes = self._store.load_recent_episodes(10)
        lines = [
            f"=== Memory State ===",
            f"Session: {self._session.session_id}  "
            f"(started {self._session.started_at[:16]})",
            f"Facts: {self._store.fact_count()}  |  "
            f"Episodes: {self._store.episode_count()}  |  "
            f"Semantic entries: {self._semantic.entry_count}",
        ]
        if facts:
            lines.append("\n-- Facts --")
            current_entity = ""
            for f in facts:
                suppressed = "  [suppressed]" if f["confidence"] < self.injection_threshold else ""
                if f["entity"] != current_entity:
                    current_entity = f["entity"]
                    lines.append(f"  {f['entity']}:")
                lines.append(
                    f"    {f['attribute']}: {f['value']!r}  "
                    f"conf={f['confidence']:.2f}  ×{f['assertion_count']}{suppressed}"
                )
        if episodes:
            lines.append("\n-- Episodes --")
            for ep in episodes:
                lines.append(f"  [{ep['started_at'][:10]}] {ep['summary']}")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "session_id":     self._session.session_id,
            "turn":           self._session.turn_index,
            "buffer":         len(self._buffer),
            "facts":          self._store.fact_count(),
            "episodes":       self._store.episode_count(),
            "semantic_idx":   self._semantic.entry_count,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _run_ttl_cleanup(self) -> dict:
        pruned_facts    = self._store.prune_stale(min_confidence=0.15, max_age_days=90)
        pruned_episodes = self._store.prune_old_episodes(max_age_days=365)
        # Sync semantic index: remove entries for deleted episodes
        valid_ids: set[str] = set()
        with self._store._connect() as conn:
            rows = conn.execute("SELECT session_id FROM episodes").fetchall()
            valid_ids = {r[0] for r in rows}
        pruned_semantic = self._semantic.prune_old_sessions(valid_ids)
        return {
            "pruned_facts": pruned_facts,
            "pruned_episodes": pruned_episodes,
            "pruned_semantic": pruned_semantic,
        }

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        nl = cut.rfind("\n")
        if nl > max_chars // 2:
            cut = cut[:nl]
        return cut + "\n  … (truncated)"
