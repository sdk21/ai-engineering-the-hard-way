"""
Episodic Memory
---------------
All prior memory experiments store data in RAM — lost the moment the
process exits. Episodic memory persists conversation sessions to durable
storage so the model can recall what happened in *previous* conversations,
not just the current one.

In human cognition, episodic memory is the record of specific events
experienced at a particular time: "last Tuesday I talked to Alice about
the Redis migration." AI episodic memory mirrors this: each conversation
session is an episode stored with a timestamp and a summary.

Architecture:

    Session start
        ↓
    Load recent past episodes from store
        ↓
    Inject episode summaries into system prompt
        ↓
    Conversation runs (current buffer in RAM)
        ↓
    Session end → summarise session → persist to store

Two storage backends are provided to show the contrast:

    JSONEpisodeStore    — one JSON file per episode, human-readable,
                          good for debugging and small deployments

    SQLiteEpisodeStore  — single SQLite database, proper indexing and
                          querying, the natural choice for production
                          (stdlib only — no extra dependencies)

Both implement EpisodeStore so EpisodicMemory works with either.
"""

import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: _now())


@dataclass
class Episode:
    session_id: str
    started_at: str
    ended_at: str
    summary: str
    turns: list[Turn] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """One-line representation for injection into the system prompt."""
        date = self.started_at[:10]   # YYYY-MM-DD
        return f"[{date}] {self.summary}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


SummarizerFn = Callable[[list[Turn]], str]


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------

class EpisodeStore(ABC):
    @abstractmethod
    def save(self, episode: Episode) -> None: ...

    @abstractmethod
    def load_recent(self, n: int = 5) -> list[Episode]: ...

    @abstractmethod
    def count(self) -> int: ...


class JSONEpisodeStore(EpisodeStore):
    """
    One JSON file per episode, stored in a directory.

    Pros:  Human-readable, trivial to inspect with any text editor,
           no dependencies beyond stdlib.
    Cons:  No indexing — loading N recent episodes requires scanning
           all file mtimes. Slow for large episode counts.

    Best for: Development, debugging, small deployments (< ~1000 sessions).
    """

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, episode: Episode) -> None:
        path = self.dir / f"{episode.session_id}.json"
        data = {
            "session_id": episode.session_id,
            "started_at": episode.started_at,
            "ended_at": episode.ended_at,
            "summary": episode.summary,
            "turns": [
                {"role": t.role, "content": t.content, "timestamp": t.timestamp}
                for t in episode.turns
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    def load_recent(self, n: int = 5) -> list[Episode]:
        files = sorted(self.dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        episodes = []
        for f in files[:n]:
            data = json.loads(f.read_text())
            episodes.append(Episode(
                session_id=data["session_id"],
                started_at=data["started_at"],
                ended_at=data["ended_at"],
                summary=data["summary"],
                turns=[Turn(**t) for t in data.get("turns", [])],
            ))
        return episodes

    def count(self) -> int:
        return len(list(self.dir.glob("*.json")))


class SQLiteEpisodeStore(EpisodeStore):
    """
    All episodes in a single SQLite database.

    Schema:
        episodes(session_id, started_at, ended_at, summary)
        turns(id, session_id, role, content, timestamp)

    Pros:  Proper indexing on started_at — loading recent episodes is
           O(log N) regardless of total episode count. Supports arbitrary
           SQL queries (e.g. "find all sessions where I mentioned Redis").
           Single file, easy to back up or ship.
    Cons:  Slightly more setup than JSON; requires understanding of SQL
           for advanced queries.

    Best for: Production deployments, long-running agents, shared stores.
    This is the right choice for any real application.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at   TEXT NOT NULL,
                    summary    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_started
                    ON episodes(started_at DESC);

                CREATE TABLE IF NOT EXISTS turns (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES episodes(session_id),
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    timestamp  TEXT NOT NULL
                );
            """)

    def save(self, episode: Episode) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?)",
                (episode.session_id, episode.started_at, episode.ended_at, episode.summary),
            )
            conn.executemany(
                "INSERT INTO turns(session_id, role, content, timestamp) VALUES (?,?,?,?)",
                [(episode.session_id, t.role, t.content, t.timestamp) for t in episode.turns],
            )

    def load_recent(self, n: int = 5) -> list[Episode]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, started_at, ended_at, summary "
                "FROM episodes ORDER BY started_at DESC LIMIT ?",
                (n,),
            ).fetchall()

            episodes = []
            for session_id, started_at, ended_at, summary in rows:
                turn_rows = conn.execute(
                    "SELECT role, content, timestamp FROM turns "
                    "WHERE session_id=? ORDER BY id",
                    (session_id,),
                ).fetchall()
                episodes.append(Episode(
                    session_id=session_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    summary=summary,
                    turns=[Turn(role=r, content=c, timestamp=ts) for r, c, ts in turn_rows],
                ))
        return episodes

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Manages the current session and past episode recall.

    Lifecycle:
        memory = EpisodicMemory(store, summarizer)
        memory.start_session()          # loads past episodes into context
        ... conversation turns ...
        memory.end_session()            # summarises + persists to store

    Args:
        store:          EpisodeStore (JSON or SQLite)
        summarizer:     Callable(turns) → str summary of the session
        system_prompt:  Base system instructions
        buffer_size:    Recent message buffer size
        recall_n:       How many past episodes to inject into context
    """

    def __init__(
        self,
        store: EpisodeStore,
        summarizer: SummarizerFn,
        system_prompt: str = "You are a helpful assistant.",
        buffer_size: int = 8,
        recall_n: int = 3,
    ) -> None:
        self.store = store
        self.summarizer = summarizer
        self.system_prompt = system_prompt
        self.buffer_size = buffer_size
        self.recall_n = recall_n

        self._session_id: str = ""
        self._started_at: str = ""
        self._turns: list[Turn] = []          # all turns this session (for persistence)
        self._buffer: list[dict[str, str]] = []   # recent turns for LLM context
        self._past_episodes: list[Episode] = []

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Begin a new session — load past episodes, mint a session ID."""
        self._session_id = str(uuid.uuid4())[:8]
        self._started_at = _now()
        self._past_episodes = self.store.load_recent(self.recall_n)
        self._turns = []
        self._buffer = []

    def end_session(self) -> Episode | None:
        """Summarise the current session and persist it to the store."""
        if not self._turns:
            return None

        summary = self.summarizer(self._turns)
        episode = Episode(
            session_id=self._session_id,
            started_at=self._started_at,
            ended_at=_now(),
            summary=summary,
            turns=self._turns,
        )
        self.store.save(episode)
        return episode

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self._add(Turn("user", content))

    def add_assistant_message(self, content: str) -> None:
        self._add(Turn("assistant", content))

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._buffer)

    def get_system_prompt(self) -> str:
        """System prompt augmented with summaries of past episodes."""
        if not self._past_episodes:
            return self.system_prompt

        lines = ["## Previous conversations"]
        for ep in reversed(self._past_episodes):   # chronological order
            lines.append(f"  {ep.format_for_prompt()}")

        return f"{self.system_prompt}\n\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def past_episode_count(self) -> int:
        return len(self._past_episodes)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add(self, turn: Turn) -> None:
        self._turns.append(turn)
        self._buffer.append({"role": turn.role, "content": turn.content})
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)
