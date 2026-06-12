"""
Vector Memory
-------------
Every conversation turn is embedded and stored in an in-memory vector store.
At each new turn, the most semantically relevant past turns are retrieved
and injected into the model's context alongside the recent buffer.

This decouples *what the model remembers* from *how recent it is*.
A fact mentioned 200 turns ago is retrievable if it is semantically
similar to the current query — something neither the buffer nor the
sliding window can do.

Architecture:

    [system prompt]
    [retrieved memories]  ← top-k turns most similar to current query
    [recent buffer]       ← last N verbatim turns
    [current user turn]

Components:
    VectorStore     — stores embeddings + original text, returns top-k by cosine sim
    VectorMemory    — orchestrates the store, buffer, and retrieval
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Embedder protocol — allows swapping mock vs real embedder
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a 2-D float32 array of shape (len(texts), dim)."""
        ...


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    role: str
    content: str
    turn_index: int
    embedding: np.ndarray = field(repr=False)


class VectorStore:
    """
    Simple in-memory vector store backed by numpy.
    Stores (role, content, embedding) tuples and retrieves by cosine similarity.
    """

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def add(self, role: str, content: str, turn_index: int, embedding: np.ndarray) -> None:
        self._entries.append(MemoryEntry(role, content, turn_index, embedding))

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[MemoryEntry]:
        """Return the top_k entries most similar to query_embedding."""
        if not self._entries:
            return []

        matrix = np.stack([e.embedding for e in self._entries])  # (N, dim)
        # Cosine similarity: normalise both sides
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        scores = m_norm @ q_norm  # (N,)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._entries[i] for i in top_indices]

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Vector memory
# ---------------------------------------------------------------------------

class VectorMemory:
    """
    Combines a fixed recent buffer with semantic retrieval from a vector store.

    On each turn:
      1. Embed the new user message
      2. Retrieve top-k semantically relevant past turns from the store
      3. Add the new turn to both the buffer and the store
      4. Build a context: [retrieved memories] + [recent buffer]

    Args:
        embedder:           Embedder instance (SentenceTransformer or mock)
        system_prompt:      Base system instructions
        buffer_size:        Number of most-recent turns to always include verbatim
        top_k:              Number of semantically similar turns to retrieve
        min_similarity:     Minimum cosine similarity to include a retrieved memory
                            (prevents injecting unrelated memories)
    """

    def __init__(
        self,
        embedder: Embedder,
        system_prompt: str = "You are a helpful assistant.",
        buffer_size: int = 6,
        top_k: int = 3,
        min_similarity: float = 0.3,
    ) -> None:
        self.embedder = embedder
        self.system_prompt = system_prompt
        self.buffer_size = buffer_size
        self.top_k = top_k
        self.min_similarity = min_similarity

        self._store = VectorStore()
        self._buffer: list[dict[str, str]] = []
        self._turn_index: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self._add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self._add_message("assistant", content)

    def get_messages(self, query: str) -> list[dict[str, str]]:
        """
        Build the message list for the next API call.

        Returns retrieved relevant memories (as a synthetic user message)
        followed by the recent verbatim buffer.
        """
        retrieved = self._retrieve(query)
        messages: list[dict[str, str]] = []

        if retrieved:
            memory_text = self._format_memories(retrieved)
            # Inject as a system-style note before the recent buffer
            messages.append({
                "role": "user",
                "content": f"[Relevant memory from earlier in our conversation:\n{memory_text}]",
            })
            messages.append({
                "role": "assistant",
                "content": "I'll keep that context in mind.",
            })

        messages.extend(self._buffer)
        return messages

    def get_system_prompt(self) -> str:
        return self.system_prompt

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def store_size(self) -> int:
        return len(self._store)

    @property
    def buffer_length(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_message(self, role: str, content: str) -> None:
        embedding = self.embedder.encode([content])[0]
        self._store.add(role, content, self._turn_index, embedding)
        self._buffer.append({"role": role, "content": content})

        # Keep buffer bounded
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)

        self._turn_index += 1

    def _retrieve(self, query: str) -> list[MemoryEntry]:
        """Find past turns semantically relevant to the query."""
        if len(self._store) == 0:
            return []

        query_embedding = self.embedder.encode([query])[0]

        # Compute similarity for all entries to filter by min_similarity
        matrix = np.stack([e.embedding for e in self._store._entries])
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        scores = m_norm @ q_norm

        candidates = [
            (self._store._entries[i], float(scores[i]))
            for i in range(len(self._store._entries))
            if scores[i] >= self.min_similarity
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Exclude turns already in the recent buffer (no need to show them twice)
        buffer_contents = {m["content"] for m in self._buffer}
        filtered = [
            entry for entry, _ in candidates
            if entry.content not in buffer_contents
        ]

        return filtered[: self.top_k]

    def _format_memories(self, entries: list[MemoryEntry]) -> str:
        # Sort by original turn order so they read chronologically
        sorted_entries = sorted(entries, key=lambda e: e.turn_index)
        return "\n".join(
            f"  [{e.role.upper()}, turn {e.turn_index}]: {e.content}"
            for e in sorted_entries
        )
