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

Two vector store implementations are provided to show the contrast
between naive linear search and ANN (Approximate Nearest Neighbour):

    NumpyVectorStore    — exact cosine similarity, O(N) per search
                          simple, no extra deps, fine for short conversations
    FaissVectorStore    — HNSW index, O(log N) per search
                          production-grade, scales to millions of vectors

Both implement the same VectorStoreBase interface so VectorMemory works
with either one.
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
# Shared data model
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    role: str
    content: str
    turn_index: int
    embedding: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# VectorStore implementations
# ---------------------------------------------------------------------------

class NumpyVectorStore:
    """
    Exact cosine similarity search backed by a numpy matrix.

    Complexity: O(N) per search — computes similarity against every stored vector.

    When to use:
      - Conversations up to ~5,000 turns
      - No extra dependencies desired
      - Exact results required (no approximation)

    When NOT to use:
      - Long-running agents accumulating tens of thousands of turns
      - Shared memory stores across many users (millions of vectors)
      → Switch to FaissVectorStore in those cases
    """

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def add(self, role: str, content: str, turn_index: int, embedding: np.ndarray) -> None:
        self._entries.append(MemoryEntry(role, content, turn_index, embedding))

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[MemoryEntry]:
        """
        Exact nearest neighbour search.
        Normalises both sides and computes dot products — equivalent to cosine similarity.
        Time complexity: O(N * d) where N = stored vectors, d = embedding dimension.
        """
        if not self._entries:
            return []

        matrix = np.stack([e.embedding for e in self._entries])      # (N, d)
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        scores = m_norm @ q_norm                                      # (N,)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._entries[i] for i in top_indices]

    def __len__(self) -> int:
        return len(self._entries)


class FaissVectorStore:
    """
    Approximate nearest neighbour search using FAISS HNSW index.

    HNSW (Hierarchical Navigable Small World) builds a multi-layer proximity
    graph. Search navigates the graph greedily, visiting only a small fraction
    of stored vectors before converging on the approximate nearest neighbours.

    Complexity: O(log N) per search — largely independent of corpus size.

    How HNSW works (briefly):
      - Layer 0: dense graph — every node connects to its M nearest neighbours
      - Layer 1: sparser graph — a random subset of nodes, each connected to M neighbours
      - Layer 2+: progressively sparser
      Search starts at the top layer (few nodes, fast traversal), finds approximate
      entry point, descends to layer 0 for fine-grained search.

    Parameters:
      dim:         Embedding dimension (must match the embedder)
      M:           Connections per node per layer. Higher = better recall, more memory.
                   Typical values: 16–64. Default 32.
      ef_search:   Search-time beam width. Higher = better recall, slower search.
                   Typical values: 32–256. Default 64.

    When to use:
      - Long-running agents or shared memory stores (10k+ vectors)
      - Latency-sensitive applications
      - Acceptable to trade a tiny recall loss (~1–2%) for orders-of-magnitude speedup

    Requires: faiss-cpu (pip install faiss-cpu)
    """

    def __init__(self, dim: int, M: int = 32, ef_search: int = 64) -> None:
        import faiss

        self.dim = dim
        self._entries: list[MemoryEntry] = []

        # IndexHNSWFlat: HNSW graph over flat (exact) distance computations at each node.
        # We use inner product (IP) as the distance metric; pre-normalising vectors
        # makes inner product equivalent to cosine similarity.
        self._index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efSearch = ef_search

    def add(self, role: str, content: str, turn_index: int, embedding: np.ndarray) -> None:
        # FAISS requires float32 and a 2-D array
        vec = self._normalise(embedding).reshape(1, -1).astype(np.float32)
        self._index.add(vec)
        self._entries.append(MemoryEntry(role, content, turn_index, embedding))

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[MemoryEntry]:
        """
        Approximate nearest neighbour search via HNSW graph traversal.
        Time complexity: O(log N) — sub-linear, scales to millions of vectors.
        Returns approximate top-k; may occasionally miss the true nearest neighbour.
        """
        if not self._entries:
            return []

        actual_k = min(top_k, len(self._entries))
        vec = self._normalise(query_embedding).reshape(1, -1).astype(np.float32)
        _scores, indices = self._index.search(vec, actual_k)   # both shape (1, k)
        return [self._entries[i] for i in indices[0] if i >= 0]

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)


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
        store:              NumpyVectorStore (default) or FaissVectorStore
        system_prompt:      Base system instructions
        buffer_size:        Number of most-recent turns to always include verbatim
        top_k:              Number of semantically similar turns to retrieve
        min_similarity:     Minimum cosine similarity to include a retrieved memory
                            (prevents injecting unrelated memories)
    """

    def __init__(
        self,
        embedder: Embedder,
        store: NumpyVectorStore | FaissVectorStore | None = None,
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

        self._store: NumpyVectorStore | FaissVectorStore = store if store is not None else NumpyVectorStore()
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
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        # Retrieve candidates from the store (exact or ANN depending on store type)
        candidates = self._store.search(q_norm, top_k=self.top_k + len(self._buffer))

        # Filter by minimum similarity threshold
        buffer_contents = {m["content"] for m in self._buffer}
        results = []
        for entry in candidates:
            # Recompute exact cosine similarity for threshold filtering
            e_norm = entry.embedding / (np.linalg.norm(entry.embedding) + 1e-9)
            sim = float(np.dot(q_norm, e_norm))
            if sim < self.min_similarity:
                continue
            # Skip turns already visible in the recent buffer
            if entry.content in buffer_contents:
                continue
            results.append(entry)
            if len(results) == self.top_k:
                break

        return results

    def _format_memories(self, entries: list[MemoryEntry]) -> str:
        # Sort by original turn order so they read chronologically
        sorted_entries = sorted(entries, key=lambda e: e.turn_index)
        return "\n".join(
            f"  [{e.role.upper()}, turn {e.turn_index}]: {e.content}"
            for e in sorted_entries
        )
