"""
Entity Memory
-------------
Rather than storing raw conversation text or vectors, entity memory
explicitly extracts structured facts — entities and their attributes —
from each turn and maintains them in a typed store.

The key insight: a fact like "Alice is allergic to peanuts" is better
stored as:

    entities["Alice"]["allergy"] = "peanuts"

than as a raw sentence buried in a summary or a vector. It is always
available, never gets lossy-compressed, and can be queried precisely.

Architecture:

    Each turn
        ↓
    EntityExtractor (mock or LLM-based)
        ↓
    {entity: {attribute: value}, ...}
        ↓
    EntityStore.merge()       ← upserts, newer values overwrite older
        ↓
    Injected into system prompt as a structured "Known facts" block

Components:
    EntityStore      — dict-of-dicts store with merge/upsert semantics
    EntityExtractor  — callable(role, content) → dict[str, dict[str, str]]
    EntityMemory     — orchestrates extraction, storage, and injection
"""

from typing import Callable


# Type alias: {entity_name: {attribute: value}}
EntityDict = dict[str, dict[str, str]]
ExtractorFn = Callable[[str, str], EntityDict]   # (role, content) → entities


# ---------------------------------------------------------------------------
# Entity store
# ---------------------------------------------------------------------------

class EntityStore:
    """
    Stores entities as a nested dict: {entity → {attribute → value}}.

    Merge semantics:
      - New entity: added in full
      - Existing entity, new attribute: added
      - Existing entity, existing attribute: newer value overwrites older
        (last-write-wins — assumes conversation is chronological)
    """

    def __init__(self) -> None:
        self._store: EntityDict = {}

    def merge(self, entities: EntityDict) -> None:
        """Upsert extracted entities into the store."""
        for entity, attributes in entities.items():
            if entity not in self._store:
                self._store[entity] = {}
            self._store[entity].update(attributes)

    def get(self, entity: str) -> dict[str, str]:
        return dict(self._store.get(entity, {}))

    def all(self) -> EntityDict:
        return {e: dict(attrs) for e, attrs in self._store.items()}

    def format_for_prompt(self) -> str:
        """
        Render the store as a structured block suitable for injection
        into a system prompt.

        Example output:
            Alice:
              - role: software engineer
              - location: Tokyo
              - allergy: peanuts
            project:
              - type: distributed caching system
        """
        if not self._store:
            return ""
        lines = []
        for entity, attrs in sorted(self._store.items()):
            lines.append(f"{entity}:")
            for attr, value in sorted(attrs.items()):
                lines.append(f"  - {attr}: {value}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, entity: str) -> bool:
        return entity in self._store


# ---------------------------------------------------------------------------
# Entity memory
# ---------------------------------------------------------------------------

class EntityMemory:
    """
    Maintains a structured store of entities extracted from the conversation.

    On each turn:
      1. Run the extractor on the new message → {entity: {attr: val}}
      2. Merge extracted entities into the store
      3. Build context: system prompt augmented with the entity store

    Args:
        extractor:      Callable(role, content) → EntityDict
        system_prompt:  Base system instructions
        buffer_size:    Recent message buffer (always included verbatim)
    """

    def __init__(
        self,
        extractor: ExtractorFn,
        system_prompt: str = "You are a helpful assistant.",
        buffer_size: int = 6,
    ) -> None:
        self.extractor = extractor
        self.system_prompt = system_prompt
        self.buffer_size = buffer_size

        self._store = EntityStore()
        self._buffer: list[dict[str, str]] = []
        self.extractions: int = 0   # total extraction calls made

    def add_user_message(self, content: str) -> None:
        self._ingest("user", content)

    def add_assistant_message(self, content: str) -> None:
        self._ingest("assistant", content)

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._buffer)

    def get_system_prompt(self) -> str:
        """System prompt augmented with all known entities."""
        facts = self._store.format_for_prompt()
        if not facts:
            return self.system_prompt
        return (
            f"{self.system_prompt}\n\n"
            f"## Known facts about the user and context\n"
            f"{facts}"
        )

    @property
    def store(self) -> EntityStore:
        return self._store

    @property
    def buffer_length(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ingest(self, role: str, content: str) -> None:
        # Extract entities from this message and merge into the store
        entities = self.extractor(role, content)
        if entities:
            self._store.merge(entities)
            self.extractions += 1

        # Maintain the recent buffer
        self._buffer.append({"role": role, "content": content})
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)
