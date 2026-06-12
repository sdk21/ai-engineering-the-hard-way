"""
Summarization Memory
--------------------
When the conversation buffer grows beyond a threshold, the oldest messages
are compressed into a running plain-text summary rather than dropped.

The summary is injected back into the system prompt so the model always has
access to a compressed view of the full history, not just the recent window.

Architecture:
    [system prompt]
    [running summary]   ← compressed history (grows slowly)
    [recent buffer]     ← last N messages (fixed window)

The summarizer is injected as a callable so this class works with both
a mock summarizer (for tests / demos) and a real LLM summarizer.
"""

from typing import Callable


SummarizerFn = Callable[[str, list[dict[str, str]]], str]


class SummarizationMemory:
    """
    Maintains a rolling summary of older turns plus a fixed window of recent turns.

    Args:
        summarizer:       Callable(existing_summary, messages_to_compress) -> new_summary
        system_prompt:    Base system prompt (summary is appended to this)
        buffer_size:      Max messages to keep in the recent window before summarizing
        summarize_after:  How many messages over buffer_size trigger a summarization pass
    """

    def __init__(
        self,
        summarizer: SummarizerFn,
        system_prompt: str = "You are a helpful assistant.",
        buffer_size: int = 6,
        summarize_after: int = 2,
    ):
        self.summarizer = summarizer
        self.system_prompt = system_prompt
        self.buffer_size = buffer_size
        self.summarize_after = summarize_after

        self._buffer: list[dict[str, str]] = []
        self._summary: str = ""
        self.summarization_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self._buffer.append({"role": "user", "content": content})
        self._maybe_summarize()

    def add_assistant_message(self, content: str) -> None:
        self._buffer.append({"role": "assistant", "content": content})
        self._maybe_summarize()

    def get_messages(self) -> list[dict[str, str]]:
        """Return the recent buffer, ready to send to the API."""
        return list(self._buffer)

    def get_system_prompt(self) -> str:
        """Return the system prompt, augmented with the running summary if one exists."""
        if not self._summary:
            return self.system_prompt
        return (
            f"{self.system_prompt}\n\n"
            f"## Summary of earlier conversation\n{self._summary}"
        )

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def buffer_length(self) -> int:
        return len(self._buffer)

    @property
    def token_estimate(self) -> int:
        summary_tokens = len(self._summary) // 4
        buffer_tokens = sum(len(m["content"]) for m in self._buffer) // 4
        return summary_tokens + buffer_tokens

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_summarize(self) -> None:
        """Compress oldest messages into the summary when the buffer overflows."""
        if len(self._buffer) <= self.buffer_size + self.summarize_after:
            return

        # Take the oldest messages (everything except the recent buffer_size)
        n_to_compress = len(self._buffer) - self.buffer_size
        to_compress = self._buffer[:n_to_compress]
        self._buffer = self._buffer[n_to_compress:]

        # Merge them into the running summary
        self._summary = self.summarizer(self._summary, to_compress)
        self.summarization_count += 1
