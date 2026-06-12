"""
Conversation Buffer Memory
--------------------------
Maintains the full message history and replays it on every LLM call.
This is the simplest possible memory pattern — the baseline everything else builds on.
"""

from typing import Any


class ConversationBuffer:
    """Stores every turn and replays the full history to the model."""

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.history: list[dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        """Return the full history ready to send to the API."""
        return list(self.history)

    def token_estimate(self) -> int:
        """Rough token count (4 chars ≈ 1 token) — illustrates growth over time."""
        total_chars = sum(len(m["content"]) for m in self.history)
        return total_chars // 4

    def clear(self) -> None:
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)
