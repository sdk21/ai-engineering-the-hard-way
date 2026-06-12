"""
Sliding Window Memory
---------------------
Keeps only the last `window_size` message pairs in context.
Optionally accumulates a rolling summary of dropped messages.
"""


class SlidingWindowBuffer:
    def __init__(self, window_size: int = 4, system_prompt: str = "You are a helpful assistant."):
        """
        Args:
            window_size: Max number of *messages* (not pairs) to keep.
                         A value of 4 means 2 user + 2 assistant turns.
        """
        self.window_size = window_size
        self.system_prompt = system_prompt
        self._history: list[dict[str, str]] = []
        self._dropped_summary: list[str] = []  # plain-text notes about dropped turns

    def add_user_message(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self._history.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        """Drop oldest messages when window is exceeded."""
        while len(self._history) > self.window_size:
            dropped = self._history.pop(0)
            self._dropped_summary.append(f"[{dropped['role']}]: {dropped['content'][:80]}…")

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._history)

    def get_effective_system(self) -> str:
        """System prompt optionally augmented with a summary of dropped turns."""
        if not self._dropped_summary:
            return self.system_prompt
        summary = "\n".join(self._dropped_summary[-10:])  # keep last 10 dropped notes
        return (
            f"{self.system_prompt}\n\n"
            f"[Earlier conversation (summarised):\n{summary}]"
        )

    @property
    def dropped_count(self) -> int:
        return len(self._dropped_summary)

    def __len__(self) -> int:
        return len(self._history)
