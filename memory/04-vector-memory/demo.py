"""
Demo: Vector Memory
Usage:
    uv run python demo.py --mock
    uv run python demo.py --real

Suggested conversation to see semantic retrieval in action:
    1. "My name is Alice and I'm a Python engineer."
    2. "I'm working on a distributed caching system."
    3. "My dog's name is Biscuit."
    4. "I prefer tabs over spaces."
    5. "Tell me a joke." (filler)
    6. "Tell me another." (filler)
    7. "Tell me one more." (filler)    ← turn 1 falls out of the buffer
    8. "What are best practices for Redis?"
       → should retrieve turn 2 (caching) despite it being out of the buffer
    9. "What's my name?"
       → should retrieve turn 1 (name) despite it being out of the buffer
"""

import argparse
import os
import sys

import numpy as np

from experiment import Embedder, VectorMemory


# ---------------------------------------------------------------------------
# Mock embedder — deterministic, no model download
# Encodes text as a sparse bag-of-words-style vector over a fixed vocabulary.
# Similar words produce similar vectors; completely unrelated text does not.
# ---------------------------------------------------------------------------

VOCAB = [
    "name", "alice", "python", "engineer", "software", "code", "programming",
    "dog", "biscuit", "pet", "animal",
    "cache", "redis", "distributed", "system", "backend", "database", "storage",
    "tabs", "spaces", "indent", "style",
    "joke", "funny", "laugh", "tell", "another",
    "best", "practices", "recommend", "use",
    "weather", "temperature", "forecast",
    "hello", "hi", "how", "are", "you",
]


class MockEmbedder:
    """
    Produces a normalised bag-of-words vector over VOCAB.
    Good enough to demonstrate retrieval mechanics without a real model.
    """
    dim = len(VOCAB)

    def encode(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            for j, vocab_word in enumerate(VOCAB):
                if vocab_word in words:
                    result[i, j] = 1.0
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm
        return result


# ---------------------------------------------------------------------------
# Real embedder — sentence-transformers all-MiniLM-L6-v2
# ---------------------------------------------------------------------------

class RealEmbedder:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model ready.\n")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Chat backends
# ---------------------------------------------------------------------------

def mock_chat(messages: list[dict], system: str) -> str:
    # Collect all text visible to the model (system + messages)
    full_context = system + " " + " ".join(m["content"] for m in messages)
    ctx = full_context.lower()
    last = messages[-1]["content"].lower() if messages else ""

    if "name" in last:
        for phrase in ["name is ", "i'm ", "i am "]:
            if phrase in ctx:
                idx = ctx.index(phrase) + len(phrase)
                name = ctx[idx:].split()[0].rstrip(".,!]")
                if len(name) > 1:
                    return f"Your name is {name.capitalize()}."
        return "I don't see your name in my current context."

    if any(w in last for w in ["redis", "cache", "caching", "practices"]):
        if "cache" in ctx or "redis" in ctx or "distributed" in ctx:
            return "Based on your work on a distributed caching system, key Redis best practices include: use TTLs on all keys, prefer connection pooling, and avoid storing large objects."
        return "Use connection pooling, set appropriate TTLs, and monitor memory usage."

    return f"[Mock] Context window has {len(messages)} message(s). I can see relevant memories injected above the buffer."


def real_chat(messages: list[dict], system: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=messages,
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Demo loop
# ---------------------------------------------------------------------------

def run_demo(use_mock: bool, buffer_size: int, top_k: int, api_key: str | None) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Vector Memory Demo [{mode}] | buffer={buffer_size}, top_k={top_k} ===")
    print("Past turns are embedded and retrieved by semantic similarity.")
    print("Commands: 'store' to inspect the vector store, 'stats' for memory stats, 'quit' to exit.\n")

    embedder: Embedder = MockEmbedder() if use_mock else RealEmbedder()
    memory = VectorMemory(
        embedder=embedder,
        buffer_size=buffer_size,
        top_k=top_k,
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() == "quit":
            break

        if user_input.lower() == "store":
            print(f"\n--- Vector Store ({memory.store_size} entries) ---")
            for entry in memory._store._entries:
                print(f"  [turn {entry.turn_index}] {entry.role.upper()}: {entry.content[:80]}")
            print()
            continue

        if user_input.lower() == "stats":
            print(
                f"[store={memory.store_size} turns | "
                f"buffer={memory.buffer_length}/{buffer_size} msgs | "
                f"top_k={top_k}]\n"
            )
            continue

        memory.add_user_message(user_input)
        messages = memory.get_messages(query=user_input)
        system = memory.get_system_prompt()

        if use_mock:
            reply = mock_chat(messages, system)
        else:
            reply = real_chat(messages, system, api_key)

        memory.add_assistant_message(reply)
        print(f"Assistant: {reply}")
        print(
            f"  [store={memory.store_size} | "
            f"buffer={memory.buffer_length}/{buffer_size}]\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    parser.add_argument("--buffer", type=int, default=6, help="Recent message buffer size")
    parser.add_argument("--top-k", type=int, default=3, help="Memories to retrieve per turn")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.real and not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock, buffer_size=args.buffer, top_k=args.top_k, api_key=api_key)
