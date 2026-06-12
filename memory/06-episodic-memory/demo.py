"""
Demo: Episodic Memory
Usage:
    uv run python demo.py --mock                          # JSON store
    uv run python demo.py --mock --store sqlite           # SQLite store
    uv run python demo.py --real                          # SQLite + Claude
    uv run python demo.py --mock --clear                  # wipe the store and start fresh

The key behaviour to observe:
  - Run the demo, introduce yourself, chat, then quit.
  - Run the demo again — the model remembers the previous session.
  - Run a third time — it remembers both prior sessions.

Each run is a new session (new session_id). Past sessions are loaded
from disk and injected into the system prompt as episode summaries.

Store location (default): /tmp/ai-hardway-episodic/
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from experiment import (
    EpisodicMemory,
    JSONEpisodeStore,
    SQLiteEpisodeStore,
    SummarizerFn,
    Turn,
)

DEFAULT_STORE_DIR = Path("/tmp/ai-hardway-episodic")


# ---------------------------------------------------------------------------
# Mock summarizer
# ---------------------------------------------------------------------------

def mock_summarizer(turns: list[Turn]) -> str:
    """
    Extract key facts from the session turns without an LLM call.
    Produces a compact summary suitable for injection in future sessions.
    """
    user_messages = [t.content for t in turns if t.role == "user"]
    if not user_messages:
        return "Empty session."

    # Pull out first and last user messages as anchors
    first = user_messages[0][:80].rstrip(".,")
    facts = [f"User said: '{first}'"]

    # Grab any name mentions
    import re
    for msg in user_messages:
        m = re.search(r"\bmy name is\s+([A-Z][a-z]+)", msg, re.IGNORECASE)
        if m:
            facts.append(f"User's name: {m.group(1)}")
            break

    facts.append(f"Session had {len(user_messages)} user turn(s).")
    return " | ".join(facts)


# ---------------------------------------------------------------------------
# Real summarizer — Claude condenses the session
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT = """\
Summarise the following conversation session in 1-2 sentences.
Focus on: who the user is, what they discussed, any decisions or facts shared.
Be concise — this summary will be shown to the model in a future session.

{transcript}

Return only the summary, nothing else."""


def make_real_summarizer(api_key: str) -> SummarizerFn:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def summarizer(turns: list[Turn]) -> str:
        transcript = "\n".join(f"{t.role.upper()}: {t.content}" for t in turns)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": SUMMARIZE_PROMPT.format(transcript=transcript)}],
        )
        return response.content[0].text.strip()

    return summarizer


# ---------------------------------------------------------------------------
# Chat backends
# ---------------------------------------------------------------------------

def mock_chat(messages: list[dict], system: str) -> str:
    last = messages[-1]["content"].lower() if messages else ""
    ctx = system.lower()

    if any(w in last for w in ["remember", "last time", "previous", "before", "told you"]):
        if "previous conversations" in ctx:
            # Extract first episode line to quote back
            import re
            m = re.search(r"\[(\d{4}-\d{2}-\d{2})\]\s+(.+)", ctx)
            if m:
                return (
                    f"Yes! In our session on {m.group(1)} I noted: \"{m.group(2).strip()}\". "
                    f"I have {ctx.count('[20')} past session(s) in memory."
                )
        return "This appears to be our first conversation — I have no prior sessions stored."

    if "name" in last:
        import re
        m = re.search(r"user'?s? name[:\s]+([a-z]+)", ctx)
        if m:
            return f"Your name is {m.group(1).capitalize()} — I remember from a previous session."
        m2 = re.search(r"my name is\s+([a-z]+)", " ".join(m["content"] for m in messages), re.IGNORECASE)
        if m2:
            return f"Your name is {m2.group(1).capitalize()}."
        return "I don't have your name recorded yet."

    past_count = ctx.count("[20")   # rough count of episode lines
    return (
        f"[Mock] Session active. I have {past_count} past session summary/summaries "
        f"in my context and {len(messages)} message(s) in the current buffer."
    )


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

def run_demo(
    use_mock: bool,
    use_sqlite: bool,
    store_dir: Path,
    api_key: str | None,
) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    backend = "SQLite" if use_sqlite else "JSON"
    print(f"\n=== Episodic Memory Demo [{mode}] | store={backend} @ {store_dir} ===")
    print("Memory persists across runs. Quit and re-run to experience cross-session recall.")
    print("Commands: 'history' to list past episodes, 'stats', 'quit' to end session.\n")

    # Build store
    if use_sqlite:
        store = SQLiteEpisodeStore(store_dir / "episodes.db")
    else:
        store = JSONEpisodeStore(store_dir / "episodes")

    summarizer = mock_summarizer if use_mock else make_real_summarizer(api_key)
    memory = EpisodicMemory(store=store, summarizer=summarizer)
    memory.start_session()

    past = memory.past_episode_count
    if past:
        print(f"Loaded {past} past episode(s) from store.")
        for ep in memory._past_episodes:
            print(f"  {ep.format_for_prompt()}")
        print()
    else:
        print("No past episodes found — this is the first session.\n")

    print(f"Session ID: {memory.session_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() == "quit":
            break

        if user_input.lower() == "history":
            episodes = store.load_recent(10)
            if not episodes:
                print("\n(no episodes stored yet)\n")
            else:
                print(f"\n--- Episode History ({store.count()} total) ---")
                for ep in reversed(episodes):
                    print(f"  {ep.format_for_prompt()}  [{len(ep.turns)} turns]")
                print()
            continue

        if user_input.lower() == "stats":
            print(
                f"[session={memory.session_id} | "
                f"turns={memory.turn_count} | "
                f"past_episodes={memory.past_episode_count} | "
                f"store_total={store.count()}]\n"
            )
            continue

        memory.add_user_message(user_input)
        system = memory.get_system_prompt()

        if use_mock:
            reply = mock_chat(memory.get_messages(), system)
        else:
            reply = real_chat(memory.get_messages(), system, api_key)

        memory.add_assistant_message(reply)
        print(f"Assistant: {reply}\n")

    # Persist the session on exit
    if memory.turn_count > 0:
        print("Saving session to store...", end=" ")
        episode = memory.end_session()
        print(f"done. (session {episode.session_id}, {episode.turns.__len__()} turns)")
        print(f"Store now has {store.count()} episode(s).\n")
    else:
        print("No turns to save.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    parser.add_argument(
        "--store",
        choices=["json", "sqlite"],
        default="json",
        help="Storage backend (default: json)",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=DEFAULT_STORE_DIR,
        help=f"Directory for episode storage (default: {DEFAULT_STORE_DIR})",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe the episode store before starting",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.real and not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    if args.clear and args.store_dir.exists():
        shutil.rmtree(args.store_dir)
        print(f"Cleared store at {args.store_dir}\n")

    run_demo(
        use_mock=args.mock,
        use_sqlite=(args.store == "sqlite"),
        store_dir=args.store_dir,
        api_key=api_key,
    )
