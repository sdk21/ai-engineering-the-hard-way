"""
Demo: Summarization Memory
Usage:
    uv run python demo.py --mock
    uv run python demo.py --real
    uv run python demo.py --mock --buffer 4   # smaller buffer, more frequent summarization

Try this sequence to see summarization in action:
    1. "My name is Alice and I work as a software engineer."
    2. "I live in Tokyo."
    3. "My favourite language is Python."
    4. "What's the weather like today?" (filler)
    5. "Tell me a joke." (filler)
    6. "What's my name?"       ← tests whether name survived summarization
    7. "Where do I live?"      ← tests whether location survived
"""

import argparse
import os
import sys

from experiment import SummarizationMemory, SummarizerFn


# ---------------------------------------------------------------------------
# Mock summarizer — extracts key facts without an LLM call
# ---------------------------------------------------------------------------

def mock_summarizer(existing_summary: str, messages: list[dict]) -> str:
    """
    Naive fact extractor: looks for 'name is', 'live in', 'work as', etc.
    Good enough to demonstrate that key facts survive compression.
    """
    facts = []
    for m in messages:
        content = m["content"]
        role = m["role"].upper()
        # Truncate long messages to their first sentence for the summary
        first_sentence = content.split(".")[0].strip()
        if first_sentence:
            facts.append(f"[{role}] {first_sentence}.")

    new_lines = "\n".join(facts)
    if existing_summary:
        return f"{existing_summary}\n{new_lines}"
    return new_lines


# ---------------------------------------------------------------------------
# Real summarizer — uses Claude to produce a coherent rolling summary
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT = """\
You are maintaining a running summary of a conversation.

Existing summary (may be empty):
{existing_summary}

New messages to incorporate:
{messages}

Update the summary to include all important facts, decisions, and context from \
the new messages. Merge naturally with the existing summary. Be concise — \
preserve facts, names, goals, and decisions. Omit small talk and filler.

Return only the updated summary text, nothing else."""


def make_real_summarizer(api_key: str) -> SummarizerFn:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def summarizer(existing_summary: str, messages: list[dict]) -> str:
        formatted = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        prompt = SUMMARIZE_PROMPT.format(
            existing_summary=existing_summary or "(none yet)",
            messages=formatted,
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    return summarizer


# ---------------------------------------------------------------------------
# Chat function
# ---------------------------------------------------------------------------

def mock_chat(messages: list[dict], system: str) -> str:
    last = messages[-1]["content"].lower() if messages else ""
    # Scan both the recent buffer and the system prompt (which contains summary)
    full_context = system + " " + " ".join(m["content"] for m in messages)
    context_lower = full_context.lower()

    if "name" in last:
        for phrase in ["name is ", "i'm ", "i am ", "[user] my name is "]:
            if phrase in context_lower:
                idx = context_lower.index(phrase) + len(phrase)
                name = context_lower[idx:].split()[0].rstrip(".,!")
                return f"Your name is {name.capitalize()}."
        return "I don't have your name in my current context."
    if "live" in last or "where" in last:
        for phrase in ["live in ", "living in ", "i'm in ", "[user] i live in "]:
            if phrase in context_lower:
                idx = context_lower.index(phrase) + len(phrase)
                place = context_lower[idx:].split()[0].rstrip(".,!")
                return f"You live in {place.capitalize()}."
        return "I don't know where you live from the current context."
    return f"[Mock] I have {len(messages)} message(s) in my buffer plus a running summary."


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

def run_demo(use_mock: bool, buffer_size: int, api_key: str | None) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Summarization Memory Demo [{mode}] | buffer={buffer_size} ===")
    print("Old messages are compressed into a rolling summary instead of being dropped.")
    print("Commands: 'summary' to inspect current summary, 'stats' for memory stats, 'quit' to exit.\n")

    summarizer = mock_summarizer if use_mock else make_real_summarizer(api_key)
    memory = SummarizationMemory(summarizer=summarizer, buffer_size=buffer_size)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() == "quit":
            break

        if user_input.lower() == "summary":
            print(f"\n--- Current Summary ---")
            print(memory.summary or "(no summary yet)")
            print(f"-----------------------\n")
            continue

        if user_input.lower() == "stats":
            print(
                f"[buffer={memory.buffer_length}/{buffer_size} msgs | "
                f"summarizations={memory.summarization_count} | "
                f"~{memory.token_estimate} tokens total]\n"
            )
            continue

        memory.add_user_message(user_input)
        system = memory.get_system_prompt()

        if use_mock:
            reply = mock_chat(memory.get_messages(), system)
        else:
            reply = real_chat(memory.get_messages(), system, api_key)

        memory.add_assistant_message(reply)
        print(f"Assistant: {reply}")
        print(
            f"  [buffer={memory.buffer_length}/{buffer_size} | "
            f"summaries={memory.summarization_count}]\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    parser.add_argument("--buffer", type=int, default=6, help="Recent message buffer size")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.real and not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock, buffer_size=args.buffer, api_key=api_key)
