"""
Demo: Entity Memory
Usage:
    uv run python demo.py --mock
    uv run python demo.py --real

Suggested conversation to see entity extraction in action:
    1. "My name is Alice and I work as a software engineer."
    2. "I live in Tokyo."
    3. "I'm allergic to peanuts."
    4. "I prefer Python over JavaScript."
    5. "I'm building a project called Orion — a distributed cache."
    6. "Tell me a joke." (filler — no entities to extract)
    7. "Tell me another." (filler)
    8. "Tell me one more." (filler — buffer has moved past turn 1)
    9. Type 'entities' — inspect the structured store
   10. "What's my name?"         ← answered from entity store, not buffer
   11. "What am I allergic to?"  ← same
   12. "What project am I building?"
"""

import argparse
import json
import os
import sys

from experiment import EntityDict, EntityMemory


# ---------------------------------------------------------------------------
# Mock extractor — regex/keyword heuristics, no LLM call
# ---------------------------------------------------------------------------

import re

# Each pattern is (regex, entity_key, attribute, value_group_index)
# Ordered from most-specific to least-specific to avoid partial matches.
_PATTERNS: list[tuple[str, str, str, int]] = [
    # "My name is Alice"
    (r"\bmy name is\s+([A-Z][a-z]+)", "user", "name", 1),
    # "I work as a software engineer" / "I work as an engineer"
    (r"\bi work(?:ed)? as an?\s+([\w][\w\s]+?)(?:\.|,|and|$)", "user", "role", 1),
    # "I'm a software engineer" / "I am a developer" (only when followed by more context)
    (r"\bi(?:'m| am) an?\s+([\w]+(?:\s+[\w]+){1,3})\s+(?:living|based|working|who|and)", "user", "role", 1),
    # "I live in Tokyo" / "I'm based in Tokyo" / "I moved to Osaka"
    (r"\bi(?:'m| am)?\s+(?:live|lives|lived|based|moved)\s+(?:in|to)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)(?:\s+last|\s+recently|\.|,|$)", "user", "location", 1),
    # "I'm allergic to peanuts" / "allergic to tree nuts"
    (r"\ballergic to\s+([\w][\w\s]+?)(?:\.|,|and|$)", "user", "allergy", 1),
    # "I prefer Python" / "I love Python" (before "over" or end of clause)
    (r"\bi\s+(?:prefer|love|like|use)\s+([A-Za-z][a-zA-Z+#\.]+)(?:\s+over|\s+rather|\.|,|$)", "user", "preference", 1),
    # "project called Orion" / "project named Orion"
    (r"\bproject\s+(?:called|named)\s+([A-Z][a-zA-Z\s]+?)(?:\s*[—\-]|\.|,|$)", "project", "name", 1),
    # "a distributed cache" / "a real-time pipeline"
    (r"\ba\s+(distributed|real-time|web|mobile|data|ml|ai)\s+([\w\s]+?)(?:\.|,|$)", "project", "type", 2),
]


def mock_extractor(role: str, content: str) -> EntityDict:
    """Extract entities using regex patterns. Fast, deterministic, zero cost."""
    if role != "user":
        return {}

    results: EntityDict = {}
    text = content.strip()

    for pattern, entity_key, attribute, group_index in _PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(group_index).strip().rstrip(".,!")
            if value and len(value) < 60:   # skip garbage matches
                if entity_key not in results:
                    results[entity_key] = {}
                results[entity_key][attribute] = value

    return results


# ---------------------------------------------------------------------------
# Real extractor — LLM returns structured JSON
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """\
Extract named entities and their attributes from the following message.
Return a JSON object where each key is an entity name and the value is
an object of {attribute: value} pairs.

Only extract concrete facts explicitly stated. Do not infer or guess.
Focus on: people (names, roles, locations, preferences, constraints),
projects (names, types, technologies), and organizations.

If there is nothing to extract, return {}.

Message ({role}): {content}

Return only valid JSON, nothing else."""


def make_real_extractor(api_key: str):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def extractor(role: str, content: str) -> EntityDict:
        if role != "user":
            return {}
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": EXTRACT_PROMPT.format(role=role, content=content),
            }],
        )
        raw = response.content[0].text.strip()
        try:
            data = json.loads(raw)
            # Flatten any nested dicts deeper than 2 levels
            return {k: v for k, v in data.items() if isinstance(v, dict)}
        except json.JSONDecodeError:
            return {}

    return extractor


# ---------------------------------------------------------------------------
# Chat backends
# ---------------------------------------------------------------------------

def mock_chat(messages: list[dict], system: str) -> str:
    last = messages[-1]["content"].lower() if messages else ""
    ctx = system.lower()

    if "name" in last:
        # Look specifically under the user entity block
        m = re.search(r"user:.*?name:\s*(\w+)", ctx, re.DOTALL)
        if m:
            return f"Your name is {m.group(1).capitalize()}."
        return "I don't have your name in the entity store yet."

    if "allerg" in last:
        m = re.search(r"allergy:\s*([\w\s]+)", ctx)
        if m:
            return f"You're allergic to {m.group(1).strip()}."
        return "I don't have any allergy information recorded."

    if "project" in last:
        name = re.search(r"project:.*?name:\s*([\w\s]+)", ctx, re.DOTALL)
        kind = re.search(r"project:.*?type:\s*([\w\s]+)", ctx, re.DOTALL)
        if name:
            detail = f" — {kind.group(1).strip()}" if kind else ""
            return f"You're building {name.group(1).strip()}{detail}."
        return "I don't have a project recorded yet."

    if "prefer" in last or "language" in last:
        m = re.search(r"preference:\s*([\w\s+#]+)", ctx)
        if m:
            return f"You prefer {m.group(1).strip()}."
        return "No language preference recorded."

    return f"[Mock] Entity store has been populated. Ask me about your name, allergies, project, or preferences."


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
    print(f"\n=== Entity Memory Demo [{mode}] | buffer={buffer_size} ===")
    print("Facts are extracted from each turn and stored as structured entities.")
    print("Commands: 'entities' to inspect the store, 'stats' for counts, 'quit' to exit.\n")

    extractor = mock_extractor if use_mock else make_real_extractor(api_key)
    memory = EntityMemory(extractor=extractor, buffer_size=buffer_size)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() == "quit":
            break

        if user_input.lower() == "entities":
            store = memory.store.all()
            if not store:
                print("\n(no entities extracted yet)\n")
            else:
                print(f"\n--- Entity Store ({len(store)} entities) ---")
                print(memory.store.format_for_prompt())
                print()
            continue

        if user_input.lower() == "stats":
            print(
                f"[entities={len(memory.store)} | "
                f"buffer={memory.buffer_length}/{buffer_size} | "
                f"extractions={memory.extractions}]\n"
            )
            continue

        memory.add_user_message(user_input)
        system = memory.get_system_prompt()

        if use_mock:
            reply = mock_chat(memory.get_messages(), system)
        else:
            reply = real_chat(memory.get_messages(), system, api_key)

        memory.add_assistant_message(reply)
        extracted = memory.store.all()
        print(f"Assistant: {reply}")
        print(f"  [entities={len(extracted)} | buffer={memory.buffer_length}/{buffer_size}]\n")


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
