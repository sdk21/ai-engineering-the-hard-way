"""
Demo: Provenance and Confidence Memory
Usage:
    uv run python memory/08-provenance-confidence-memory/demo.py --mock
    uv run python memory/08-provenance-confidence-memory/demo.py --real

Key behaviour to observe:
  - State a fact → it gets a confidence score.
  - State the same fact again → score rises (reinforcement).
  - State a contradicting fact → score drops and the conflict is recorded.
  - Use 'facts' to see all facts with their confidence and provenance.
  - Use 'suppressed' to see facts below the injection threshold.
  - After many turns without reinforcement, use 'decay' to force decay
    and watch low-assertion facts fade below the threshold.

Suggested conversation to exercise all behaviours:

    # New facts
    1.  "My name is Alice."
    2.  "I work as a software engineer."
    3.  "I live in Tokyo."

    # Reinforcement
    4.  "Just to confirm — my name is Alice."
    5.  facts  ← Alice.name score rose

    # Contradiction
    6.  "Actually, I'm a backend architect, not an engineer."
    7.  facts  ← role score dropped, conflict recorded

    # Suppression via contradiction
    8.  "I live in Osaka."         ← contradicts turn 3
    9.  "I live in Kyoto."         ← contradicts again
    10. facts  ← location near suppression threshold

    # Force decay to show suppression
    11. decay 30                   ← simulate 30 idle turns
    12. facts  ← low-confidence facts suppressed from prompt

    # Audit trail
    13. audit  ← full provenance for every fact
"""

import argparse
import json
import os
import re
import sys

from experiment import Assertion, FactExtractorFn, ProvenanceMemory


# ---------------------------------------------------------------------------
# Mock extractor — regex-based fact extraction
# ---------------------------------------------------------------------------

# Each pattern: (regex, entity, attribute, value_group)
_FACT_PATTERNS: list[tuple[str, str, str, int]] = [
    # "My name is Alice"
    (r"\bmy name is\s+([A-Z][a-z]+)", "user", "name", 1),
    # "I work as a software engineer" / "I work as an architect"
    (r"\bi work(?:ed)? as an?\s+([\w][\w\s]+?)(?:\.|,|$)", "user", "role", 1),
    # "I'm a backend architect" / "I am a senior engineer"
    (r"\bi(?:'m| am) an?\s+([\w]+(?:\s+[\w]+){0,3}?)(?:\s+now|\.|,|$)", "user", "role", 1),
    # "I live in Tokyo" / "I moved to Osaka"
    (r"\bi\s+(?:live|moved|am based)\s+(?:in|to)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", "user", "location", 1),
    # "I prefer Python" / "I use Go"
    (r"\bi\s+(?:prefer|love|use|like)\s+([A-Za-z][a-zA-Z+#\.]+)(?:\s+over|\s+for|\.|,|$)", "user", "language", 1),
    # "I'm allergic to X"
    (r"\ballergic to\s+([\w][\w\s]+?)(?:\.|,|$)", "user", "allergy", 1),
    # "I'm X years old"
    (r"\bi(?:'m| am)\s+(\d+)\s+years?\s+old", "user", "age", 1),
    # "My project is called X" / "I'm working on X"
    (r"\bmy project (?:is called|is named|is)\s+([A-Z][a-zA-Z]+)", "project", "name", 1),
    (r"\bi(?:'m| am) working on\s+([A-Z][a-zA-Z]+)", "project", "name", 1),
]


def mock_extractor(role: str, content: str, turn_index: int) -> list[Assertion]:
    if role != "user":
        return []

    assertions: list[Assertion] = []
    seen_keys: set[tuple[str, str]] = set()

    for pattern, entity, attribute, vg in _FACT_PATTERNS:
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            value = m.group(vg).strip().rstrip(".,!")
            if not value or len(value) > 60:
                continue
            key = (entity, attribute)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            assertions.append(Assertion(
                entity=entity,
                attribute=attribute,
                value=value,
                source_text=content.strip(),
                extractor="regex",
            ))

    return assertions


# ---------------------------------------------------------------------------
# Real extractor — Claude returns structured JSON
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """\
Extract factual claims about entities from the following message.
Return a JSON array where each item has:
  entity     (string) — the entity the fact is about: "user", or a proper noun
  attribute  (string) — the property being stated, e.g. "name", "role", "location",
                        "language", "allergy", "age", "project_name"
  value      (string) — the asserted value

Rules:
- Only extract explicitly stated facts. Do not infer.
- Use lowercase for entity and attribute.
- If nothing is extractable, return [].

Message (turn {turn_index}, user): {content}

Return only valid JSON, nothing else."""


def make_real_extractor(api_key: str) -> FactExtractorFn:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def extractor(role: str, content: str, turn_index: int) -> list[Assertion]:
        if role != "user":
            return []
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(
                turn_index=turn_index, content=content,
            )}],
        )
        raw = response.content[0].text.strip()
        try:
            items = json.loads(raw)
            return [
                Assertion(
                    entity=item["entity"],
                    attribute=item["attribute"],
                    value=item["value"],
                    source_text=content.strip(),
                    extractor="llm",
                )
                for item in items
                if all(k in item for k in ("entity", "attribute", "value"))
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    return extractor


# ---------------------------------------------------------------------------
# Chat backends
# ---------------------------------------------------------------------------

def mock_chat(messages: list[dict], system: str) -> str:
    last = messages[-1]["content"].lower() if messages else ""
    ctx = system.lower()

    for question, attr in [
        ("name", "name"),
        ("live", "location"),
        ("work", "role"),
        ("role", "role"),
        ("allergic", "allergy"),
        ("language", "language"),
        ("prefer", "language"),
        ("age", "age"),
        ("project", "name"),
    ]:
        if question in last:
            # Find attr in the facts section
            m = re.search(rf"user:\n(?:.*\n)*?\s+{attr}:\s+([\w\s]+?)(?:\s+\[|\n|$)", ctx)
            if m:
                return f"Based on my records, your {attr} is: {m.group(1).strip()}."

    fact_count = ctx.count("conf=") if "conf=" in ctx else ctx.count(": ")
    return (
        f"[Mock] I have {fact_count} facts injected in my context. "
        f"Ask me about your name, role, location, language, age, allergies, or project."
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

OUTCOME_SYMBOL = {
    "new":          "+",
    "reinforced":   "↑",
    "contradicted": "⚡",
}


def run_demo(use_mock: bool, api_key: str | None) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Provenance & Confidence Memory Demo [{mode}] ===")
    print("Commands: 'facts', 'audit', 'suppressed', 'decay N', 'stats', 'quit'\n")

    extractor = mock_extractor if use_mock else make_real_extractor(api_key)
    memory = ProvenanceMemory(extractor=extractor, session_id="demo-session")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() == "quit":
            break

        # --- commands ---

        if user_input.lower() == "facts":
            facts = memory.store.injectable_facts()
            if not facts:
                print("\n(no facts above confidence threshold)\n")
            else:
                print(f"\n--- Injectable Facts ({len(facts)} facts, threshold={memory.store.injection_threshold:.0%}) ---")
                for r in facts:
                    reinf = f"  reinforced×{r.confidence.reinforcement_count}" if r.confidence.reinforcement_count else ""
                    confl = f"  ⚡ {r.confidence.contradiction_count} conflict(s)" if r.confidence.contradiction_count else ""
                    print(f"  {r.entity}.{r.attribute} = {r.value!r:<20} conf={r.confidence.score:.3f} ({r.confidence.label}){reinf}{confl}")
            print()
            continue

        if user_input.lower() == "suppressed":
            facts = memory.store.suppressed_facts()
            if not facts:
                print("\n(no suppressed facts)\n")
            else:
                print(f"\n--- Suppressed Facts ({len(facts)}, below {memory.store.injection_threshold:.0%} threshold) ---")
                for r in facts:
                    confl = f"  ⚡ {r.confidence.contradiction_count} conflict(s)" if r.confidence.contradiction_count else ""
                    print(f"  {r.entity}.{r.attribute} = {r.value!r:<20} conf={r.confidence.score:.3f}{confl}")
            print()
            continue

        if user_input.lower() == "audit":
            print(f"\n--- Full Audit Trail ({len(memory.store)} facts) ---")
            print(memory.store.format_for_audit())
            print()
            continue

        if user_input.lower().startswith("decay"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            before = {r.key: r.confidence.score for r in memory.store.all_facts()}
            # Simulate N idle turns by advancing turn_index without assertions
            for _ in range(n):
                memory._turn_index += 1
                memory.store.apply_decay(memory._turn_index)
            print(f"\nSimulated {n} idle turns (turn now = {memory.turn_index}):")
            for r in memory.store.all_facts():
                old = before[r.key]
                delta = r.confidence.score - old
                marker = "▼ SUPPRESSED" if r.confidence.score < memory.store.injection_threshold and old >= memory.store.injection_threshold else ""
                print(f"  {r.entity}.{r.attribute}: {old:.3f} → {r.confidence.score:.3f} ({delta:+.3f}) {marker}")
            print()
            continue

        if user_input.lower() == "stats":
            print(
                f"[turn={memory.turn_index} | "
                f"total_facts={memory.total_facts} | "
                f"injectable={memory.injectable_count} | "
                f"suppressed={memory.suppressed_count} | "
                f"threshold={memory.store.injection_threshold:.0%} | "
                f"buffer={len(memory.get_messages())}]\n"
            )
            continue

        # --- normal turn ---
        results = memory.add_user_message(user_input)
        system = memory.get_system_prompt()

        if use_mock:
            reply = mock_chat(memory.get_messages(), system)
        else:
            reply = real_chat(memory.get_messages(), system, api_key)

        memory.add_assistant_message(reply)
        print(f"Assistant: {reply}")

        if results:
            for assertion, outcome in results:
                record = memory.store.get(assertion.entity, assertion.attribute)
                score_str = f"{record.confidence.score:.3f}" if record else "?"
                symbol = OUTCOME_SYMBOL.get(outcome, "?")
                print(
                    f"  [{symbol}] {assertion.entity}.{assertion.attribute}"
                    f" = {assertion.value!r}  conf={score_str}  ({outcome})"
                )
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Confidence injection threshold (default: 0.40)",
    )
    parser.add_argument(
        "--half-life",
        type=float,
        default=20.0,
        help="Decay half-life in turns (default: 20)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.real and not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock, api_key=api_key)
