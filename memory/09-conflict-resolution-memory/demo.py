"""
Demo: Conflict Resolution Memory
Usage:
    uv run python memory/09-conflict-resolution-memory/demo.py --mock
    uv run python memory/09-conflict-resolution-memory/demo.py --real
    uv run python memory/09-conflict-resolution-memory/demo.py --mock --strategy highest_conf
    uv run python memory/09-conflict-resolution-memory/demo.py --mock --strategy user_arbitration

Available strategies: last_write, highest_conf, most_frequent,
                      source_priority, user_arbitration

Suggested conversation to exercise all five conflict types:

    # 1. Legitimate update (temporal marker → no penalty)
    "My name is Alice."
    "I work at Acme Corp."
    "I moved to Osaka."          ← temporal UPDATE, no confidence penalty

    # 2. Correction (near-match → soft penalty)
    "My name is Alicia."         ← CORRECTION of "Alice", edit-distance ≤ 2

    # 3. Genuine conflict (no temporal cue → hard penalty)
    "I work at Beta Corp."       ← CONFLICT (no "moved"/"left" etc.)

    # 4. Explicit retraction
    "I no longer work at Beta Corp."  ← RETRACTION

    # 5. User arbitration (with --strategy user_arbitration)
    "I live in Kyoto."           ← conflicts with "Osaka" → UNRESOLVED
    resolve location Osaka       ← you choose the winner
    audit                        ← see the resolution record

    # Dependency propagation
    "My role is engineer."
    "My team is backend."        ← depends on role
    "My role is manager."        ← role changes → team marked STALE
    audit

Commands: facts, suppressed, unresolved, audit, resolve <attr> <value>,
          stats, quit
"""

import argparse
import json
import os
import re
import sys

from experiment import (
    Assertion,
    ConflictResolutionMemory,
    ConflictResolutionStore,
    ConflictType,
    FactExtractorFn,
    Strategy,
)


# ---------------------------------------------------------------------------
# Mock extractor
# ---------------------------------------------------------------------------

_FACT_PATTERNS: list[tuple[str, str, str, int]] = [
    (r"\bmy name is\s+([A-Z][a-z]+)", "user", "name", 1),
    (r"\bi work(?:ed)? at\s+([\w][\w\s&]+?)(?:\.|,|$)", "user", "employer", 1),
    (r"\bi(?:'m| am) (?:now )?(?:an?\s+)?([\w]+(?:\s+[\w]+){0,2})\s+(?:at|for|engineer|developer|architect|manager|designer)(?:\b)", "user", "role", 1),
    (r"\bi work(?:ed)? as an?\s+([\w][\w\s]+?)(?:\.|,|$)", "user", "role", 1),
    (r"\bi(?:'m| am) (?:now )?(?:an?\s+)?([\w]+(?:\s+[\w]+){0,2})\b(?=\s*\.|\s*,|\s*$)", "user", "role", 1),
    (r"\bi\s+(?:live|moved|am based|moved to|relocated to)\s+(?:in|to)?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", "user", "location", 1),
    (r"\bmy (?:current )?(?:role|title|job) is\s+([\w][\w\s]+?)(?:\.|,|$)", "user", "role", 1),
    (r"\bmy team is\s+([\w][\w\s]+?)(?:\.|,|$)", "user", "team", 1),
    (r"\bi prefer\s+([A-Za-z][a-zA-Z+#\.]+)(?:\s+over|\.|,|$)", "user", "language", 1),
]

_RETRACTION_PATTERNS: list[tuple[str, str, str]] = [
    (r"\bno longer work(?:ing)? at\s+([\w][\w\s&]+?)(?:\.|,|$)", "user", "employer"),
    (r"\bleft\s+([\w][\w\s&]+?)(?:\.|,|$)", "user", "employer"),
    (r"\bquit\s+([\w][\w\s&]+?)(?:\.|,|$)", "user", "employer"),
    (r"\bno longer\s+(?:a |an )?([\w][\w\s]+?)(?:\.|,|$)", "user", "role"),
]


def mock_extractor(role: str, content: str, turn_index: int) -> list[Assertion]:
    if role != "user":
        return []

    assertions: list[Assertion] = []
    seen_keys: set[tuple[str, str]] = set()

    # Check for retractions first
    for pattern, entity, attribute in _RETRACTION_PATTERNS:
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            key = (entity, attribute)
            if key not in seen_keys:
                seen_keys.add(key)
                assertions.append(Assertion(
                    entity=entity,
                    attribute=attribute,
                    value=None,   # retraction
                    source_text=content.strip(),
                    extractor="regex",
                ))

    # Then regular facts
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
# Real extractor
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """\
Extract factual claims and retractions from the following message.
Return a JSON array where each item has:
  entity     (string) — "user" for self-statements, or a proper noun
  attribute  (string) — e.g. "name", "role", "employer", "location", "team", "language"
  value      (string or null) — the asserted value; null if the fact is being retracted
                                (e.g. "I no longer work at Acme" → value: null, attribute: "employer")

Rules:
- Only extract explicitly stated facts. Do not infer.
- Use null for retractions ("no longer", "left", "quit", "not anymore").
- Use lowercase for entity and attribute.
- Return [] if nothing extractable.

Message (turn {turn_index}): {content}

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
                    value=item.get("value"),   # None for retractions
                    source_text=content.strip(),
                    extractor="llm",
                )
                for item in items
                if "entity" in item and "attribute" in item
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
        ("name", "name"), ("work at", "employer"), ("employer", "employer"),
        ("role", "role"), ("live", "location"), ("location", "location"),
        ("team", "team"), ("language", "language"), ("prefer", "language"),
    ]:
        if question in last:
            m = re.search(
                rf"user:\n(?:.*\n)*?\s+{re.escape(attr)}:\s+([\w][\w\s,&]+?)(?:\s+\[|\n|$)",
                ctx,
            )
            if m:
                return f"Based on my records, your {attr} is: {m.group(1).strip()}."
            return f"I don't have your {attr} recorded (it may be suppressed or retracted)."

    if "unresolved" in ctx:
        m = re.search(r"unresolved conflicts.*?\n\s+([\w.]+)", ctx)
        if m:
            return f"I notice there's an unresolved conflict on {m.group(1)} — please use 'resolve' to clarify."

    injectable = ctx.count(": ") - ctx.count("## ")
    return (
        f"[Mock] {injectable} fact(s) injected. "
        f"Ask about your name, employer, role, location, team, or language."
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
# Outcome display
# ---------------------------------------------------------------------------

OUTCOME_SYMBOL = {
    "new":         "[+]",
    "reinforced":  "[↑]",
    "update":      "[→]",   # temporal update, no penalty
    "correction":  "[~]",   # near-match, soft penalty
    "conflict":    "[⚡]",   # genuine conflict, hard penalty
    "retraction":  "[✗]",   # fact retracted
    "unresolved":  "[?]",   # awaiting user arbitration
}

CONFLICT_TYPE_NOTE = {
    ConflictType.UPDATE:     "temporal update — no penalty",
    ConflictType.CORRECTION: "near-match correction — soft penalty",
    ConflictType.CONFLICT:   "genuine conflict — hard penalty",
    ConflictType.RETRACTION: "explicit retraction",
}


# ---------------------------------------------------------------------------
# Demo loop
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "last_write":       Strategy.LAST_WRITE,
    "highest_conf":     Strategy.HIGHEST_CONF,
    "most_frequent":    Strategy.MOST_FREQUENT,
    "source_priority":  Strategy.SOURCE_PRIORITY,
    "user_arbitration": Strategy.USER_ARBITRATION,
}


def run_demo(use_mock: bool, strategy: Strategy, api_key: str | None) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Conflict Resolution Memory Demo [{mode}] | strategy={strategy.name} ===")
    print("Commands: facts, suppressed, unresolved, audit, resolve <attr> <value>, stats, quit\n")

    # Build store with a dependency: if role changes, team becomes stale
    store = ConflictResolutionStore(
        default_strategy=strategy,
        attribute_strategies={
            "name":     Strategy.HIGHEST_CONF,    # corrections rare; trust confident value
            "employer": Strategy.SOURCE_PRIORITY, # manual notes trump extractions
        },
        dependency_map={
            ("user", "employer"): [("user", "team")],
            ("user", "role"):     [("user", "team")],
        },
    )

    extractor = mock_extractor if use_mock else make_real_extractor(api_key)
    memory = ConflictResolutionMemory(extractor=extractor, store=store)

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
            facts = store.injectable_facts()
            if not facts:
                print("\n(no injectable facts)\n")
            else:
                print(f"\n--- Injectable Facts ({len(facts)}, threshold={store.injection_threshold:.0%}) ---")
                for r in facts:
                    conflicts = f"  {len(r.conflicts)} conflict(s)" if r.conflicts else ""
                    print(f"  {r.entity}.{r.attribute} = {r.value!r:<25} conf={r.confidence.score:.3f} ({r.confidence.label}){conflicts}")
            print()
            continue

        if user_input.lower() == "suppressed":
            suppressed = [r for r in store.all_facts()
                          if r.confidence.score < store.injection_threshold
                          or r.status.value != "active"]
            if not suppressed:
                print("\n(no suppressed or non-active facts)\n")
            else:
                print(f"\n--- Non-injectable Facts ---")
                for r in suppressed:
                    print(f"  {r.entity}.{r.attribute} = {r.value!r}  [{r.status.value}] conf={r.confidence.score:.3f}")
            print()
            continue

        if user_input.lower() == "unresolved":
            pending = store.unresolved_conflicts()
            if not pending:
                print("\n(no unresolved conflicts)\n")
            else:
                print(f"\n--- Unresolved Conflicts ({len(pending)}) ---")
                for r in pending:
                    last_c = r.conflicts[-1]
                    print(f"  {r.entity}.{r.attribute}: old={last_c.old_value!r}  new={last_c.new_value!r}")
                    print(f"    Use: resolve {r.attribute} <your chosen value>")
            print()
            continue

        if user_input.lower() == "audit":
            print(f"\n--- Full Audit Trail ({len(store)} facts) ---")
            print(store.format_for_audit())
            print()
            continue

        if user_input.lower().startswith("resolve "):
            parts = user_input.split(None, 2)
            if len(parts) < 3:
                print("Usage: resolve <attribute> <value>\n")
                continue
            attr, chosen = parts[1], parts[2]
            ok = store.resolve_conflict("user", attr, chosen)
            if ok:
                r = store.get("user", attr)
                print(f"Resolved user.{attr} = {chosen!r}  (conf={r.confidence.score:.3f})\n")
            else:
                print(f"No unresolved conflict found for user.{attr}\n")
            continue

        if user_input.lower() == "stats":
            all_f = store.all_facts()
            injectable = store.injectable_facts()
            unresolved = store.unresolved_conflicts()
            print(
                f"[turn={memory.turn_index} | total={len(all_f)} | "
                f"injectable={len(injectable)} | "
                f"unresolved={len(unresolved)} | "
                f"strategy={strategy.name}]\n"
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

        for assertion, outcome, conflict in results:
            symbol = OUTCOME_SYMBOL.get(outcome, "[?]")
            record = store.get(assertion.entity, assertion.attribute)
            score_str = f"{record.confidence.score:.3f}" if record else "?"
            note = ""
            if conflict:
                note = f"  — {CONFLICT_TYPE_NOTE.get(conflict.conflict_type, '')}"
                if conflict.winner and conflict.winner != conflict.old_value:
                    note += f"  winner={conflict.winner!r}"
                elif conflict.winner == conflict.old_value:
                    note += f"  (kept old value)"
            print(
                f"  {symbol} {assertion.entity}.{assertion.attribute}"
                f" = {str(assertion.value)!r:<20} conf={score_str}{note}"
            )
        if results:
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
        "--strategy",
        choices=list(STRATEGY_MAP.keys()),
        default="last_write",
        help="Default conflict resolution strategy (default: last_write)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.real and not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(
        use_mock=args.mock,
        strategy=STRATEGY_MAP[args.strategy],
        api_key=api_key,
    )
