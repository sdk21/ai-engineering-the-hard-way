"""
Personal Assistant — Memory Vertical Capstone
Usage:
    uv run python memory/capstone/personal-assistant/assistant.py --mock
    uv run python memory/capstone/personal-assistant/assistant.py --real
    uv run python memory/capstone/personal-assistant/assistant.py --real --new
    uv run python memory/capstone/personal-assistant/assistant.py --real --clear

Data is persisted to ~/.config/ai-hardway/assistant/ by default.
Use --data-dir to override.

Commands during a session:
    memory          — inspect full memory state (all layers)
    prompt          — show the assembled system prompt
    prefix          — show the stable (cacheable) prefix only
    forget <entity> <attribute>   — delete a specific fact
    endsession      — save this session to episodic layer and start fresh
    stats           — one-line layer counts
    quit / exit     — save and exit

Production features demonstrated:
    Persistence     — quit and restart; memory is intact
    Session resume  — restart within 4h; conversation buffer is restored
    Prompt caching  — 'prefix' shows the stable cache-worthy portion
    TTL cleanup     — stale facts pruned on startup (see startup log)
    Async writes    — extraction runs after the LLM response (simulated)
"""

import argparse
import os
import re
import sys
from pathlib import Path

from memory import Embedder, ExtractorFn, ProductionMemory, SummarizerFn


# ---------------------------------------------------------------------------
# Default data directory
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path.home() / ".config" / "ai-hardway" / "assistant"


# ---------------------------------------------------------------------------
# Mock embedder — bag-of-words, no dependencies
# ---------------------------------------------------------------------------

_VOCAB = [
    "name", "work", "project", "team", "language", "use", "build",
    "engineer", "manager", "developer", "architect", "alice", "bob",
    "orion", "redis", "kafka", "acme", "beta", "infrastructure",
    "remember", "last", "previous", "session", "colleague",
    "live", "location", "role", "employer", "technology",
]

def _bow(text: str) -> list[float]:
    import math
    words = re.findall(r"\b\w+\b", text.lower())
    vec = [float(words.count(w)) for w in _VOCAB]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec

class MockEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [_bow(t) for t in texts]

class RealEmbedder:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

_FACT_PATTERNS = [
    (r"\bmy name is\s+([A-Z][a-z]+)", "user", "name", 1),
    (r"\bi work(?:ed)? at\s+([\w][\w\s&]+?)(?:\s+on\s|\s+and\s|\.|,|$)", "user", "employer", 1),
    (r"\bi work(?:ed)? as an?\s+([\w][\w\s]+?)(?:\.|,|$)", "user", "role", 1),
    (r"\bi(?:'m| am) an?\s+([\w]+(?:\s+[\w]+){0,2}\s+(?:engineer|developer|architect|manager|designer))", "user", "role", 1),
    (r"\bi\s+(?:live|moved)\s+(?:in|to)\s+([A-Z][a-zA-Z]+)", "user", "location", 1),
    (r"\bmy (?:role|title|job) is\s+([\w][\w\s]+?)(?:\.|,|$)", "user", "role", 1),
    (r"\bi(?:'m| am) working on\s+(?:project\s+)?([A-Z][a-zA-Z]+)", "project", "name", 1),
    (r"\bon project\s+([A-Z][a-zA-Z]+)", "project", "name", 1),
    (r"\bmy (?:preferred\s+)?(?:language|stack) is\s+([\w+#\.]+)", "user", "language", 1),
    (r"\bi prefer\s+([A-Za-z][a-zA-Z+#\.]+)(?:\s+over|\.|,|$)", "user", "language", 1),
]

_EDGE_PATTERNS = [
    (r"\b([A-Z][a-zA-Z]+)\s+uses?\s+([A-Z][a-zA-Z]+)\s+and\s+([A-Z][a-zA-Z]+)",
     "multi_uses"),
    (r"\b([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\s+(?:both\s+)?work\s+on\s+(?:project\s+)?([A-Z][a-zA-Z]+)",
     "multi_works_on"),
    (r"\b([A-Z][a-zA-Z]+)\s+uses?\s+([A-Z][a-zA-Z]+)", "uses"),
    (r"\b([A-Z][a-z]+)\s+reports?\s+to\s+([A-Z][a-z]+)", "reports_to"),
    (r"\b([A-Z][a-z]+)\s+works?\s+on\s+(?:project\s+)?([A-Z][a-zA-Z]+)", "works_on"),
    (r"\b([A-Z][a-z]+)\s+is my colleague", "colleague"),
]

def mock_extractor(role: str, content: str) -> tuple[list[tuple], list[tuple]]:
    if role != "user":
        return [], []

    facts: list[tuple] = []
    seen_f: set[tuple] = set()
    for pattern, entity, attribute, vg in _FACT_PATTERNS:
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            value = m.group(vg).strip().rstrip(".,!")
            if not value or len(value) > 60:
                continue
            key = (entity, attribute)
            if key not in seen_f:
                seen_f.add(key)
                facts.append((entity, attribute, value))

    edges: list[tuple] = []
    seen_e: set[tuple] = set()

    for pattern, kind in _EDGE_PATTERNS:
        m = re.search(pattern, content)
        if not m:
            continue
        if kind == "multi_uses":
            for tech in (m.group(2), m.group(3)):
                k = (m.group(1).lower(), "USES", tech.lower())
                if k not in seen_e:
                    seen_e.add(k); edges.append((m.group(1), "USES", tech))
        elif kind == "multi_works_on":
            for person in (m.group(1), m.group(2)):
                k = (person.lower(), "WORKS_ON", m.group(3).lower())
                if k not in seen_e:
                    seen_e.add(k); edges.append((person, "WORKS_ON", m.group(3)))
        elif kind == "uses":
            k = (m.group(1).lower(), "USES", m.group(2).lower())
            if k not in seen_e:
                seen_e.add(k); edges.append((m.group(1), "USES", m.group(2)))
        elif kind == "reports_to":
            k = (m.group(1).lower(), "REPORTS_TO", m.group(2).lower())
            if k not in seen_e:
                seen_e.add(k); edges.append((m.group(1), "REPORTS_TO", m.group(2)))
        elif kind == "works_on":
            k = (m.group(1).lower(), "WORKS_ON", m.group(2).lower())
            if k not in seen_e:
                seen_e.add(k); edges.append((m.group(1), "WORKS_ON", m.group(2)))
        elif kind == "colleague":
            k = ("user", "COLLEAGUE_OF", m.group(1).lower())
            if k not in seen_e:
                seen_e.add(k); edges.append(("user", "COLLEAGUE_OF", m.group(1)))

    return facts, edges


def make_real_extractor(api_key: str) -> ExtractorFn:
    import anthropic, json as _json
    client = anthropic.Anthropic(api_key=api_key)
    PROMPT = """\
Extract facts and relationships from this message.
Return JSON: {{"facts": [[entity, attribute, value], ...], "edges": [[source, RELATION, target], ...]}}
Only explicit facts. entity/attribute in lowercase. Relations in ALL_CAPS.
Message: {content}
Return only valid JSON."""

    def extractor(role: str, content: str) -> tuple[list[tuple], list[tuple]]:
        if role != "user":
            return [], []
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=256,
            messages=[{"role": "user", "content": PROMPT.format(content=content)}],
        )
        try:
            d = _json.loads(r.content[0].text.strip())
            facts = [tuple(f) for f in d.get("facts", []) if len(f) == 3]
            edges = [tuple(e) for e in d.get("edges", []) if len(e) == 3]
            return facts, edges
        except Exception:
            return [], []
    return extractor


# ---------------------------------------------------------------------------
# Summarizers
# ---------------------------------------------------------------------------

def mock_summarizer(turns: list[dict]) -> str:
    user_msgs = [t["content"] for t in turns if t["role"] == "user"]
    if not user_msgs:
        return "Empty session."
    first = user_msgs[0][:80].rstrip(".,")
    m = re.search(r"\bmy name is\s+([A-Z][a-z]+)", " ".join(user_msgs), re.IGNORECASE)
    name_note = f" User's name: {m.group(1)}." if m else ""
    return f"User said: '{first}'.{name_note} {len(user_msgs)} user turn(s)."

def make_real_summarizer(api_key: str) -> SummarizerFn:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    PROMPT = "Summarise this conversation in 1-2 sentences. Focus on who the user is and what was discussed.\n\n{transcript}\n\nReturn only the summary."
    def summarizer(turns: list[dict]) -> str:
        transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=150,
            messages=[{"role": "user", "content": PROMPT.format(transcript=transcript)}],
        )
        return r.content[0].text.strip()
    return summarizer


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def mock_chat(messages: list[dict], system: str) -> str:
    last = messages[-1]["content"].lower() if messages else ""
    ctx = system.lower()

    for q, attr in [
        ("name", "name"), ("work at", "employer"), ("employer", "employer"),
        ("project", "name"), ("role", "role"), ("live", "location"),
        ("language", "language"), ("prefer", "language"),
    ]:
        if q in last:
            m = re.search(
                rf"(?:user|project):\n.*?{re.escape(attr)}:\s+([\w][\w\s,&+#\.]+?)(?:\s+\[|\n|$)",
                ctx, re.DOTALL,
            )
            if m:
                return f"Based on my memory: your {attr} is {m.group(1).strip()!r}."

    if any(w in last for w in ["remember", "last time", "previous", "before", "session"]):
        if "previous sessions" in ctx:
            ep = re.search(r"\[(\d{4}-\d{2}-\d{2})\]\s+(.+)", ctx)
            if ep:
                return f"Yes — in our session on {ep.group(1)}: \"{ep.group(2)[:100]}\""
        return "I don't have any prior sessions recorded yet."

    if "use" in last or "technolog" in last:
        techs = re.findall(r"--\[uses\]-->\s+(\w+)", ctx)
        if techs:
            return f"Based on my knowledge graph: {', '.join(t.capitalize() for t in techs)}."

    layers = []
    if "previous sessions" in ctx:   layers.append("episodic")
    if "known facts" in ctx:          layers.append("structured")
    if "relevant past" in ctx:        layers.append("semantic")
    return (
        f"[Mock] Active layers: {', '.join(layers) or 'working only'}. "
        "Ask about your name, employer, project, role, or past sessions."
    )

def real_chat(messages: list[dict], system: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=system, messages=messages,
    )
    return r.content[0].text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    use_mock: bool,
    data_dir: Path,
    force_new: bool,
    api_key: str | None,
) -> None:
    mode = "MOCK" if use_mock else "REAL"
    print(f"\n╔══ Personal Assistant [{mode}] ══╗")
    print(f"  Data: {data_dir}")

    embedder:   Embedder    = MockEmbedder() if use_mock else RealEmbedder()
    extractor:  ExtractorFn = mock_extractor if use_mock else make_real_extractor(api_key)
    summarizer: SummarizerFn = mock_summarizer if use_mock else make_real_summarizer(api_key)

    memory = ProductionMemory(
        data_dir=data_dir,
        embedder=embedder,
        extractor=extractor,
        summarizer=summarizer,
    )

    if force_new and memory.was_resumed:
        memory.new_session()
        print("  Started new session (--new flag).")
    elif memory.was_resumed:
        print(f"  Resumed session {memory.session_id} "
              f"({len(memory.messages())} turn(s) in buffer).")
    else:
        print(f"  New session {memory.session_id}.")

    ep_count = memory.stats()["episodes"]
    if ep_count:
        print(f"  {ep_count} past episode(s) in memory.")

    print("\n  Commands: memory, prompt, prefix, forget <entity> <attr>, "
          "endsession, stats, quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            break

        if user_input.lower() == "memory":
            print("\n" + memory.inspect() + "\n")
            continue

        if user_input.lower() == "prompt":
            sp = memory.get_system_prompt()
            print(f"\n--- System Prompt ({len(sp)} chars) ---\n{sp}\n")
            continue

        if user_input.lower() == "prefix":
            p = memory.stable_prefix()
            print(f"\n--- Stable Prefix / Cache Target ({len(p)} chars) ---\n{p}")
            print("\n[This portion is suitable for prompt caching.]")
            print("[Only the semantic suffix changes per turn.]\n")
            continue

        if user_input.lower().startswith("forget "):
            parts = user_input.split(None, 2)
            if len(parts) < 3:
                print("Usage: forget <entity> <attribute>\n"); continue
            ok = memory._store.delete_fact(parts[1], parts[2])
            print(f"{'Deleted' if ok else 'Not found'}: {parts[1]}.{parts[2]}\n")
            continue

        if user_input.lower() == "endsession":
            summary = memory.end_session()
            if summary:
                print(f"Session saved: \"{summary}\"\n")
                memory.new_session()
                print(f"New session started: {memory.session_id}\n")
            else:
                print("No turns to save.\n")
            continue

        if user_input.lower() == "stats":
            s = memory.stats()
            print(
                f"[session={s['session_id']} | turn={s['turn']} | "
                f"buffer={s['buffer']} | facts={s['facts']} | "
                f"episodes={s['episodes']} | semantic={s['semantic_idx']}]\n"
            )
            continue

        # Normal turn
        result = memory.add_user_message(user_input)
        system = memory.get_system_prompt(query=user_input)

        reply = mock_chat(memory.messages(), system) if use_mock \
                else real_chat(memory.messages(), system, api_key)

        memory.add_assistant_message(reply)
        print(f"Assistant: {reply}")

        if result["facts"] or result["edges"]:
            parts = []
            if result["facts"]:
                parts.append("+facts: " + ", ".join(
                    f"{e}.{a}={v!r}({o[0]})" for e, a, v, o in result["facts"]
                ))
            if result["edges"]:
                parts.append("+edges: " + ", ".join(
                    f"{s}—[{r}]→{t}" for s, r, t in result["edges"]
                ))
            print(f"  [mem] {' | '.join(parts)}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Personal Assistant — Memory Capstone")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="No API calls")
    group.add_argument("--real", action="store_true", help="Use Claude API")
    parser.add_argument("--new",      action="store_true", help="Force a new session")
    parser.add_argument("--clear",    action="store_true", help="Wipe all memory and start fresh")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Memory storage directory (default: {DEFAULT_DATA_DIR})")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.real and not api_key:
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)

    if args.clear and args.data_dir.exists():
        import shutil
        shutil.rmtree(args.data_dir)
        print(f"Cleared memory at {args.data_dir}\n")

    run(
        use_mock=args.mock,
        data_dir=args.data_dir,
        force_new=args.new,
        api_key=api_key,
    )
