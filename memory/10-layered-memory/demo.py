"""
Demo: Layered Memory
Usage:
    uv run python memory/10-layered-memory/demo.py --mock
    uv run python memory/10-layered-memory/demo.py --real

Key behaviour to observe:
  - Layer 1 (working): recent turns appear in the messages list
  - Layer 2 (structured): facts and relationships extracted from each turn
  - Layer 3 (episodic): 'endsession' saves a summary; 'newsession' starts fresh
  - Layer 4 (semantic): past turns retrieved by similarity to current query
  - 'prompt' shows the assembled system prompt at any point
  - 'stats' shows a count of what each layer holds

Suggested sequence to exercise all layers:

    Session 1:
        "My name is Alice and I'm a software engineer."
        "I work at Acme Corp on project Orion."
        "Orion uses Redis and Kafka."
        "Bob is my colleague — he works on the infrastructure side."
        stats
        prompt  ← see all four layers assembled
        endsession  ← save to episodic layer

    Session 2 (same run, new session):
        newsession
        "Do you remember what project I'm working on?"
        ← episodic layer recalls session 1
        "What does Orion use?"
        ← structured (KG) layer answers
        stats
        prompt  ← episodic section now populated
"""

import argparse
import os
import re
import sys

from experiment import (
    Embedder,
    ExtractorFn,
    LayerBudget,
    LayeredMemory,
    SummarizerFn,
)


# ---------------------------------------------------------------------------
# Mock embedder — bag-of-words, no dependencies
# ---------------------------------------------------------------------------

_VOCAB = [
    "name", "work", "project", "team", "language", "tool", "use", "build",
    "engineer", "manager", "developer", "architect", "alice", "bob", "carol",
    "orion", "redis", "kafka", "acme", "beta", "infrastructure", "backend",
    "remember", "last", "time", "previous", "session", "colleague", "report",
    "live", "location", "city", "role", "employer", "technology", "stack",
]

def _bow(text: str) -> list[float]:
    words = re.findall(r"\b\w+\b", text.lower())
    vec = [float(words.count(w)) for w in _VOCAB]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


class MockEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [_bow(t) for t in texts]


# ---------------------------------------------------------------------------
# Real embedder — sentence-transformers
# ---------------------------------------------------------------------------

class RealEmbedder:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Mock extractor — regex, returns (entity_facts, kg_edges)
# ---------------------------------------------------------------------------

_FACT_PATTERNS: list[tuple[str, str, str, int]] = [
    (r"\bmy name is\s+([A-Z][a-z]+)", "user", "name", 1),
    # Stop employer capture at "on", "and", comma, or period
    (r"\bi work(?:ed)? at\s+([\w][\w\s&]+?)(?:\s+on\s|\s+and\s|\.|,|$)", "user", "employer", 1),
    (r"\bi work(?:ed)? as an?\s+([\w][\w\s]+?)(?:\.|,|$)", "user", "role", 1),
    # Capture full "software engineer" / "backend developer" etc.
    (r"\bi(?:'m| am) an?\s+([\w]+(?:\s+[\w]+){0,2}\s+(?:engineer|developer|architect|manager|designer))", "user", "role", 1),
    (r"\bi\s+(?:live|moved)\s+(?:in|to)\s+([A-Z][a-zA-Z]+)", "user", "location", 1),
    (r"\bmy (?:role|title|job) is\s+([\w][\w\s]+?)(?:\.|,|$)", "user", "role", 1),
    (r"\bi(?:'m| am) working on\s+(?:project\s+)?([A-Z][a-zA-Z]+)", "project", "name", 1),
    (r"\bproject\s+(?:called|named|is)\s+([A-Z][a-zA-Z]+)", "project", "name", 1),
    # "on project Orion" standalone
    (r"\bon project\s+([A-Z][a-zA-Z]+)", "project", "name", 1),
]

_EDGE_PATTERNS: list[tuple[str, str, str, int, int]] = [
    # Alice and Bob work on Orion → WORKS_ON
    (r"\b([A-Z][a-z]+)\s+(?:and\s+)?([A-Z][a-z]+)\s+(?:both\s+)?work\s+on\s+(?:project\s+)?([A-Z][a-zA-Z]+)",
     None, "WORKS_ON", None, 3),   # multi-subject handled separately
    # Orion uses Redis
    (r"\b([A-Z][a-zA-Z]+)\s+uses?\s+([A-Z][a-zA-Z]+)",
     1, "USES", 2, None),
    # Bob reports to Carol
    (r"\b([A-Z][a-z]+)\s+reports?\s+to\s+([A-Z][a-z]+)",
     1, "REPORTS_TO", 2, None),
    # Alice works on Orion
    (r"\b([A-Z][a-z]+)\s+works?\s+on\s+(?:project\s+)?([A-Z][a-zA-Z]+)",
     1, "WORKS_ON", 2, None),
    # Bob is my colleague
    (r"\b([A-Z][a-z]+)\s+is my colleague",
     1, "COLLEAGUE_OF", None, None),   # special-cased below
]


def mock_extractor(role: str, content: str) -> tuple[list[tuple], list[tuple]]:
    if role != "user":
        return [], []

    facts: list[tuple] = []
    seen_fact_keys: set[tuple] = set()

    for pattern, entity, attribute, vg in _FACT_PATTERNS:
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            value = m.group(vg).strip().rstrip(".,!")
            if not value or len(value) > 60:
                continue
            key = (entity, attribute)
            if key not in seen_fact_keys:
                seen_fact_keys.add(key)
                facts.append((entity, attribute, value))

    edges: list[tuple] = []
    seen_edge_keys: set[tuple] = set()

    # Multi-subject: "Alice and Bob work on Orion"
    ms = re.search(
        r"\b([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\s+(?:both\s+)?work\s+on\s+(?:project\s+)?([A-Z][a-zA-Z]+)",
        content,
    )
    if ms:
        for person in (ms.group(1), ms.group(2)):
            k = (person.lower(), "WORKS_ON", ms.group(3).lower())
            if k not in seen_edge_keys:
                seen_edge_keys.add(k)
                edges.append((person, "WORKS_ON", ms.group(3)))

    # Multi-object: "Orion uses Redis and Kafka"
    mo = re.search(
        r"\b([A-Z][a-zA-Z]+)\s+uses?\s+([A-Z][a-zA-Z]+)\s+and\s+([A-Z][a-zA-Z]+)",
        content,
    )
    if mo:
        for tech in (mo.group(2), mo.group(3)):
            k = (mo.group(1).lower(), "USES", tech.lower())
            if k not in seen_edge_keys:
                seen_edge_keys.add(k)
                edges.append((mo.group(1), "USES", tech))

    # Colleague
    mc = re.search(r"\b([A-Z][a-z]+)\s+is my colleague", content)
    if mc:
        k = ("user", "COLLEAGUE_OF", mc.group(1).lower())
        if k not in seen_edge_keys:
            seen_edge_keys.add(k)
            edges.append(("user", "COLLEAGUE_OF", mc.group(1)))

    # Single-subject/object patterns
    for pattern, sg, rel, tg, _ in _EDGE_PATTERNS:
        if sg is None:
            continue   # multi-subject, already handled
        m = re.search(pattern, content)
        if m and sg and tg:
            s, t = m.group(sg), m.group(tg)
            k = (s.lower(), rel, t.lower())
            if k not in seen_edge_keys:
                seen_edge_keys.add(k)
                edges.append((s, rel, t))

    return facts, edges


def make_extraction_fn(extractor_fn) -> ExtractorFn:
    """Wrap a two-arg extractor into the ExtractorFn signature."""
    def fn(role: str, content: str) -> tuple[list[tuple], list[tuple]]:
        return extractor_fn(role, content)
    return fn


# ---------------------------------------------------------------------------
# Mock summarizer
# ---------------------------------------------------------------------------

def mock_summarizer(turns: list[dict]) -> str:
    user_msgs = [t["content"] for t in turns if t["role"] == "user"]
    if not user_msgs:
        return "Empty session."
    first = user_msgs[0][:80].rstrip(".,")
    m = re.search(r"\bmy name is\s+([A-Z][a-z]+)", " ".join(user_msgs), re.IGNORECASE)
    name_note = f" User's name: {m.group(1)}." if m else ""
    return f"User said: '{first}'.{name_note} Session had {len(user_msgs)} user turn(s)."


# ---------------------------------------------------------------------------
# Real summarizer
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT = """\
Summarise this conversation session in 1-2 sentences.
Focus on: who the user is, what they discussed, key facts shared.

{transcript}

Return only the summary."""


def make_real_summarizer(api_key: str) -> SummarizerFn:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def summarizer(turns: list[dict]) -> str:
        transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": SUMMARIZE_PROMPT.format(transcript=transcript)}],
        )
        return r.content[0].text.strip()

    return summarizer


# ---------------------------------------------------------------------------
# Chat backends
# ---------------------------------------------------------------------------

def mock_chat(messages: list[dict], system: str) -> str:
    last = messages[-1]["content"].lower() if messages else ""
    ctx = system.lower()

    for q, attr in [
        ("name", "name"), ("project", "name"), ("work at", "employer"),
        ("employer", "employer"), ("role", "role"), ("live", "location"),
    ]:
        if q in last:
            m = re.search(rf"(?:user|project):\n.*?{attr}:\s+([\w][\w\s,&]+?)(?:\s+\[|\n)", ctx, re.DOTALL)
            if m:
                return f"Based on my memory, your {attr} is: {m.group(1).strip()}."

    if any(w in last for w in ["remember", "last time", "previous", "before"]):
        if "previous sessions" in ctx:
            ep_m = re.search(r"\[(\d{4}-\d{2}-\d{2})\]\s+(.+)", ctx)
            if ep_m:
                return (f"Yes! In our previous session ({ep_m.group(1)}) I noted: "
                        f"\"{ep_m.group(2)[:100]}\"")
        return "This appears to be our first conversation — no prior sessions found."

    if "use" in last or "technolog" in last:
        techs = re.findall(r"--\[uses\]-->\s+(\w+)", ctx)
        if techs:
            return f"Based on my knowledge graph: {', '.join(t.capitalize() for t in techs)}."

    layer_counts = {
        "episodic": ctx.count("previous sessions"),
        "structured": ctx.count("facts:"),
        "semantic": ctx.count("relevant past"),
    }
    active = [k for k, v in layer_counts.items() if v > 0]
    return (
        f"[Mock] Active memory layers: {', '.join(active) or 'working only'}. "
        f"Ask me about your name, project, employer, role, or what tools you use."
    )


def real_chat(messages: list[dict], system: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=messages,
    )
    return r.content[0].text


# ---------------------------------------------------------------------------
# Demo loop
# ---------------------------------------------------------------------------

def run_demo(use_mock: bool, api_key: str | None) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Layered Memory Demo [{mode}] ===")
    print("Layers: [1] working  [2] entity+KG  [3] episodic  [4] semantic")
    print("Commands: prompt, stats, endsession, newsession, quit\n")

    embedder: Embedder = MockEmbedder() if use_mock else RealEmbedder()
    summarizer: SummarizerFn = (
        mock_summarizer if use_mock else make_real_summarizer(api_key)
    )
    extractor: ExtractorFn = make_extraction_fn(mock_extractor)

    memory = LayeredMemory(
        extractor=extractor,
        embedder=embedder,
        summarizer=summarizer,
        budget=LayerBudget(episodic_chars=600, structured_chars=800, semantic_chars=500),
    )
    memory.start_session()
    print(f"Session started.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() == "quit":
            break

        if user_input.lower() == "prompt":
            sp = memory.get_system_prompt()
            print(f"\n--- System Prompt ({len(sp)} chars) ---")
            print(sp)
            print()
            continue

        if user_input.lower() == "stats":
            s = memory.stats()
            print(
                f"[turn={s['turn']} | "
                f"working={s['working_turns']} | "
                f"facts={s['facts']} | "
                f"kg_edges={s['kg_edges']} | "
                f"episodes={s['episodes']} | "
                f"semantic_idx={s['semantic_idx']}]\n"
            )
            continue

        if user_input.lower() == "endsession":
            ep = memory.end_session()
            if ep:
                print(f"Session saved: [{ep.started_at[:10]}] {ep.summary}\n")
            else:
                print("No turns to save.\n")
            continue

        if user_input.lower() == "newsession":
            memory.start_session()
            print("New session started. Episodic layer retains past sessions.\n")
            continue

        result = memory.add_user_message(user_input)
        system = memory.get_system_prompt(query=user_input)

        if use_mock:
            reply = mock_chat(memory.messages(), system)
        else:
            reply = real_chat(memory.messages(), system, api_key)

        memory.add_assistant_message(reply)
        print(f"Assistant: {reply}")

        if result["facts"] or result["edges"]:
            parts = []
            if result["facts"]:
                parts.append(
                    f"+{len(result['facts'])} fact(s): "
                    + ", ".join(f"{e}.{a}={v!r}" for e, a, v, _ in result["facts"])
                )
            if result["edges"]:
                parts.append(
                    f"+{len(result['edges'])} edge(s): "
                    + ", ".join(f"{s}--[{r}]-->{t}" for s, r, t in result["edges"])
                )
            print(f"  [L2] {' | '.join(parts)}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.real and not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock, api_key=api_key)
