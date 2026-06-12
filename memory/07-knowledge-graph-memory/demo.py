"""
Demo: Knowledge Graph Memory
Usage:
    uv run python memory/07-knowledge-graph-memory/demo.py --mock
    uv run python memory/07-knowledge-graph-memory/demo.py --real
    uv run python memory/07-knowledge-graph-memory/demo.py --mock --persist  # saves graph to disk

Key behaviour to observe:
  - Tell the assistant about people, projects, and their relationships.
  - Ask questions that require following edges (e.g. "What does Bob work on?").
  - Type 'graph' to inspect the full knowledge graph.
  - The mock extractor uses regex patterns; the real extractor uses Claude.

Suggested conversation:
    1. "Alice and Bob both work on project Orion."
    2. "Orion uses Redis and Kafka."
    3. "Bob reports to Carol."
    4. "Carol manages the infrastructure team."
    5. graph  ← see the full graph
    6. "What does Alice work on?"
    7. "Who does Bob report to?"
    8. "What technologies does Orion use?"
    9. "What team does Carol manage?"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from experiment import KGMemory, KnowledgeGraph, RelationExtractorFn, Triple

DEFAULT_GRAPH_PATH = Path("/tmp/ai-hardway-kg/graph.json")


# ---------------------------------------------------------------------------
# Mock extractor — regex-based triple extraction, no LLM call
# ---------------------------------------------------------------------------

# Each pattern is (regex, subject_group, subject_type, relation, object_group, object_type)
# subject_group / object_group are indices into match.group()
#
# Relation names follow the convention used in knowledge graph literature:
# ALL_CAPS verbs, e.g. WORKS_ON, USES, REPORTS_TO.

_TRIPLE_PATTERNS: list[tuple] = [
    # "Alice and Bob both work on project Orion"   (two subjects, one object)
    # Handled specially below via _extract_multi_subject

    # "Alice works on project Orion" / "Alice works on Orion"
    (r"\b([A-Z][a-z]+)\s+works?\s+on\s+(?:project\s+)?([A-Z][a-zA-Z]+)",
     1, "person", "WORKS_ON", 2, "project"),

    # "Alice and Bob work on Orion"  → separate pattern to anchor multi-subject
    # (handled via _extract_multi_subject)

    # "Orion uses Redis" / "Orion uses Redis and Kafka"
    # (multi-object handled below)
    (r"\b([A-Z][a-zA-Z]+)\s+uses?\s+([A-Z][a-zA-Z]+)",
     1, "project", "USES", 2, "technology"),

    # "Bob reports to Carol"
    (r"\b([A-Z][a-z]+)\s+reports?\s+to\s+([A-Z][a-z]+)",
     1, "person", "REPORTS_TO", 2, "person"),

    # "Carol manages the infrastructure team" / "Carol manages the X team"
    (r"\b([A-Z][a-z]+)\s+manages?\s+the\s+([\w\s]+?)\s+team",
     1, "person", "MANAGES", 2, "team"),

    # "Alice is the lead of Orion" / "Alice leads Orion"
    (r"\b([A-Z][a-z]+)\s+(?:is the lead of|leads?)\s+(?:project\s+)?([A-Z][a-zA-Z]+)",
     1, "person", "LEADS", 2, "project"),

    # "Alice works with Bob"
    (r"\b([A-Z][a-z]+)\s+works?\s+with\s+([A-Z][a-z]+)",
     1, "person", "WORKS_WITH", 2, "person"),

    # "Orion is a distributed cache" / "Orion is a real-time pipeline"
    (r"\b([A-Z][a-zA-Z]+)\s+is an?\s+([\w][\w\s\-]+?)(?:\.|,|$)",
     1, "project", "IS_A", 2, "concept"),

    # "Alice is a software engineer" / "Bob is a backend developer"
    (r"\b([A-Z][a-z]+)\s+is an?\s+([\w]+(?:\s+[\w]+){0,2})\s+(?:engineer|developer|architect|designer|manager)",
     1, "person", "HAS_ROLE", 2, "role"),
]


def _extract_multi_subject(text: str, triples: list[Triple]) -> None:
    """
    Handle 'Alice and Bob both work on Orion' → two WORKS_ON triples.
    """
    m = re.search(
        r"\b([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\s+(?:both\s+)?work(?:s)?\s+on\s+(?:project\s+)?([A-Z][a-zA-Z]+)",
        text,
    )
    if m:
        for person in (m.group(1), m.group(2)):
            triples.append(Triple(
                subject=person, subject_type="person",
                relation="WORKS_ON",
                object=m.group(3), object_type="project",
            ))


def _extract_multi_object(text: str, triples: list[Triple]) -> None:
    """
    Handle 'Orion uses Redis and Kafka' → two USES triples.
    """
    m = re.search(
        r"\b([A-Z][a-zA-Z]+)\s+uses?\s+([A-Z][a-zA-Z]+)\s+and\s+([A-Z][a-zA-Z]+)",
        text,
    )
    if m:
        for tech in (m.group(2), m.group(3)):
            triples.append(Triple(
                subject=m.group(1), subject_type="project",
                relation="USES",
                object=tech, object_type="technology",
            ))


def mock_extractor(role: str, content: str) -> list[Triple]:
    """Extract triples using regex. Fast, deterministic, no LLM cost."""
    if role != "user":
        return []

    triples: list[Triple] = []
    text = content.strip()

    # Multi-subject / multi-object patterns first
    _extract_multi_subject(text, triples)
    _extract_multi_object(text, triples)

    # Collect keys already found to avoid duplicates from single-subject patterns
    found_keys: set[tuple] = {(t.subject.lower(), t.relation, t.object.lower()) for t in triples}

    for pattern, sg, st, rel, og, ot in _TRIPLE_PATTERNS:
        for m in re.finditer(pattern, text):
            subj = m.group(sg).strip()
            obj = m.group(og).strip().rstrip(".,!")
            if not subj or not obj or len(obj) > 60:
                continue
            key = (subj.lower(), rel, obj.lower())
            if key not in found_keys:
                found_keys.add(key)
                triples.append(Triple(
                    subject=subj, subject_type=st,
                    relation=rel,
                    object=obj, object_type=ot,
                ))

    return triples


# ---------------------------------------------------------------------------
# Real extractor — Claude returns structured JSON
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """\
Extract knowledge graph triples from the following message.
Return a JSON array of objects, each with these fields:
  subject      (string)  — the entity performing or being described
  subject_type (string)  — one of: person, project, technology, team, concept, organization
  relation     (string)  — ALL_CAPS verb phrase, e.g. WORKS_ON, USES, REPORTS_TO, MANAGES, LEADS, IS_A, HAS_ROLE, WORKS_WITH
  object       (string)  — the entity the relation points to
  object_type  (string)  — same type vocabulary as subject_type

Only extract explicitly stated facts. Do not infer.
If there is nothing to extract, return [].

Message (user): {content}

Return only valid JSON, nothing else."""


def make_real_extractor(api_key: str) -> RelationExtractorFn:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def extractor(role: str, content: str) -> list[Triple]:
        if role != "user":
            return []
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": EXTRACT_PROMPT.format(content=content),
            }],
        )
        raw = response.content[0].text.strip()
        try:
            items = json.loads(raw)
            triples = []
            for item in items:
                if all(k in item for k in ("subject", "subject_type", "relation", "object", "object_type")):
                    triples.append(Triple(
                        subject=item["subject"],
                        subject_type=item["subject_type"],
                        relation=item["relation"],
                        object=item["object"],
                        object_type=item["object_type"],
                    ))
            return triples
        except (json.JSONDecodeError, KeyError):
            return []

    return extractor


# ---------------------------------------------------------------------------
# Chat backends
# ---------------------------------------------------------------------------

def mock_chat(messages: list[dict], system: str) -> str:
    """
    Answer questions by searching the injected graph section.

    The formatted graph looks like (lowercased in ctx):
        alice (person):
          --[works_on]--> orion (project)
          --[works_with]--> bob (person)
        orion (project):
          --[uses]--> redis (technology)

    We search for the entity header then collect its outgoing edges,
    or search for edges pointing to the entity for inbound queries.
    """
    last = messages[-1]["content"].lower() if messages else ""
    ctx = system.lower()

    def outgoing(entity: str, relation: str) -> list[str]:
        """Find all targets of entity --[relation]--> ?"""
        # Find the entity's block, then collect lines with the given relation
        targets = []
        in_block = False
        for line in ctx.splitlines():
            if re.match(rf"^{re.escape(entity)}\s*\(", line):
                in_block = True
                continue
            if in_block:
                if line and not line.startswith(" "):
                    break   # new entity block
                m = re.search(rf"--\[{re.escape(relation)}\]-->\s+(\w[\w\s]*?)\s+\(", line)
                if m:
                    targets.append(m.group(1).strip())
        return targets

    if "what does" in last and "work on" in last:
        m = re.search(r"what does\s+(\w+)\s+work on", last)
        if m:
            person = m.group(1).lower()
            projects = outgoing(person, "works_on")
            if projects:
                return f"Based on my knowledge graph, {person.capitalize()} works on: {', '.join(p.capitalize() for p in projects)}."
            return f"I don't have any WORKS_ON edges for {person.capitalize()} yet."

    if "who does" in last and "report to" in last:
        m = re.search(r"who does\s+(\w+)\s+report to", last)
        if m:
            person = m.group(1).lower()
            managers = outgoing(person, "reports_to")
            if managers:
                return f"{person.capitalize()} reports to {managers[0].capitalize()}."
            return f"I don't have a REPORTS_TO edge for {person.capitalize()}."

    if "what technologies" in last or ("what" in last and "use" in last):
        m = re.search(r"what (?:technologies )?does\s+(\w+)\s+use", last)
        if m:
            proj = m.group(1).lower()
            techs = outgoing(proj, "uses")
            if techs:
                return f"{proj.capitalize()} uses: {', '.join(t.capitalize() for t in techs)}."
            return f"I don't have USES edges for {proj.capitalize()}."

    if "what team" in last or ("team" in last and "manage" in last):
        m = re.search(r"what team does\s+(\w+)\s+manage", last)
        if m:
            person = m.group(1).lower()
            teams = outgoing(person, "manages")
            if teams:
                return f"{person.capitalize()} manages the {teams[0]} team."
        # fallback: scan for any manages edge
        fm = re.search(r"(\w+) \(person\):\n(?:\s+--\[.*\].*\n)*\s+--\[manages\]-->\s+(\w[\w\s]*?)\s+\(", ctx)
        if fm:
            return f"{fm.group(1).capitalize()} manages the {fm.group(2).strip()} team."
        return "I don't have any MANAGES edges in the graph yet."

    node_count = ctx.count("--[")
    return (
        f"[Mock] Knowledge graph has {node_count} edge(s) in the injected subgraph. "
        f"Ask me about who works on what, who reports to whom, or what technologies a project uses."
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

def run_demo(use_mock: bool, persist: bool, graph_path: Path, api_key: str | None) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== Knowledge Graph Memory Demo [{mode}] ===")
    if persist:
        print(f"Graph persistence: {graph_path}")
    print("Commands: 'graph' to inspect the full graph, 'stats', 'quit' to exit.\n")

    graph = KnowledgeGraph.load(graph_path) if persist else KnowledgeGraph()
    if persist and graph.node_count() > 0:
        print(f"Loaded graph: {graph.node_count()} nodes, {graph.edge_count()} edges.\n")

    extractor = mock_extractor if use_mock else make_real_extractor(api_key)
    memory = KGMemory(extractor=extractor, graph=graph)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() == "quit":
            break

        if user_input.lower() == "graph":
            if memory.graph.edge_count() == 0:
                print("\n(no relationships in graph yet)\n")
            else:
                print(f"\n--- Knowledge Graph ({memory.graph.node_count()} nodes, {memory.graph.edge_count()} edges) ---")
                print(memory.graph.format_subgraph())
                print()
            continue

        if user_input.lower() == "stats":
            print(
                f"[nodes={memory.graph.node_count()} | "
                f"edges={memory.graph.edge_count()} | "
                f"extractions={memory.extractions} | "
                f"buffer={memory.buffer_length}]\n"
            )
            continue

        triples = memory.add_user_message(user_input)
        system = memory.get_system_prompt()

        if use_mock:
            reply = mock_chat(memory.get_messages(), system)
        else:
            reply = real_chat(memory.get_messages(), system, api_key)

        memory.add_assistant_message(reply)
        print(f"Assistant: {reply}")
        if triples:
            print(f"  [+{len(triples)} triple(s): {', '.join(t.subject + ' --[' + t.relation + ']--> ' + t.object for t in triples)}]")
        print()

    if persist and memory.graph.edge_count() > 0:
        memory.graph.save(graph_path)
        print(f"Graph saved to {graph_path} ({memory.graph.node_count()} nodes, {memory.graph.edge_count()} edges).\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    parser.add_argument(
        "--persist",
        action="store_true",
        help=f"Save/load graph to disk (default path: {DEFAULT_GRAPH_PATH})",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help=f"Path for persistent graph JSON (default: {DEFAULT_GRAPH_PATH})",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.real and not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(
        use_mock=args.mock,
        persist=args.persist,
        graph_path=args.graph_path,
        api_key=api_key,
    )
