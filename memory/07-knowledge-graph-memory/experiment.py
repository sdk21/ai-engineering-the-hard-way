"""
Knowledge Graph Memory
----------------------
Entity memory stores {entity → {attribute → value}} — a flat per-entity
fact bag. It cannot represent relationships *between* entities:

    "Alice works with Bob on project Orion."
    "Bob reports to Carol."
    "Orion uses Redis and Kafka."

A knowledge graph stores these as labeled edges:

    Alice  --[WORKS_WITH]-->  Bob
    Alice  --[WORKS_ON]-->    Orion
    Bob    --[REPORTS_TO]-->  Carol
    Orion  --[USES]-->        Redis
    Orion  --[USES]-->        Kafka

Each node has a type and attribute bag. Each edge has a relation label.
Context injection traverses the graph from recently mentioned entities,
pulling in their direct neighbours and the edges connecting them.

Architecture:

    Each turn
        ↓
    RelationExtractorFn(role, content) → list[Triple]
        ↓
    Merge triples into KnowledgeGraph (upsert nodes, upsert edges)
        ↓
    get_system_prompt(focus_entities=None)
        → find neighbourhood of focus entities
        → format as readable triple list
        → inject into system prompt

Storage is in-memory (a pair of dicts: nodes, edges) with optional
JSON serialisation so the graph can be persisted across sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class KGNode:
    id: str                            # canonical name, lower-case
    node_type: str                     # "person", "project", "technology", …
    attributes: dict[str, str] = field(default_factory=dict)

    def display(self) -> str:
        attrs = ", ".join(f"{k}={v}" for k, v in sorted(self.attributes.items()))
        return f"{self.id} ({self.node_type})" + (f" [{attrs}]" if attrs else "")


@dataclass
class KGEdge:
    source: str        # node id
    target: str        # node id
    relation: str      # e.g. "WORKS_WITH", "USES", "REPORTS_TO"
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.relation, self.target)

    def display(self) -> str:
        return f"{self.source} --[{self.relation}]--> {self.target}"


# A triple is the atomic unit extracted from text:
#   (subject_id, subject_type, relation, object_id, object_type)
@dataclass
class Triple:
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    subject_attrs: dict[str, str] = field(default_factory=dict)
    object_attrs: dict[str, str] = field(default_factory=dict)


RelationExtractorFn = Callable[[str, str], list[Triple]]


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    In-memory knowledge graph backed by two dicts.

    nodes: {node_id → KGNode}
    edges: {(source, relation, target) → KGEdge}
    adjacency: {node_id → set of edge keys}   (both directions)
    """

    def __init__(self) -> None:
        self.nodes: dict[str, KGNode] = {}
        self.edges: dict[tuple[str, str, str], KGEdge] = {}
        self._adj: dict[str, set[tuple[str, str, str]]] = {}  # node_id → edge keys

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_triple(self, triple: Triple) -> None:
        """Upsert both nodes and the edge from a triple."""
        self._upsert_node(triple.subject, triple.subject_type, triple.subject_attrs)
        self._upsert_node(triple.object, triple.object_type, triple.object_attrs)
        self._upsert_edge(triple.subject, triple.relation, triple.object)

    def _upsert_node(self, node_id: str, node_type: str, attrs: dict[str, str]) -> None:
        node_id = node_id.lower()
        if node_id not in self.nodes:
            self.nodes[node_id] = KGNode(id=node_id, node_type=node_type)
        node = self.nodes[node_id]
        node.node_type = node_type   # last-write-wins on type
        node.attributes.update(attrs)

    def _upsert_edge(self, source: str, relation: str, target: str) -> None:
        source, target = source.lower(), target.lower()
        key = (source, relation, target)
        if key not in self.edges:
            self.edges[key] = KGEdge(source=source, relation=relation, target=target)
        # Record adjacency from both ends
        self._adj.setdefault(source, set()).add(key)
        self._adj.setdefault(target, set()).add(key)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def neighbours(self, node_id: str, depth: int = 1) -> tuple[set[str], list[KGEdge]]:
        """
        BFS from node_id to the given depth.
        Returns (visited node ids, list of traversed edges).
        """
        node_id = node_id.lower()
        visited: set[str] = {node_id}
        frontier: set[str] = {node_id}
        edges_seen: list[KGEdge] = []

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                for key in self._adj.get(nid, []):
                    edge = self.edges[key]
                    edges_seen.append(edge)
                    for end in (edge.source, edge.target):
                        if end not in visited:
                            visited.add(end)
                            next_frontier.add(end)
            frontier = next_frontier

        # Deduplicate edges
        seen_keys: set[tuple] = set()
        deduped: list[KGEdge] = []
        for e in edges_seen:
            if e.key not in seen_keys:
                seen_keys.add(e.key)
                deduped.append(e)

        return visited, deduped

    def all_edges(self) -> list[KGEdge]:
        return list(self.edges.values())

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def format_subgraph(
        self,
        node_ids: set[str] | None = None,
        edges: list[KGEdge] | None = None,
    ) -> str:
        """
        Render the subgraph (or full graph if node_ids/edges are None)
        as a human-readable triple list for injection into the system prompt.
        """
        if edges is None:
            edges = self.all_edges()
        if node_ids is None:
            node_ids = set(self.nodes.keys())

        if not edges:
            return "(no relationships recorded yet)"

        lines: list[str] = []
        # Group edges by subject for readability
        by_subject: dict[str, list[KGEdge]] = {}
        for e in edges:
            by_subject.setdefault(e.source, []).append(e)

        for subject in sorted(by_subject):
            node = self.nodes.get(subject)
            node_label = f"{subject.capitalize()} ({node.node_type})" if node else subject.capitalize()
            lines.append(f"{node_label}:")
            for e in sorted(by_subject[subject], key=lambda x: x.relation):
                target_node = self.nodes.get(e.target)
                target_label = (
                    f"{e.target.capitalize()} ({target_node.node_type})"
                    if target_node else e.target.capitalize()
                )
                lines.append(f"  --[{e.relation}]--> {target_label}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": {
                nid: {"node_type": n.node_type, "attributes": n.attributes}
                for nid, n in self.nodes.items()
            },
            "edges": [
                {"source": e.source, "relation": e.relation, "target": e.target,
                 "attributes": e.attributes}
                for e in self.edges.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        kg = cls()
        for nid, ndata in data.get("nodes", {}).items():
            kg.nodes[nid] = KGNode(id=nid, node_type=ndata["node_type"],
                                   attributes=ndata.get("attributes", {}))
        for edata in data.get("edges", []):
            kg._upsert_edge(edata["source"], edata["relation"], edata["target"])
        return kg

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text()))


# ---------------------------------------------------------------------------
# KG Memory
# ---------------------------------------------------------------------------

class KGMemory:
    """
    Manages a KnowledgeGraph alongside a bounded recent-message buffer.

    Every user turn is passed to the RelationExtractorFn. Extracted triples
    are merged into the graph. The system prompt is augmented with the
    neighbourhood of recently mentioned entities.

    Args:
        extractor:       RelationExtractorFn — extracts triples from a turn
        graph:           Pre-existing KnowledgeGraph (or a fresh one)
        system_prompt:   Base system instructions
        buffer_size:     Max recent turns kept for the LLM context window
        neighbourhood_depth: BFS depth when building focused subgraph (1 or 2)
        max_graph_lines: Cap on graph lines injected into the system prompt
    """

    def __init__(
        self,
        extractor: RelationExtractorFn,
        graph: KnowledgeGraph | None = None,
        system_prompt: str = "You are a helpful assistant.",
        buffer_size: int = 8,
        neighbourhood_depth: int = 1,
        max_graph_lines: int = 40,
    ) -> None:
        self.extractor = extractor
        self.graph = graph or KnowledgeGraph()
        self.system_prompt = system_prompt
        self.buffer_size = buffer_size
        self.neighbourhood_depth = neighbourhood_depth
        self.max_graph_lines = max_graph_lines

        self._buffer: list[dict[str, str]] = []
        self._recent_entities: list[str] = []   # entities mentioned in recent turns
        self._extractions: int = 0

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> list[Triple]:
        triples = self.extractor("user", content)
        for t in triples:
            self.graph.add_triple(t)
            self._recent_entities.append(t.subject.lower())
            self._recent_entities.append(t.object.lower())
        self._extractions += len(triples)
        self._push({"role": "user", "content": content})
        # Keep the N most recently mentioned distinct entities (most recent first)
        seen: set[str] = set()
        deduped: list[str] = []
        for e in reversed(self._recent_entities):
            if e not in seen:
                seen.add(e)
                deduped.append(e)
        # deduped is already most-recent-first; trim and store
        self._recent_entities = deduped[:10]
        return triples

    def add_assistant_message(self, content: str) -> None:
        self._push({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._buffer)

    def get_system_prompt(self) -> str:
        """
        Build system prompt with a focused subgraph neighbourhood.

        If there are recent entities, retrieve their neighbourhood from the
        graph and inject it. Otherwise inject the full graph (up to the line
        cap).
        """
        graph_section = self._build_graph_section()
        if not graph_section:
            return self.system_prompt
        return f"{self.system_prompt}\n\n## Known relationships\n{graph_section}"

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def extractions(self) -> int:
        return self._extractions

    @property
    def buffer_length(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _push(self, msg: dict[str, str]) -> None:
        self._buffer.append(msg)
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)

    def _build_graph_section(self) -> str:
        if self.graph.edge_count() == 0:
            return ""

        if self._recent_entities:
            all_nodes: set[str] = set()
            all_edges: list[KGEdge] = []
            seen_edge_keys: set[tuple] = set()
            for eid in self._recent_entities[:5]:
                nodes, edges = self.graph.neighbours(eid, depth=self.neighbourhood_depth)
                all_nodes |= nodes
                for e in edges:
                    if e.key not in seen_edge_keys:
                        seen_edge_keys.add(e.key)
                        all_edges.append(e)
            text = self.graph.format_subgraph(node_ids=all_nodes, edges=all_edges)
        else:
            text = self.graph.format_subgraph()

        # Cap to max_graph_lines to avoid context bloat
        lines = text.splitlines()
        if len(lines) > self.max_graph_lines:
            lines = lines[: self.max_graph_lines] + ["  … (graph truncated)"]
        return "\n".join(lines)
