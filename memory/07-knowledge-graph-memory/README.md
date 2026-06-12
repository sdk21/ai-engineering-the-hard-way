# Lesson: Knowledge Graph Memory

**Vertical:** Memory | **Difficulty:** Intermediate–Advanced | **Status:** ✅ Ready

---

## Table of Contents

1. [The Limits of Flat Fact Storage](#1-the-limits-of-flat-fact-storage)
2. [What Is a Knowledge Graph?](#2-what-is-a-knowledge-graph)
3. [Architecture](#3-architecture)
4. [The Data Model: Nodes, Edges, Triples](#4-the-data-model-nodes-edges-triples)
5. [Triple Extraction](#5-triple-extraction)
6. [Graph Storage: Adjacency Lists](#6-graph-storage-adjacency-lists)
7. [Focused Subgraph Injection](#7-focused-subgraph-injection)
8. [BFS Neighbourhood Traversal](#8-bfs-neighbourhood-traversal)
9. [Persistence](#9-persistence)
10. [Failure Modes](#10-failure-modes)
11. [Key Principles](#11-key-principles)
12. [In the Real World](#12-in-the-real-world)
13. [Running the Experiment](#13-running-the-experiment)

---

## 1. The Limits of Flat Fact Storage

Entity memory (Experiment 05) stores a flat bag of facts per entity:

```
user:
  name: Alice
  role: software engineer
  location: Tokyo

project:
  name: Orion
  type: distributed cache
```

This is useful but fundamentally limited. It cannot represent **relationships between entities**:

- "Alice works with Bob on project Orion."
- "Bob reports to Carol."
- "Orion uses Redis and Kafka."
- "Carol manages the infrastructure team."

With flat entity memory, you know Alice exists and Orion exists — but you cannot answer "Who else works on Orion?" or "What does Alice's manager manage?" without those connections being explicit.

Knowledge graph memory solves this by storing **labeled, directed edges** between entities. The graph is a first-class data structure, not a side-effect of text storage.

---

## 2. What Is a Knowledge Graph?

A knowledge graph is a directed, labeled multigraph where:

- **Nodes** represent entities (people, projects, technologies, teams, concepts)
- **Edges** represent typed relationships between entities

```
Alice  --[WORKS_ON]-->    Orion
Alice  --[WORKS_WITH]-->  Bob
Bob    --[WORKS_ON]-->    Orion
Bob    --[REPORTS_TO]-->  Carol
Carol  --[MANAGES]-->     infrastructure team
Orion  --[USES]-->        Redis
Orion  --[USES]-->        Kafka
```

This structure enables **graph traversal queries**:

- "What does Alice work on?" → Follow `Alice --[WORKS_ON]-->` edges → `Orion`
- "Who else works on Orion?" → Find all nodes with `--[WORKS_ON]--> Orion`
- "What does Alice's team use?" → `Alice → Orion → Redis, Kafka`

These multi-hop queries are impossible with flat entity memory but trivial with a graph.

---

## 3. Architecture

```
Each user turn
    ↓
RelationExtractorFn(role, content)
    → list[Triple]  e.g. ("Alice", person, WORKS_ON, "Orion", project)
    ↓
KnowledgeGraph.add_triple(triple)
    → upsert KGNode for subject
    → upsert KGNode for object
    → upsert KGEdge (source, relation, target)
    → update adjacency index
    ↓
get_system_prompt()
    → collect recently-mentioned entities
    → BFS neighbourhood from those entities (depth=1 or 2)
    → format subgraph as triple list
    → inject under "## Known relationships"
```

Two extractor modes, same interface:
- `mock_extractor` — regex patterns, no LLM cost
- `make_real_extractor(api_key)` — Claude returns structured JSON array of triples

---

## 4. The Data Model: Nodes, Edges, Triples

### KGNode

```python
@dataclass
class KGNode:
    id: str            # canonical lower-case identifier
    node_type: str     # "person", "project", "technology", "team", "concept"
    attributes: dict[str, str]
```

Nodes carry a type to aid prompt formatting and downstream reasoning. The type is last-write-wins — the extractor's most recent type wins if there is disagreement across turns.

### KGEdge

```python
@dataclass
class KGEdge:
    source: str        # node id (lower-case)
    target: str        # node id (lower-case)
    relation: str      # ALL_CAPS verb, e.g. "WORKS_ON", "USES", "REPORTS_TO"
```

Edges are keyed by `(source, relation, target)` — a triple forms a unique edge. Duplicate triples from the same or different turns are silently deduplicated.

### Triple

The atomic unit extracted from text:

```python
@dataclass
class Triple:
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    subject_attrs: dict[str, str]   # optional extra attributes for the subject node
    object_attrs: dict[str, str]    # optional extra attributes for the object node
```

A single sentence can yield multiple triples:

> "Alice and Bob both work on Orion, which uses Redis and Kafka."

```
Triple(Alice, person, WORKS_ON, Orion, project)
Triple(Bob,   person, WORKS_ON, Orion, project)
Triple(Orion, project, USES, Redis, technology)
Triple(Orion, project, USES, Kafka, technology)
```

---

## 5. Triple Extraction

### Mock Extractor

Regex patterns with a known vocabulary of relation types:

```python
_TRIPLE_PATTERNS = [
    (r"\b([A-Z][a-z]+)\s+works?\s+on\s+(?:project\s+)?([A-Z][a-zA-Z]+)",
     1, "person", "WORKS_ON", 2, "project"),

    (r"\b([A-Z][a-zA-Z]+)\s+uses?\s+([A-Z][a-zA-Z]+)",
     1, "project", "USES", 2, "technology"),

    (r"\b([A-Z][a-z]+)\s+reports?\s+to\s+([A-Z][a-z]+)",
     1, "person", "REPORTS_TO", 2, "person"),
    ...
]
```

Multi-subject and multi-object patterns are handled separately:
- "Alice **and Bob** work on Orion" → two WORKS_ON triples
- "Orion uses **Redis and Kafka**" → two USES triples

**Limitations of regex extraction:** brittle, limited relation vocabulary, misses paraphrases. Good for demos and testing; not production-quality.

### Real Extractor (Claude)

```python
EXTRACT_PROMPT = """\
Extract knowledge graph triples from the following message.
Return a JSON array of objects with: subject, subject_type, relation, object, object_type.
...
"""
```

Claude returns:
```json
[
  {"subject": "Alice", "subject_type": "person", "relation": "WORKS_ON",
   "object": "Orion", "object_type": "project"},
  {"subject": "Bob", "subject_type": "person", "relation": "WORKS_ON",
   "object": "Orion", "object_type": "project"}
]
```

The real extractor handles:
- Paraphrases: "Alice is the lead engineer on Orion" → LEADS
- Implicit facts: "Alice, our backend lead" → HAS_ROLE
- Nested entities: "Bob joined Alice's team" → WORKS_WITH
- Synonyms: "handles", "is in charge of", "oversees" → MANAGES

---

## 6. Graph Storage: Adjacency Lists

The graph is stored as two dicts plus an adjacency index:

```python
nodes: dict[str, KGNode]                     # node_id → KGNode
edges: dict[tuple[str,str,str], KGEdge]      # (src, rel, tgt) → KGEdge
_adj:  dict[str, set[tuple[str,str,str]]]    # node_id → set of edge keys
```

The adjacency index is bidirectional — both source and target node point to the edge. This makes neighbourhood lookup O(degree) regardless of total graph size.

```
_adj["alice"] = {("alice","WORKS_ON","orion"), ("alice","WORKS_WITH","bob")}
_adj["orion"]  = {("alice","WORKS_ON","orion"), ("bob","WORKS_ON","orion"),
                  ("orion","USES","redis"), ("orion","USES","kafka")}
```

An edge key `(source, relation, target)` uniquely identifies the edge. Inserting the same triple twice is a no-op — upsert semantics prevent duplicate edges.

---

## 7. Focused Subgraph Injection

Injecting the entire graph into every prompt is wasteful and eventually exceeds the context window. Instead, we inject a **focused neighbourhood** of the entities mentioned in recent turns.

```python
def get_system_prompt(self) -> str:
    ...
    # Collect up to 5 recently mentioned entities
    for eid in self._recent_entities[:5]:
        nodes, edges = self.graph.neighbours(eid, depth=self.neighbourhood_depth)
        ...
    # Format as triple list under "## Known relationships"
```

Example: if the current turn mentions "Alice", the injected subgraph includes:
- All edges incident to Alice
- All edges incident to Alice's direct neighbours (at depth=2)

This typically yields 5–15 edges for a conversation-sized graph — a handful of lines, not the entire graph.

The `max_graph_lines` cap (default: 40) prevents the graph section from dominating the context window as the graph grows.

---

## 8. BFS Neighbourhood Traversal

```python
def neighbours(self, node_id: str, depth: int = 1) -> tuple[set[str], list[KGEdge]]:
    visited = {node_id}
    frontier = {node_id}

    for _ in range(depth):
        next_frontier = set()
        for nid in frontier:
            for edge_key in self._adj.get(nid, []):
                edge = self.edges[edge_key]
                for end in (edge.source, edge.target):
                    if end not in visited:
                        visited.add(end)
                        next_frontier.add(end)
        frontier = next_frontier

    return visited, deduped_edges
```

At `depth=1`: Alice's immediate neighbours (Bob, Orion). Cost: O(degree(Alice)).
At `depth=2`: Alice's neighbours and their neighbours. Cost: O(degree²).

For conversation-scale graphs (< 500 nodes), BFS is negligibly fast. For production graphs with millions of nodes, this is where you would swap in a real graph database.

---

## 9. Persistence

The graph serialises to a single JSON file:

```json
{
  "nodes": {
    "alice": {"node_type": "person", "attributes": {}},
    "orion": {"node_type": "project", "attributes": {}}
  },
  "edges": [
    {"source": "alice", "relation": "WORKS_ON", "target": "orion", "attributes": {}}
  ]
}
```

`KnowledgeGraph.save(path)` and `KnowledgeGraph.load(path)` handle serialisation. Use `--persist` in the demo to save the graph across runs.

For production, replace this with a graph database (Neo4j, Memgraph, Kuzu) — the same `KnowledgeGraph` interface can wrap any backend.

---

## 10. Failure Modes

**Entity disambiguation** — "Alice" in one turn and "Alice Smith" in another create two different nodes. Mitigation: entity normalisation (e.g. always lower-case, canonical name resolution), or extract canonical identifiers alongside names.

**Relation vocabulary drift** — The extractor may use LEADS in one turn and MANAGES in another for the same relationship. Mitigation: define a closed relation vocabulary in the extraction prompt and validate the relation field against it.

**Stale edges** — "Alice works on Orion" is asserted, then "Alice left Orion." The graph has no way to retract unless explicit negation is handled ("Alice no longer works on Orion" → delete WORKS_ON edge). Most systems ignore negation; a more rigorous implementation uses timestamped edges and recency weighting.

**Subgraph explosion** — At depth=2 on a dense graph, the neighbourhood can include hundreds of edges. Mitigation: `max_graph_lines` cap and `neighbourhood_depth=1` default.

**Extraction hallucination** — The LLM extractor may fabricate triples not present in the text ("Alice WORKS_WITH Carol" when only Alice and Bob were mentioned). Mitigation: extraction prompt with "only extract explicitly stated facts, do not infer"; structured output validation.

**Graph growth over long sessions** — Without pruning, the graph accumulates all facts ever stated, including outdated ones. Mitigation: edge timestamps + TTL, or explicit contradiction handling.

---

## 11. Key Principles

> **Principle 1 — Relationships are first-class data.**
> Flat fact stores cannot represent connections between entities. A graph with labeled edges makes relationships queryable, not just readable. The LLM can reason over injected graph structure that would be ambiguous or absent in a flat fact dump.

> **Principle 2 — Extract triples, not text.**
> The unit of storage is a (subject, relation, object) triple, not a sentence or paragraph. This makes the graph programmable: you can query by relation type, find all nodes of a type, or traverse multi-hop paths.

> **Principle 3 — Inject focused neighbourhoods, not the whole graph.**
> A growing graph must not grow the context window at the same rate. Neighbourhood-based injection bounds the cost at O(degree × depth) per session, independent of total graph size.

> **Principle 4 — The graph composes with other memory types.**
> Knowledge graph memory captures structure; episodic memory captures history; entity memory captures attributes. In a layered system, the graph stores *relationships*, episodic memory stores *when they were established*, and entity memory stores *properties of the nodes*. They are complementary layers.

---

## 12. In the Real World

**Google Knowledge Graph**
Google's Knowledge Graph (2012) underpins the "Knowledge Panel" on the right of search results. It stores billions of entities (people, places, organisations) as nodes and their relationships as labeled edges. When you search "Albert Einstein" the panel shows structured facts ("educated at ETH Zurich", "field: physics") extracted from the graph — the same triple injection pattern used in this experiment, but at planetary scale.

**Wikidata**
Wikidata is an openly editable knowledge graph with ~100 million statements in (subject, property, value) triple form — exactly the data model in this experiment. It underlies structured facts in Wikipedia infoboxes and powers the semantic search features of many knowledge management tools.

**Microsoft GraphRAG**
Microsoft's GraphRAG (2024) extends RAG by first extracting a knowledge graph from documents (entities + relationships), then using graph traversal to retrieve relevant context at query time. Instead of finding relevant *chunks*, it finds relevant *graph neighbourhoods*, enabling multi-hop reasoning over the document corpus — the neighbourhood traversal from this experiment applied to document retrieval.

**Neo4j + LLM integrations**
Neo4j's LLM integration stack (LangChain, LlamaIndex connectors) follows the exact pattern here: extract triples from user messages → store in Neo4j → query the graph at each turn → inject the subgraph into the LLM prompt. Neo4j's Cypher query language lets the injection step be a precise graph query rather than a BFS approximation.

**Mem0 — relationship memory**
Mem0's memory layer extracts not just entity attributes but also inter-entity relationships. The storage backend is a vector store for semantic search + a property graph for structured queries — episodic, entity, and graph memory combined in one system.

**Apple Intelligence — on-device personal graph**
Apple's on-device intelligence (iOS 18+) maintains a personal knowledge graph: your contacts, their relationships to each other, which emails they sent you, which events they are invited to, which apps they use. Siri queries this graph for context — "schedule a meeting with my manager's team" traverses person → REPORTS_TO → manager → MANAGES → team without any cloud call.

**Notion AI**
Notion's AI features treat a Notion workspace as a knowledge graph: pages are nodes, links between pages are edges, properties are node attributes. When you ask "what projects is Alice involved in?", Notion AI traverses the graph of pages linked to Alice's profile — the same neighbourhood traversal as this experiment, on a user-built graph rather than an extracted one.

---

## 13. Running the Experiment

```bash
# From the project root

# Mock mode — regex extraction, no API key needed
uv run python memory/07-knowledge-graph-memory/demo.py --mock

# Real mode — Claude extracts triples
ANTHROPIC_API_KEY=sk-... uv run python memory/07-knowledge-graph-memory/demo.py --real

# Persist the graph across runs
uv run python memory/07-knowledge-graph-memory/demo.py --mock --persist

# Custom graph path
uv run python memory/07-knowledge-graph-memory/demo.py --mock --persist --graph-path /tmp/my-graph.json
```

**Suggested conversation:**

```
You: Alice and Bob both work on project Orion.
You: Orion uses Redis and Kafka.
You: Bob reports to Carol.
You: Carol manages the infrastructure team.
You: graph                              ← inspect full graph
You: What does Alice work on?
You: Who does Bob report to?
You: What technologies does Orion use?
You: What team does Carol manage?
```

**Compare mock vs real extraction:**

The mock extractor only handles patterns it was explicitly programmed for. Try:

```
You: Alice is the principal engineer leading Orion.
```

- **Mock**: may miss this (no explicit LEADS pattern match for "principal engineer leading")
- **Real**: Claude understands "principal engineer leading X" → LEADS triple

This gap illustrates why production systems use LLM-based extraction — natural language is too varied for exhaustive regex coverage.

---

*Previous: [Episodic Memory](../06-episodic-memory/) | Next: Layered Memory (coming soon)*
