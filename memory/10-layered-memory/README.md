# Lesson: Layered Memory

**Vertical:** Memory | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## Table of Contents

1. [From Isolated Experiments to a Complete System](#1-from-isolated-experiments-to-a-complete-system)
2. [The Five Layers](#2-the-five-layers)
3. [Architecture](#3-architecture)
4. [Write Routing](#4-write-routing)
5. [Read Composition and Token Budgets](#5-read-composition-and-token-budgets)
6. [Layer Priority and Conflict](#6-layer-priority-and-conflict)
7. [Layer Fallback](#7-layer-fallback)
8. [Session Lifecycle](#8-session-lifecycle)
9. [What Each Layer Is Good For](#9-what-each-layer-is-good-for)
10. [Failure Modes](#10-failure-modes)
11. [Key Principles](#11-key-principles)
12. [In the Real World](#12-in-the-real-world)
13. [Running the Experiment](#13-running-the-experiment)

---

## 1. From Isolated Experiments to a Complete System

Every prior experiment in this vertical solved one memory problem in isolation:

| Experiment | Problem solved |
|-----------|---------------|
| 01–02 | What happened in the last few turns? |
| 03 | How do we fit a long conversation into a bounded context? |
| 04 | Which past turns are relevant to this question? |
| 05 | What structured facts do we know about the user? |
| 06 | What happened in past sessions? |
| 07 | What are the relationships between entities? |
| 08 | How confident are we in each fact? Where did it come from? |
| 09 | When facts conflict, which one wins? |

In isolation, each is a partial solution. A real assistant needs all of them simultaneously:

- The model needs to know what was just said (working memory)
- It needs to know who the user is and what they are working on (structured facts)
- It needs to know what happened in past sessions (episodic)
- It needs to retrieve relevant context that didn't make it into the structured layer (semantic)
- It needs to know how confident it should be in each piece of recalled information (provenance)

This experiment assembles these into a **layered stack** — a single system where each layer handles a different type of memory, they are written to independently, and they are composed into a single system prompt on every LLM call.

---

## 2. The Five Layers

```
┌─────────────────────────────────────────────────────┐
│  Layer 5  Provenance + Confidence  (meta-layer)     │
│           wraps layers 2–4; tracks confidence,      │
│           reinforcement, and contradictions          │
├─────────────────────────────────────────────────────┤
│  Layer 4  Semantic / Vector        (long-tail)       │
│           embeds every turn; retrieves top-k by      │
│           similarity to the current query            │
├─────────────────────────────────────────────────────┤
│  Layer 3  Episodic                 (history)         │
│           persists session summaries across runs;    │
│           injects N most recent on every call        │
├─────────────────────────────────────────────────────┤
│  Layer 2  Entity + KG              (structure)       │
│           extracts facts and relationships;          │
│           confidence-filtered injection              │
├─────────────────────────────────────────────────────┤
│  Layer 1  Working memory           (buffer)          │
│           FIFO buffer of recent turns;               │
│           becomes the messages list                  │
└─────────────────────────────────────────────────────┘
```

**Layer 1 — Working memory** is the conversation buffer from experiments 01–02. Recent turns, bounded size, always in context. This is not part of the system prompt — it is the `messages` list passed to the LLM.

**Layer 2 — Structured facts + KG** is the entity store and knowledge graph from experiments 05 and 07. Structured beliefs about people, projects, technologies, and their relationships. Confidence-filtered: facts below the threshold are not injected.

**Layer 3 — Episodic** is the session persistence layer from experiment 06. End-of-session summaries persisted to a store, recalled at the start of each new session. Cross-session context without replaying raw turns.

**Layer 4 — Semantic** is the vector retrieval layer from experiment 04. Every turn is embedded and indexed. On each query, the top-k most similar past turns are retrieved and injected. Captures long-tail context that didn't make it into the structured layer.

**Layer 5 — Provenance + Confidence** is the meta-layer from experiments 08–09. It is not a separate store — it is the confidence tracking woven into layers 2–4. Every structured fact carries a confidence score. Conflicting assertions update scores rather than silently overwriting. This layer is what makes the others *reliable*.

---

## 3. Architecture

```
User turn
    ↓
ExtractorFn(role, content)
    → entity facts + KG edges

Write routing (all happen on every user turn):
    Layer 1:  WorkingMemory.push(role, content)
    Layer 2:  StructuredMemory.upsert_fact() for each fact
              StructuredMemory.upsert_edge() for each edge
    Layer 4:  SemanticMemory.index(turn_index, role, content)
    Layer 3:  (write deferred — happens at end_session())

Read composition (on every get_system_prompt(query)):
    1. Base instructions                         (always)
    2. EpisodicMemory.format_for_prompt()        (if episodes exist)
    3. StructuredMemory.format_for_prompt()      (if facts above threshold)
    4. SemanticMemory.retrieve(query)            (if hits above min_similarity)

    Each section capped to its character budget.
    Empty sections omitted.

Session lifecycle:
    start_session() → reset turn index
    ... N turns ...
    end_session()   → summarize working memory → EpisodicMemory.save_session()
```

---

## 4. Write Routing

Every user turn triggers writes to three layers simultaneously:

```python
def add_user_message(self, content: str) -> dict:
    self._turn_index += 1
    self._working.push("user", content)          # Layer 1: immediate
    self._semantic.index(turn_index, ...)        # Layer 4: embed + store
    facts, edges = self._extractor("user", content)
    for entity, attribute, value in facts:
        self._structured.upsert_fact(...)        # Layer 2: extract + upsert
    for source, relation, target in edges:
        self._structured.upsert_edge(...)        # Layer 2: graph
```

Layer 3 (episodic) is the exception: it is written at `end_session()`, not on every turn. This is intentional — the episodic write is expensive (requires an LLM summarization call) and is only meaningful at natural session boundaries.

**Why write to all layers eagerly?**

Each layer serves a different retrieval pattern. A fact that makes it into the structured layer (L2) is queryable by attribute. The same fact embedded in the semantic layer (L4) is queryable by similarity. The same fact in the episodic layer (L3) is queryable by session recency. Writing to all layers ensures that no single retrieval path is the bottleneck.

---

## 5. Read Composition and Token Budgets

The system prompt is assembled in a fixed priority order with character budgets per section:

```python
@dataclass
class LayerBudget:
    episodic_chars:   int = 600   # ~150 tokens
    structured_chars: int = 800   # ~200 tokens
    semantic_chars:   int = 500   # ~125 tokens
```

Total injected context: ~475 tokens, independent of how long the user has been talking or how many facts have accumulated.

**Priority order:**
1. Base instructions — always first, uncapped
2. Episodic summaries — oldest first (chronological), capped at `episodic_chars`
3. Structured facts + KG — entity facts then graph edges, capped at `structured_chars`
4. Semantic hits — most similar first, capped at `semantic_chars`

Each section is independently truncated at a newline boundary to avoid injecting half-sentences. Sections that are empty (no episodes yet, no facts above threshold, no semantic hits) are omitted entirely — empty headers waste tokens.

**Why this order?**

Episodic comes before structured because it provides narrative context — who the user is as a story, not just as a set of attributes. Structured comes before semantic because structured facts are more reliable (extracted and confidence-tracked) than raw retrieved turns. Semantic is last because it is the most variable — it may retrieve highly relevant context or nothing at all depending on the query.

---

## 6. Layer Priority and Conflict

When the same fact appears in multiple layers, there is no explicit conflict resolution across layers. Each layer is authoritative for its own storage:

- **Structured (L2) is authoritative for facts and relationships.** If the model needs to know the user's employer, it reads from L2. Confidence scoring (L5) ensures stale or contradicted facts are suppressed.
- **Episodic (L3) is advisory.** Session summaries are injected as narrative context, not as ground-truth facts. The model should treat episodic content as "what happened, approximately" rather than "verified current state."
- **Semantic (L4) is associative.** Retrieved turns are the raw evidence; the model must reason about whether they are still relevant.

In practice, L2 and L3 may say different things about the same attribute:

```
L2 (structured): user.employer = "Beta Corp"  [conf=0.85]
L3 (episodic):   "Session 3: User mentioned they work at Acme Corp."
```

The structured layer reflects the most recent and conflict-resolved value. The episodic layer reflects what was said in a past session. A well-designed base prompt instructs the model to prefer the structured facts section for current state and the episodic section for historical context.

---

## 7. Layer Fallback

When a layer returns nothing useful, its token budget is not wasted — it is simply omitted:

```python
episodic_text = self._episodic.format_for_prompt()
if episodic_text:
    sections.append(self._truncate(episodic_text, self.budget.episodic_chars))
# If empty (first session), section is omitted — no "Previous sessions: (none)" header
```

The semantic layer additionally excludes turns that are already in the working memory buffer:

```python
buffer_turn_indices = set(
    range(max(1, self._turn_index - self._working.turn_count + 1), self._turn_index + 1)
)
hits = self._semantic.retrieve(query, exclude_turns=buffer_turn_indices)
```

This prevents semantic retrieval from re-injecting content that is already visible to the model in the messages list — the single most common waste of context in naive layered systems.

---

## 8. Session Lifecycle

```
memory.start_session()
    → reset turn index

... user and assistant turns ...

memory.end_session()
    → summarizer(working_memory.messages()) → summary string
    → EpisodicMemory.save_session(Episode)
    → new session_id minted

memory.start_session()     ← next session
    → turn index reset
    → working memory is NOT cleared (continuation mode)
    → episodic layer now has one more episode to inject
```

The working memory is intentionally not cleared on session transitions. `start_session()` and `end_session()` manage the episodic boundary, not the conversation boundary. In a real application:

- Session = one login, one chat window, one task — defined by the application
- The working buffer continues across sessions if the conversation continues
- Episodic captures the session as a whole for future recall

---

## 9. What Each Layer Is Good For

| Question | Best layer |
|----------|-----------|
| "What did the user just say?" | L1 (working) |
| "What is the user's name / employer / role?" | L2 (structured) |
| "Who reports to whom in their org?" | L2 (KG edges) |
| "What happened in our conversation last week?" | L3 (episodic) |
| "Did the user mention X at some point?" | L4 (semantic) |
| "How confident are we that the user's role is correct?" | L5 (provenance on L2) |
| "Was this fact ever contradicted?" | L5 (provenance on L2) |

No single layer answers all questions well. The layered stack answers all of them.

---

## 10. Failure Modes

**Layer bleed — same fact injected twice**
A fact in the structured layer and a semantically similar turn retrieved by L4 can both land in the system prompt, doubling the context used for one piece of information. Mitigation: semantic retrieval already excludes buffer turns; extend the exclusion to turns whose content matches a structured fact above a similarity threshold.

**Episodic summaries contradicting structured facts**
If the user changed their employer between sessions, the episodic summary says "works at Acme" while the structured layer says "works at Beta." The model sees both. Mitigation: structure the base prompt to explicitly instruct the model to treat the "Known facts" section as current state and the "Previous sessions" section as historical context.

**Budget starvation**
If the structured layer has 30 entities and the episodic layer has 10 sessions, both sections will be heavily truncated. Mitigation: increase the budget (costs more tokens per call); or implement relevance-based injection — only inject structured facts relevant to the current query, not all facts.

**Semantic layer growing without bound**
Every turn is indexed. After 10,000 turns the retrieval scan is O(10,000). Mitigation: periodic pruning (delete turns older than N sessions), or replace the linear scan with FAISS (see experiment 04).

**Extraction quality cascades**
If the extractor misparses a fact, the error propagates to L2 (wrong fact), L5 (wrong confidence), and the system prompt indefinitely. L3 and L4 also store the raw text, so erroneous content can resurface via semantic retrieval. Mitigation: use an LLM extractor (higher accuracy) and implement human-in-the-loop correction via `resolve_conflict` from experiment 09.

---

## 11. Key Principles

> **Principle 1 — Each layer serves a different retrieval pattern.**
> Working memory answers "what just happened." Structured memory answers "what is known." Episodic answers "what happened before." Semantic answers "what is relevant." They are not alternatives — they are orthogonal. A system with only one layer has blind spots that only the other layers can fill.

> **Principle 2 — Write to all layers; read from each selectively.**
> Every turn should update every write-path layer (working, structured, semantic). The cost is low and the coverage is complete. Reading is selective — each layer is queried differently and capped to its budget. Don't conflate the write strategy with the read strategy.

> **Principle 3 — Token budget is a first-class constraint.**
> The system prompt is finite. Each layer must have a budget, and the total must leave room for the base instructions and the query. A layered system without budget management will fill the context with the loudest layer (usually episodic or semantic) and starve the others.

> **Principle 4 — Exclude what is already visible.**
> The working memory is already in the messages list. Semantic retrieval must exclude it. Injecting the same content twice in different forms wastes tokens and can confuse the model with redundant context.

> **Principle 5 — The layers are additive, not redundant.**
> Adding a new layer should not require rewriting the others. Each layer exposes a clean interface (push, retrieve, format\_for\_prompt). The composition engine assembles them independently. This is what allows you to swap in a FAISS store, a SQLite episodic store, or a full conflict-resolution engine without changing the rest of the stack.

---

## 12. In the Real World

**OpenAI ChatGPT Memory**
ChatGPT's memory system is a two-layer stack: a structured fact store (L2 equivalent — "Alice is a software engineer working on a distributed cache") and a conversation history layer (L3 equivalent — the persistent chat history sidebar). The structured layer is written to when the model judges a fact worth remembering; the conversation layer is written automatically. On each new conversation, both layers are injected into the system prompt — the exact read composition in this experiment.

**Anthropic Claude Projects**
Claude Projects compose two layers: a manually-curated system prompt (L2 equivalent — structured facts and context the user has written themselves) and uploaded files (L4 equivalent — document chunks retrievable by semantic search). The episodic layer is absent by design in Projects — sessions are independent. This is a deliberate product tradeoff: Projects optimise for task consistency, not cross-session relationship memory.

**MemGPT / Letta**
MemGPT's architecture maps directly onto the layered model: in-context memory (L1 — the buffer), external memory (L4 — a vector store), and archival memory (L3 — episode store). MemGPT adds an explicit memory management agent that decides which content to move between layers — an automated version of the write routing and budget management in this experiment.

**Mem0**
Mem0's production architecture is a three-layer composition: a vector store (L4) for semantic retrieval of past facts, a property graph (L2 equivalent) for structured entity relationships, and a summary layer (L3) for session-level context. The system prompt is assembled by querying all three and ranking by relevance — a more sophisticated version of the priority-ordered composition in this experiment.

**Replit Agent / Devin — task memory**
Long-running software engineering agents maintain multiple memory layers: the current working context (L1 — recent tool outputs, current file contents), a task memory (L2/L3 — what has been done, what is outstanding, what approaches were tried), and a codebase search index (L4 — semantic retrieval over the full repository). The layered architecture is what allows these agents to resume a multi-day task without replaying the entire history.

**Microsoft 365 Copilot**
Copilot for Microsoft 365 assembles context from: the current document/email (L1 — working context), the user's profile and organisational graph (L2 — structured facts from Azure AD), meeting summaries and past emails (L3 — episodic, via Microsoft Graph), and semantically relevant documents from SharePoint (L4 — enterprise vector search). The system prompt is assembled from all four, with token budgets managed by the Copilot orchestration layer — the most widely deployed layered memory system in the world.

---

## 13. Running the Experiment

```bash
# From the project root

# Mock mode — no API key, no sentence-transformers
uv run python memory/10-layered-memory/demo.py --mock

# Real mode — Claude extracts, embeds, and summarises
ANTHROPIC_API_KEY=sk-... uv run python memory/10-layered-memory/demo.py --real
```

**Suggested sequence — exercises all four layers:**

```
# Session 1 — build up the layers
You: My name is Alice and I'm a software engineer.
You: I work at Acme Corp on project Orion.
You: Orion uses Redis and Kafka.
You: Bob is my colleague — he works on the infrastructure side.
You: stats      ← see all layer counts
You: prompt     ← see layers 2–4 assembled in the system prompt
You: endsession ← save session to episodic layer (layer 3)

# Session 2 — recall from all layers
You: newsession
You: prompt     ← episodic section now present
You: Do you remember what project I'm working on?
     ← episodic layer provides session summary
You: What does Orion use?
     ← structured KG answers via USES edges
You: What did I say at the start of our last session?
     ← semantic layer retrieves the relevant past turn
You: stats      ← episodes=1 now shows in the count
```

**What to observe:**

- `prompt` in session 1: only layers 2 and 4 (no episodes yet)
- `prompt` in session 2: all four layers present
- The `[L2]` extraction log after each turn shows what was written to the structured layer
- `stats` shows counts for every layer independently
- Asking about past context (semantic layer) vs current facts (structured layer) vs previous sessions (episodic) exercises different retrieval paths

---

*Previous: [Conflict Resolution Memory](../09-conflict-resolution-memory/) | Next: Self-Reflecting Memory (coming soon)*
