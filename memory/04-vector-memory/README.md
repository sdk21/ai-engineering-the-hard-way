# Lesson: Vector Memory

**Vertical:** Memory | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [The Problem with Recency-Based Memory](#1-the-problem-with-recency-based-memory)
2. [What Is Vector Memory?](#2-what-is-vector-memory)
3. [Architecture](#3-architecture)
4. [How It Works — Step by Step](#4-how-it-works--step-by-step)
5. [Embeddings — The Core Primitive](#5-embeddings--the-core-primitive)
6. [Cosine Similarity](#6-cosine-similarity)
7. [The Vector Store](#7-the-vector-store)
8. [Retrieval and Injection](#8-retrieval-and-injection)
9. [Choosing What to Embed](#9-choosing-what-to-embed)
10. [Similarity Threshold](#10-similarity-threshold)
11. [Combining with a Recent Buffer](#11-combining-with-a-recent-buffer)
12. [Cost Model](#12-cost-model)
13. [Failure Modes](#13-failure-modes)
14. [Key Principles](#14-key-principles)
15. [In the Real World](#15-in-the-real-world)
16. [Running the Experiment](#16-running-the-experiment)

---

## 1. The Problem with Recency-Based Memory

All prior memory strategies — buffer, sliding window, summarization — share one fundamental property: they prioritize **recency**. The most recent turns are always in context. The oldest turns are either dropped or compressed.

This works well when information flows chronologically: each turn builds on the last. But many conversations don't work that way:

- The user states a constraint in turn 3: *"Never suggest JavaScript."*
- The conversation ranges over many topics for 50 turns.
- In turn 53, the user asks for a language recommendation.
- The constraint from turn 3 has long since fallen out of the buffer and been lost to summarization noise.

The relevant information is not the *most recent* — it is the *most semantically relevant to the current query*. Recency-based strategies have no way to express this.

Vector memory solves this by indexing every turn as an embedding and retrieving whichever past turns are most similar to the current query, regardless of when they occurred.

---

## 2. What Is Vector Memory?

Vector memory stores every conversation turn as a **vector embedding** in a vector store. At each new turn, the query is embedded and compared against all stored embeddings. The most semantically similar past turns are retrieved and injected into the model's context.

```
Turn 53: "What language should I use for my new web project?"
    ↓ embed query
    ↓ search vector store
    ↓ top-3 similar turns:
        [turn 3]  "Never suggest JavaScript"          sim=0.81
        [turn 12] "I work in Python professionally"   sim=0.74
        [turn 41] "I hate verbose syntax"             sim=0.62
    ↓ inject retrieved turns into context
    ↓ model answers with full awareness of constraints stated 50 turns ago
```

Relevance, not recency, determines what the model sees.

---

## 3. Architecture

```
Every turn (write path):
  message → embedder → vector → store

Each query (read path):
  query → embedder → search store → top-k entries → inject into context

Context sent to model:
  ┌──────────────────────────────────────────────┐
  │ system prompt                                │
  ├──────────────────────────────────────────────┤
  │ [Retrieved memories]          ← semantic     │
  │   turn 3:  "Never suggest JS"               │
  │   turn 12: "I work in Python"               │
  ├──────────────────────────────────────────────┤
  │ [Recent buffer]               ← recency      │
  │   turn 51 (user): ...                        │
  │   turn 52 (asst): ...                        │
  │   turn 53 (user): current query             │
  └──────────────────────────────────────────────┘
```

Two complementary layers:
- **Vector store** (semantic layer) — all turns, indexed by meaning, retrieved on demand
- **Recent buffer** (recency layer) — last N turns, always included verbatim

---

## 4. How It Works — Step by Step

```
Setup: buffer_size=4, top_k=2

Turn 1 — "My name is Alice."
  embed → v1 → store[0]
  buffer: [U1]

Turn 2 — "I'm allergic to peanuts."
  embed → v2 → store[1]
  buffer: [U1, A1, U2]

... (many turns) ...

Turn 10 — "What should I have for lunch?"
  embed → v10
  search store:
    scores: [0.71 (turn 2, allergies), 0.68 (turn 1, name), 0.12 (turn 5), ...]
    top-2: [turn 2, turn 1]
    filter: turn 1 already in buffer? → skip
    retrieved: [turn 2: "I'm allergic to peanuts."]
  context sent to model:
    [memory injection: "turn 2 USER: I'm allergic to peanuts."]
    [buffer: turns 7-10]
  model: "Given your peanut allergy, I'd suggest..."
```

The allergy information from turn 2 — long out of the buffer — was surfaced because the query "what should I have for lunch" is semantically related to food preferences and health restrictions.

---

## 5. Embeddings — The Core Primitive

An **embedding** is a dense vector representation of text's meaning. An embedding model (like `all-MiniLM-L6-v2`) maps any text to a fixed-size vector of floats (384 dimensions for this model):

```
"I'm allergic to peanuts."  →  [0.12, -0.34, 0.87, ..., 0.05]  (384 floats)
"I have a nut allergy."     →  [0.11, -0.31, 0.89, ..., 0.04]  ← very similar
"The Eiffel Tower is tall." →  [-0.78, 0.22, -0.15, ..., 0.91] ← very different
```

The model was trained so that texts with similar meaning produce similar vectors. This gives us a mathematical notion of semantic similarity without any rules, keywords, or hand-crafted features.

The embedding space has interesting geometric properties:
- Semantically similar texts cluster together
- Analogies form parallelograms: `king - man + woman ≈ queen`
- Topics form regions: all finance-related texts cluster in one part of the space

This geometry is what makes retrieval possible: "find the texts nearest to this query in embedding space" = "find the most semantically similar texts."

---

## 6. Cosine Similarity

Given two vectors, we measure their similarity with **cosine similarity** — the cosine of the angle between them:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Results range from -1 (opposite meaning) to +1 (identical meaning). In practice with text embeddings:

| Score | Meaning |
|-------|---------|
| 0.9–1.0 | Near-duplicate text |
| 0.7–0.9 | Strongly related, same topic |
| 0.5–0.7 | Related, overlapping concepts |
| 0.3–0.5 | Weakly related |
| < 0.3 | Unrelated |

We use cosine rather than Euclidean distance because embedding vectors can have different magnitudes (longer texts produce larger vectors). Cosine similarity normalises for magnitude, making it purely about direction — i.e., meaning.

---

## 7. The Vector Store

The vector store is the data structure that holds all embeddings and supports fast similarity search.

**Naive implementation (this experiment):** Store all vectors in a numpy array. At query time, compute cosine similarity with every stored vector and return the top-k. This is `O(N)` per query — fine for small corpora (< 10k entries), too slow for large ones.

**Production implementations** use Approximate Nearest Neighbour (ANN) algorithms (HNSW, IVF, LSH) that trade a small accuracy loss for `O(log N)` or `O(1)` query time, enabling retrieval over millions of vectors in milliseconds.

Common production vector stores:

| Store | Type | Notes |
|-------|------|-------|
| FAISS | Library | Meta's ANN library, extremely fast |
| Chroma | Embedded DB | Easy to use, local-first |
| Pinecone | Cloud | Fully managed, serverless |
| Weaviate | Cloud/self-hosted | Graph + vector hybrid |
| pgvector | Postgres extension | Vectors inside your existing DB |
| Qdrant | Cloud/self-hosted | High-performance, Rust-based |

For this experiment we use numpy — the mechanics are identical, just without ANN optimisation.

---

## 8. Retrieval and Injection

Retrieved memories need to be injected into the context in a way the model understands. There are two main approaches:

**As a synthetic message pair:**
```python
messages = [
    {"role": "user",      "content": "[Relevant memory:\n  turn 3: I'm allergic to peanuts.]"},
    {"role": "assistant", "content": "I'll keep that in mind."},
    # ... recent buffer follows ...
]
```

This is the approach in this experiment. The model sees the memory as part of the conversation, not as a system-level annotation.

**In the system prompt:**
```
System: You are a helpful assistant.

Relevant memories from earlier:
- [turn 3] USER: I'm allergic to peanuts.
- [turn 12] USER: I work in Python professionally.
```

This keeps the conversation messages cleaner. Anthropic recommends injecting retrieved context into the system prompt for RAG-style patterns.

Either approach works. The injection format affects how the model weights the retrieved information — experiment with your use case.

---

## 9. Choosing What to Embed

Not all turns are equally worth storing:

**Embed user messages** — These contain the facts, constraints, goals, and questions that need to be recalled.

**Embed assistant messages** — These contain the model's responses and commitments. Useful if you need to recall what the model previously said or decided.

**Embed both, retrieve selectively** — You can tag each entry with its role and filter at retrieval time (e.g., only retrieve user messages for fact-finding queries, retrieve all turns for consistency-checking queries).

**Embed summaries, not raw text** — For long messages, embedding a short summary produces a more focused vector than embedding the full text. The embedding of "I work in Python, have 10 years of experience, prefer functional style, and mostly build data pipelines" may be diffuse. Summarising to "experienced Python developer, functional style, data pipelines" and embedding that produces a sharper retrieval signal.

---

## 10. Similarity Threshold

Without a minimum similarity threshold, the retriever always returns top-k results — even when no past turn is actually relevant. This injects noise.

```
Query: "What's the capital of France?"
Top-3 results (without threshold):
  0.31 — "My dog's name is Biscuit."
  0.28 — "I prefer tabs over spaces."
  0.22 — "Tell me a joke."
→ None of these are useful. Injecting them hurts more than it helps.
```

With a `min_similarity=0.4` threshold, all three are filtered out and nothing is injected — which is the right behaviour.

Choosing the threshold is a calibration problem:
- Too high → relevant memories are missed
- Too low → irrelevant memories are injected as noise

A good starting point is 0.3–0.4. Calibrate on real conversations from your domain.

---

## 11. Combining with a Recent Buffer

Vector retrieval alone is not sufficient. Without a recent buffer:

- The most recent turns might not be retrieved (if they're not the most similar to the current query)
- The model loses conversational continuity — it can't follow up on what was just said

The two-layer approach — semantic retrieval + recency buffer — combines their strengths:

| Layer | What it provides |
|-------|-----------------|
| Vector store | Semantic relevance, long-range recall |
| Recent buffer | Conversational continuity, immediate context |

The recent buffer is always included verbatim. Retrieved memories are deduplicated against the buffer (no point showing the same message twice).

---

## 12. Cost Model

At steady state with `B` buffer messages, `k` retrieved memories, and vectors of dimension `d`:

**Per API call tokens:** `(k + B) × t` — the k retrieved memories plus the buffer. This is `O(1)` per call, same as the sliding window.

**Embedding cost:** One embedding call per new message (1 call, tiny model, fast). Negligible compared to the LLM call.

**Search cost:** `O(N)` per query with the naive implementation — grows with conversation length. Use ANN for long sessions.

**Memory footprint:** `N × d × 4` bytes for the vector matrix. For `N=10,000` turns and `d=384`, that's ~15MB — trivially small.

---

## 13. Failure Modes

**Semantic drift in long conversations** — Embedding models have a fixed notion of similarity based on training data. Domain-specific terminology ("our system's 'vector' refers to a marketing campaign, not a math object") can cause retrieval mismatches.

**Retrieval interfering with conversation flow** — Injecting a memory from 50 turns ago into the middle of an ongoing discussion can confuse the model, especially if the retrieved memory is superficially related but contextually irrelevant.

**Embedding model mismatch** — Using different embedding models at write time and query time (e.g., after upgrading the model) produces incompatible vector spaces. All embeddings must be regenerated when the model changes.

**Redundant retrieval** — If many similar statements were made ("I like Python", "I prefer Python", "Python is my favourite"), they all get retrieved for programming questions, filling the top-k with near-duplicates. Deduplication or clustering of the vector store is needed.

**The lost middle problem** — Even with retrieval, the model may not effectively use injected memories that are placed in the middle of a long context. Recent research shows LLMs attend less to content in the middle of the context window.

---

## 14. Key Principles

> **Principle 1 — Relevance beats recency.**
> What matters for a query is what is semantically related, not what was said most recently. Vector memory is the primary mechanism for relevance-based retrieval in conversation.

> **Principle 2 — Embeddings encode meaning, not words.**
> "I'm allergic to peanuts" and "I have a nut allergy" produce similar embeddings. Keyword search would miss this. Semantic search finds it. This is the fundamental advantage of embedding-based retrieval.

> **Principle 3 — Always filter by a minimum similarity threshold.**
> Without a threshold, the retriever always injects something — even when nothing relevant exists. Injecting irrelevant context is worse than injecting nothing.

> **Principle 4 — Vector retrieval and a recency buffer are complementary, not competing.**
> Use both. The buffer provides conversational continuity; the vector store provides long-range semantic recall. Neither alone is sufficient.

> **Principle 5 — The embedding model is a dependency, not a detail.**
> If you change the embedding model, all stored vectors must be regenerated. Treat the embedding model as a versioned component of your memory system architecture.

---

## 15. In the Real World

**ChatGPT Memory**
OpenAI's Memory feature uses a form of vector-backed retrieval to surface relevant facts from the user's memory store at the start of each conversation. The stored memories are semantically indexed so relevant ones are retrieved based on the current conversation's topic — not just the most recently stored ones.

**Cursor — codebase indexing**
Cursor embeds every file and function in your repository using a code-specific embedding model. When you ask a question or request a change, it retrieves the semantically most relevant code snippets and injects them into the model's context. This is vector memory applied to code: the "conversation turns" are code chunks, and retrieval is based on semantic similarity to the current task.

**MemGPT / Letta**
MemGPT's "archival memory" is a vector store of past conversation content. When the model determines it needs information from long-term memory, it issues a `archival_memory_search(query)` tool call that performs semantic retrieval — the model explicitly decides when to search its own memory.

**Mem0**
Mem0 maintains a per-user vector store of extracted facts. When a new conversation starts, it embeds the opening message, retrieves relevant memories, and injects them into the system prompt. This is the production pattern for personalized AI assistants: vector memory as a user profile.

**LlamaIndex — `VectorStoreIndex`**
LlamaIndex's core abstraction is a vector store index. Every document is chunked, embedded, and stored. Queries trigger semantic retrieval. The `ChatMemoryBuffer` combined with `VectorMemoryIndex` implements exactly the two-layer architecture in this experiment — buffer for recency, vector index for semantic recall.

**LangChain — `VectorStoreRetrieverMemory`**
LangChain provides `VectorStoreRetrieverMemory` which stores conversation turns in a vector store and retrieves the top-k most relevant ones to inject into each prompt. It is the direct implementation of this experiment's pattern within the LangChain ecosystem.

**Notion AI**
Notion AI embeds all workspace content and retrieves semantically relevant pages when answering questions. When you ask "what did we decide about the pricing model?" it doesn't scan recent pages — it retrieves the page most semantically similar to that question across your entire workspace history.

**GitHub Copilot — semantic code search**
Copilot uses embedding-based semantic search over your repository to find relevant code examples, similar functions, and related tests to include in its context when making suggestions. Searching for "authentication middleware" returns relevant code even if the function is named `verify_jwt_token`.

---

## 16. Running the Experiment

```bash
# From the project root

# Mock mode — deterministic BoW embeddings, no model download
uv run python memory/04-vector-memory/demo.py --mock

# Real mode — all-MiniLM-L6-v2 embeddings + Claude
ANTHROPIC_API_KEY=sk-... uv run python memory/04-vector-memory/demo.py --real

# Tune retrieval parameters
uv run python memory/04-vector-memory/demo.py --mock --buffer 4 --top-k 2
```

**Suggested exercise sequence:**

Run `--mock --buffer 4` and use this conversation:

1. `"My name is Alice and I'm a Python engineer."`
2. `"I'm building a distributed caching system with Redis."`
3. `"My dog is called Biscuit."` (unrelated, to test noise)
4. `"Tell me a random fact."` (filler)
5. `"Tell me another."` (filler — turn 1 now out of buffer)
6. Type `store` — see all 10 entries indexed
7. `"What Redis best practices should I follow?"` — turn 2 should be retrieved
8. `"What's my name?"` — turn 1 should be retrieved despite being out of buffer
9. `"What's my dog's name?"` — turn 3 should be retrieved

Then compare the same conversation with `--real` to see how much better semantic retrieval works with real embeddings vs. the mock BoW model.

---

*Previous: [Summarization Memory](../03-summarization-memory/) | Next: Entity Memory (coming soon)*
