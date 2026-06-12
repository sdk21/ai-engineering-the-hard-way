# Lesson: Entity Memory

**Vertical:** Memory | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [The Problem with Unstructured Memory](#1-the-problem-with-unstructured-memory)
2. [What Is Entity Memory?](#2-what-is-entity-memory)
3. [Architecture](#3-architecture)
4. [How It Works — Step by Step](#4-how-it-works--step-by-step)
5. [The Entity Store](#5-the-entity-store)
6. [The Extractor](#6-the-extractor)
7. [Mock vs. LLM Extraction](#7-mock-vs-llm-extraction)
8. [Merge Semantics](#8-merge-semantics)
9. [Injection into Context](#9-injection-into-context)
10. [What Entity Memory Cannot Do](#10-what-entity-memory-cannot-do)
11. [Combining with Other Memory Types](#11-combining-with-other-memory-types)
12. [Failure Modes](#12-failure-modes)
13. [Key Principles](#13-key-principles)
14. [In the Real World](#14-in-the-real-world)
15. [Running the Experiment](#15-running-the-experiment)

---

## 1. The Problem with Unstructured Memory

The memory strategies so far — buffer, sliding window, summarization, vector retrieval — all store conversation content as **text**. This works, but it has a fundamental weakness: the model must re-parse and re-interpret that text every time it needs a specific fact.

Consider the fact "Alice is allergic to peanuts":

- In a **buffer**: it's buried somewhere in the message history, surrounded by noise
- In a **summary**: it might have been paraphrased to "the user has dietary restrictions" — precision lost
- In a **vector store**: it's retrievable if the query is semantically similar enough — but what if the query is "what should I have for lunch?" and the similarity score is 0.29, just below the threshold?

In all three cases, the fact is represented as text and its retrieval is probabilistic or context-dependent. There is no guaranteed, precise, always-available representation of "user.allergy = peanuts."

Entity memory solves this by extracting facts into a **structured store** where they are explicit, typed, and always available.

---

## 2. What Is Entity Memory?

Entity memory extracts named entities and their attributes from each conversation turn and stores them in a typed key-value structure. The store is injected into every model call as a structured "known facts" block.

```
Turn: "My name is Alice and I'm a Python engineer living in Tokyo."

Extracted:
    user:
      name: Alice
      role: Python engineer
      location: Tokyo

Store after turn:
    {
        "user": {
            "name": "Alice",
            "role": "Python engineer",
            "location": "Tokyo"
        }
    }

Injected into system prompt:
    ## Known facts about the user and context
    user:
      - location: Tokyo
      - name: Alice
      - role: Python engineer
```

The model now has precise, structured access to these facts — guaranteed, in every call, regardless of how long the conversation has been.

---

## 3. Architecture

```
Each user turn
      │
      ▼
  [Extractor]  ← mock (regex) or real (LLM → JSON)
      │
      ▼
  {entity: {attribute: value}}
      │
      ▼
  [EntityStore.merge()]  ← upserts, newer values overwrite older
      │
      ▼
  [System Prompt]
    base instructions
    + "## Known facts"
    + formatted entity store   ← injected every call
      │
      ▼
  [Recent Buffer]   ← last N verbatim turns
      │
      ▼
  [LLM API call]
```

The extraction step runs on every turn. The store grows as new facts are stated and shrinks only if facts are explicitly corrected (overwritten).

---

## 4. How It Works — Step by Step

```
Turn 1: "My name is Alice and I work as a software engineer."
  extract → {"user": {"name": "Alice", "role": "software engineer"}}
  store   → {"user": {"name": "Alice", "role": "software engineer"}}

Turn 2: "I live in Tokyo."
  extract → {"user": {"location": "Tokyo"}}
  store   → {"user": {"name": "Alice", "role": "software engineer", "location": "Tokyo"}}

Turn 3: "I'm allergic to peanuts."
  extract → {"user": {"allergy": "peanuts"}}
  store   → {"user": {"name": "Alice", ..., "allergy": "peanuts"}}

Turn 4: "Tell me a joke."
  extract → {}   ← no entities in small talk
  store   → unchanged

Turn 5: "Actually, I moved to Osaka last month."
  extract → {"user": {"location": "Osaka"}}
  store   → {"user": {"name": "Alice", ..., "location": "Osaka"}}   ← location updated

Turn 10: "What's my name?" (turn 1 long out of buffer)
  system prompt contains:
    ## Known facts
    user:
      - allergy: peanuts
      - location: Osaka
      - name: Alice        ← still here, 9 turns later
      - role: software engineer
  model: "Your name is Alice."  ✅
```

The fact from turn 1 is available in turn 10 with **zero retrieval** — it was extracted, stored, and is injected unconditionally.

---

## 5. The Entity Store

The store is a nested dictionary: `{entity_name → {attribute → value}}`.

```python
{
    "Alice": {
        "role": "software engineer",
        "location": "Osaka",
        "allergy": "peanuts",
        "preference": "Python"
    },
    "project": {
        "name": "Orion",
        "type": "distributed cache",
        "technology": "Redis"
    }
}
```

Design choices:

**Flat attributes** — Each attribute is a single string value. This is deliberately simple. Nested structures (e.g., `{"skills": ["Python", "Rust"]}`) add complexity without proportionate benefit for most use cases. If you need list values, join them as comma-separated strings.

**Entity granularity** — Group attributes under the most natural entity name. Use `"user"` for facts about the conversation participant; use specific names (`"Alice"`, `"Bob"`) when multiple people are discussed; use `"project"`, `"company"`, etc. for non-person entities.

**No schema enforcement** — Any string is a valid attribute name. This is intentional: the extractor discovers attributes from natural language and you can't enumerate all possible facts in advance.

---

## 6. The Extractor

The extractor is a function `(role, content) → EntityDict`. It is the most important component — the quality of your entity store is entirely determined by extraction quality.

Two approaches:

**Regex/rule-based** (mock mode) — Fast, deterministic, zero cost. Works well for structured inputs and known patterns. Brittle: misses paraphrases, novel phrasing, and complex sentences.

```python
# "My name is Alice" → user.name = "Alice"
re.search(r"my name is\s+(\w+)", text)

# "I live in Tokyo" → user.location = "Tokyo"
re.search(r"i live in\s+([A-Z]\w+)", text)
```

**LLM-based** (real mode) — Flexible, handles any phrasing, understands context. Higher cost (one extra API call per turn). The extractor prompt instructs the model to return JSON:

```python
prompt = """
Extract entities from: "I'm Alice, a Python engineer allergic to peanuts."
Return JSON only: {"user": {"name": "Alice", "role": "Python engineer", "allergy": "peanuts"}}
"""
```

The LLM extractor handles cases the regex cannot:
- "I go by Alice but my full name is Alicia" → `name: Alice`
- "Walnuts and cashews give me hives" → `allergy: tree nuts`
- "We're using the Temporal workflow engine for orchestration" → `project.technology: Temporal`

---

## 7. Mock vs. LLM Extraction

| | Mock (regex) | Real (LLM) |
|--|--|--|
| Speed | Instant | +100–300ms per turn |
| Cost | Free | ~$0.0001 per turn (Haiku) |
| Coverage | Known patterns only | Any natural language |
| Correctness | Exact when it fires | Occasionally hallucinates |
| Maintenance | Manual pattern updates | Prompt updates only |

In production, LLM extraction is standard for general-purpose assistants. Rule-based extraction is used when inputs are structured (form fields, slot-filling dialogs) or when latency and cost are critical.

---

## 8. Merge Semantics

When new entities are extracted, they are merged into the existing store with **last-write-wins** semantics:

```
store: {"user": {"location": "Tokyo"}}
new:   {"user": {"location": "Osaka", "role": "engineer"}}
after: {"user": {"location": "Osaka", "role": "engineer"}}
```

Location is overwritten (user moved), role is added (new fact). This handles corrections naturally: the user saying "actually, I live in Osaka now" will update the location without any special handling.

**Alternative: append semantics** — Keep all values as a list: `{"location": ["Tokyo", "Osaka"]}`. This preserves history but creates ambiguity — which is current? Use last-write-wins for mutable facts (location, job, preferences) and append for immutable facts (events, history). This experiment uses last-write-wins for simplicity.

---

## 9. Injection into Context

The entity store is injected as a structured block in the system prompt, rendered just before the base instructions or after:

```
You are a helpful assistant.

## Known facts about the user and context
user:
  - allergy: peanuts
  - location: Osaka
  - name: Alice
  - preference: Python
  - role: software engineer
project:
  - name: Orion
  - technology: Redis
  - type: distributed cache
```

This is always injected in full. Unlike vector memory, there is no retrieval step — all known facts are always visible. This is only viable because the entity store is small: it contains only extracted facts, not raw conversation text.

For very long conversations with many entities, you might inject only entities relevant to the current query. But for most applications, the full store is small enough (< 100 entities, < 500 tokens) to inject unconditionally.

---

## 10. What Entity Memory Cannot Do

Entity memory is precise but narrow. It has hard limits:

**It only captures what the extractor can extract.** Nuance, tone, implicit context, and complex multi-sentence reasoning cannot be represented as `{entity: {attr: val}}` pairs.

**It has no sense of narrative.** "Alice was frustrated that the Redis migration failed, but then Bob suggested a workaround that she eventually liked." This cannot be stored in a flat entity store without significant information loss.

**It cannot handle temporal sequences.** "Alice tried approach A, then B, and is now on C" — the entity store can only hold one current value per attribute. The history of attempts is lost.

**It does not know what it doesn't know.** If a fact was never stated (or the extractor missed it), the store is silent — not wrong, just incomplete. The model cannot tell the difference between "user has no allergies" and "allergy not yet stated."

These limitations motivate combining entity memory with other strategies. Entity memory handles structured facts; vector memory handles narrative and nuance; the buffer handles the immediate conversational flow.

---

## 11. Combining with Other Memory Types

Entity memory is most powerful when combined:

```
Memory stack for a production assistant:

  EntityMemory   → structured facts (always available, O(1) lookup)
  VectorMemory   → narrative, nuance, long-range recall
  RecentBuffer   → immediate conversational context
```

The three layers are complementary:
- "What's my name?" → answered from entity store (exact, guaranteed)
- "What did we decide about the architecture?" → answered from vector retrieval
- "What did you just say?" → answered from the buffer

This multi-layer architecture is what production memory systems converge toward. It is the foundation of Experiment 08 (Layered Memory).

---

## 12. Failure Modes

**Extraction hallucination** — The LLM extractor invents facts not present in the text. "I'm going to London" might incorrectly extract `location: London` when the user is visiting, not moving. Mitigation: conservative extraction prompts ("only extract explicitly stated facts"), validation layers.

**Stale facts** — The user's role changes, but the old role is already in the store and the update phrasing ("I got promoted") doesn't trigger an overwrite. The store silently holds outdated information. Mitigation: include timestamps on fact writes; prompt the extractor to identify corrections.

**Extraction noise from filler** — "That's a great point!" might extract `user.opinion: positive` — low-signal noise that bloats the store. Mitigation: post-extraction filtering (minimum confidence, exclude assistant turns by default).

**Attribute name proliferation** — Different extraction runs produce different attribute names for the same concept: `role`, `job`, `occupation`, `profession`. The store accumulates near-duplicate attributes. Mitigation: normalise attribute names in the extractor prompt or post-process with a canonical vocabulary.

**Over-injection** — In a very long conversation, the entity store grows large and the system prompt becomes expensive. Mitigation: importance scoring + pruning (drop rarely-accessed entities), or selective injection based on query relevance.

---

## 13. Key Principles

> **Principle 1 — Structured facts are more reliable than text-based retrieval.**
> A fact stored as `user.name = "Alice"` is always available and never misses a retrieval threshold. Text-based memory is probabilistic; entity memory is deterministic.

> **Principle 2 — Extraction quality is the ceiling.**
> The entity store is only as good as what the extractor can reliably pull out. Invest in extraction quality — it pays dividends on every subsequent turn.

> **Principle 3 — Entity memory is narrow but precise; use it for structured facts only.**
> Narrative, context, and nuance don't fit in key-value pairs. Don't try to force them in. Use entity memory for facts that have a clear entity + attribute + value structure.

> **Principle 4 — Last-write-wins handles corrections naturally.**
> Users correct themselves. "Actually, I live in Osaka now" should just overwrite the old location. Design your merge semantics to be forgiving of conversational corrections.

> **Principle 5 — Entity memory is a layer, not a complete solution.**
> No single memory strategy is sufficient. Entity memory pairs with a buffer for recency and vector memory for narrative recall. The combination is greater than the sum of its parts.

---

## 14. In the Real World

**Amazon Lex — slot filling**
Lex's core memory mechanism is slot filling — a form of entity memory. When a user says "I want to book a flight to Tokyo on Friday," Lex extracts `destination: Tokyo` and `date: Friday` into typed slots. These slots are explicitly tracked across the conversation. This is entity memory specialised for dialog tasks, used in millions of production voice and chat bots.

**Rasa — entity tracking**
Rasa's NLU pipeline extracts named entities from each user utterance and tracks them in a structured slot system. The conversation policy then uses slot values — not raw text — to determine next actions. Entity memory is the foundational memory primitive in the Rasa framework.

**LangChain — `ConversationEntityMemory`**
LangChain has a class named `ConversationEntityMemory` that implements this exact pattern. It uses an LLM to extract entities from each turn, maintains an entity store, and injects relevant entities into the prompt. The implementation is nearly identical to this experiment's `EntityMemory` class.

**OpenAI ChatGPT Memory**
ChatGPT's memory feature extracts structured facts ("User's name is Alice", "User lives in Tokyo") from conversations and stores them as discrete memory entries. The extraction step is LLM-based entity extraction; the storage is an entity store; the injection is a system prompt prefix. This is entity memory at production scale, serving hundreds of millions of users.

**Voiceflow / Botpress**
Conversational AI platforms like Voiceflow and Botpress represent user state as typed variables (entities) that are set and updated through the conversation flow. When a user says their name, the `{{user.name}}` variable is set and persists through the entire session — a UI-abstracted entity store.

**Salesforce Einstein / HubSpot AI**
CRM AI assistants extract and persist customer entities (name, company, deal stage, preferences, past interactions) from conversations and emails. These entities are merged into the CRM record — the world's largest deployment of entity memory, where the "entity store" is the CRM database.

**Mem0 — structured memory layer**
Mem0 extracts structured facts from conversations using an LLM and stores them in a searchable memory store keyed by user ID. At retrieval time, relevant memories are injected into context. The extraction and storage pattern is entity memory; the retrieval adds a semantic search layer on top — a hybrid of entity and vector memory.

---

## 15. Running the Experiment

```bash
# From the project root

# Mock mode — regex extraction, no API calls
uv run python memory/05-entity-memory/demo.py --mock

# Smaller buffer to confirm facts survive beyond the buffer
uv run python memory/05-entity-memory/demo.py --mock --buffer 4

# Real mode — LLM-based extraction + Claude chat
ANTHROPIC_API_KEY=sk-... uv run python memory/05-entity-memory/demo.py --real
```

**Suggested exercise sequence (use `--mock --buffer 4`):**

1. `"My name is Alice and I work as a software engineer."`
2. `"I live in Tokyo."`
3. `"I'm allergic to peanuts."`
4. `"I prefer Python over JavaScript."`
5. `"I'm building a project called Orion — a distributed cache."`
6. Type `entities` — inspect the structured store
7. `"Tell me a joke."` × 3 (push early turns out of buffer)
8. `"What's my name?"` — answered from entity store, not buffer
9. `"What am I allergic to?"` — same
10. `"Actually, I moved to Osaka last month."` — observe location updated
11. Type `entities` again — confirm Osaka replaced Tokyo

Then repeat with `--real` and compare extraction quality: the LLM extractor handles complex phrasing the regex misses.

---

*Previous: [Vector Memory](../04-vector-memory/) | Next: Episodic Memory (coming soon)*
