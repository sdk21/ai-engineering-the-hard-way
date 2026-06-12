# Lesson: Episodic Memory

**Vertical:** Memory | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [The Problem: Memory Dies with the Process](#1-the-problem-memory-dies-with-the-process)
2. [What Is Episodic Memory?](#2-what-is-episodic-memory)
3. [Architecture](#3-architecture)
4. [How It Works — Step by Step](#4-how-it-works--step-by-step)
5. [Session Lifecycle](#5-session-lifecycle)
6. [Episode Summarisation](#6-episode-summarisation)
7. [Storage Backends: JSON vs. SQLite](#7-storage-backends-json-vs-sqlite)
8. [Cross-Session Injection](#8-cross-session-injection)
9. [How Many Past Episodes to Recall](#9-how-many-past-episodes-to-recall)
10. [Failure Modes](#10-failure-modes)
11. [Key Principles](#11-key-principles)
12. [In the Real World](#12-in-the-real-world)
13. [Running the Experiment](#13-running-the-experiment)

---

## 1. The Problem: Memory Dies with the Process

Every experiment so far stores memory in RAM. The moment the Python process exits, everything is gone. Start a new session and the model has no idea you've ever spoken before.

This is fine for single-session tools. But most real applications involve returning users:

- A personal assistant you've used for months
- A coding agent you open every morning
- A customer support bot that should recognise repeat customers
- A tutoring system that tracks what a student has already learned

In all of these cases, the model needs to recall **past sessions** — not just the current one. This requires durable storage: memory that survives process restarts, server reboots, and deployments.

Episodic memory is the pattern that solves this. Each conversation session is an **episode** that is persisted to disk at the end of the session and recalled at the start of the next.

---

## 2. What Is Episodic Memory?

Episodic memory — borrowed from cognitive science — refers to memory of specific events experienced at a particular time. In human cognition: "last Tuesday I talked to Alice about the Redis migration." In AI systems: a structured record of a past conversation session, stored durably and recalled in future sessions.

```
Session 1 (Monday):
  User: "My name is Alice. I'm building a distributed cache."
  ...session ends...
  → Episode saved: {date: Mon, summary: "User is Alice, building a distributed cache."}

Session 2 (Wednesday):
  → Past episodes loaded from disk
  → Injected into system prompt:
       "Previous conversations:
        [Mon] User is Alice, building a distributed cache."
  User: "Do you remember what I'm working on?"
  Model: "Yes — you mentioned you're building a distributed cache."  ✅
```

The model can answer questions about past sessions even though the current session just started and the buffer is empty.

---

## 3. Architecture

```
Session N (current)
  ┌─────────────────────────────────────────────────────┐
  │ System Prompt                                       │
  │   base instructions                                 │
  │   + "## Previous conversations"  ◄── from disk      │
  │       [Mon] Alice building distributed cache.       │
  │       [Tue] Discussed Redis TTL strategies.         │
  ├─────────────────────────────────────────────────────┤
  │ Recent Buffer (current session only, in RAM)        │
  │   turn N-3 ... turn N                               │
  └─────────────────────────────────────────────────────┘
            │ session ends
            ▼
  Summariser → "User asked about Lua scripting in Redis."
            │
            ▼
  EpisodeStore.save(Episode)   ← persisted to disk
```

Two storage backends, same interface:
- `JSONEpisodeStore` — one JSON file per episode, human-readable
- `SQLiteEpisodeStore` — single database file, indexed, queryable

---

## 4. How It Works — Step by Step

```
Session 1 (first ever run):
  start_session()
    → load_recent(3) returns []   ← no past episodes
    → session_id = "a3f2b1c9"
    → started_at = "2025-06-01T10:00:00Z"

  ... user chats: "My name is Alice. I'm working on Orion."

  end_session()
    → summariser(turns) → "User introduced themselves as Alice,
                            working on project Orion."
    → Episode saved to disk

Session 2 (next day):
  start_session()
    → load_recent(3) returns [Episode(Mon, "Alice, project Orion")]
    → past episodes injected into system prompt

  User: "Do you remember my name?"
  System prompt contains: "[Mon] Alice, project Orion."
  Model: "Your name is Alice!"  ✅

Session 3:
  start_session()
    → load_recent(3) returns [Episode(Mon, ...), Episode(Tue, ...)]
    → both injected
```

Each new session inherits a compressed view of the N most recent sessions. The store grows indefinitely on disk; the context cost stays bounded at `recall_n` episode summaries.

---

## 5. Session Lifecycle

A session has three phases:

**Start** — `start_session()`
- Mint a unique session ID
- Record `started_at` timestamp
- Load the N most recent past episodes from the store
- Clear the in-RAM turn buffer

**Active** — `add_user_message()` / `add_assistant_message()`
- Turns accumulate in RAM (the current buffer)
- Past episodes are injected into the system prompt on every LLM call

**End** — `end_session()`
- Summarise the session's turns
- Persist the episode to the store
- The episode is available for recall in all future sessions

The application is responsible for calling `end_session()` at the right time — e.g., when the user closes the chat window, after a timeout of inactivity, or at a natural task completion point.

If the process is killed before `end_session()`, the session is lost. Production systems handle this with periodic auto-saves or by persisting raw turns continuously (write-ahead log) and summarising asynchronously.

---

## 6. Episode Summarisation

Each episode is stored with a **summary** — a compact text description of what happened in the session. This summary is what gets injected into future sessions, not the raw turns.

Why summarise rather than store raw turns?
- Raw turns can be hundreds of messages — too expensive to inject in full
- Summaries are 1–3 sentences — cheap to inject, even for many past episodes
- The summary focuses on what is worth remembering, not every word exchanged

**Mock summariser** — Extracts key facts with regex (fast, no cost, limited quality):
```
"User said: 'My name is Alice' | User's name: Alice | Session had 8 user turn(s)."
```

**Real summariser** — Claude condenses the session (high quality, one API call per session end):
```
"Alice introduced herself as a software engineer building a distributed caching
system called Orion. She asked about Redis TTL strategies and connection pooling."
```

The summariser only runs once per session (at `end_session()`), not on every turn — it is not on the hot path.

---

## 7. Storage Backends: JSON vs. SQLite

This experiment ships two backends to make the storage tradeoff concrete.

### JSONEpisodeStore

```
/tmp/ai-hardway-episodic/episodes/
  a3f2b1c9.json
  7d4e2a1b.json
  ...
```

Each episode is a self-contained JSON file. Human-readable, trivial to inspect, no dependencies beyond stdlib.

```json
{
  "session_id": "a3f2b1c9",
  "started_at": "2025-06-01T10:00:00Z",
  "ended_at": "2025-06-01T10:23:00Z",
  "summary": "Alice discussed Redis TTL strategies.",
  "turns": [
    {"role": "user", "content": "My name is Alice.", "timestamp": "..."},
    ...
  ]
}
```

**Loading N recent episodes** requires scanning all file mtimes — `O(N_total)`. Fine for tens or hundreds of sessions; slow for thousands.

### SQLiteEpisodeStore

```
/tmp/ai-hardway-episodic/episodes.db   ← single file
```

Schema:
```sql
episodes(session_id PK, started_at, ended_at, summary)
turns(id PK, session_id FK, role, content, timestamp)

INDEX ON episodes(started_at DESC)
```

**Loading N recent episodes** is `O(log N_total)` due to the index on `started_at`. Supports arbitrary SQL queries:

```sql
-- Find all sessions where Redis was mentioned
SELECT * FROM turns WHERE content LIKE '%Redis%';

-- Sessions in the last 7 days
SELECT * FROM episodes WHERE started_at > datetime('now', '-7 days');

-- Most talked-about topics (rough)
SELECT content, COUNT(*) FROM turns GROUP BY content ORDER BY 2 DESC;
```

### Which to use

| Situation | Recommendation |
|-----------|---------------|
| Development / debugging | JSON (human-readable, inspect with `cat`) |
| < ~500 sessions total | Either (JSON is simpler) |
| > 500 sessions | SQLite |
| Need to query episode content | SQLite |
| Shared store across multiple processes | SQLite (handles concurrent writes) |
| Production application | SQLite |

Switch with `--store json` or `--store sqlite`.

---

## 8. Cross-Session Injection

Past episodes are injected into the system prompt as a chronological list of summaries:

```
## Previous conversations
  [2025-06-01] Alice introduced herself as a software engineer building Orion.
  [2025-06-02] Discussed Redis TTL strategies and connection pooling best practices.
  [2025-06-03] Asked about Lua scripting in Redis for atomic operations.
```

Design considerations:

**Chronological order** — Oldest first reads naturally as a narrative. The most recent session is at the bottom, closest to the current conversation — models attend more to content near the end of the context.

**Only summaries, not raw turns** — Each summary is 1–3 sentences (~50 tokens). Three past episodes ≈ 150 tokens — negligible context cost.

**Recency bias** — Load only the N most recent episodes (default: 3). There is no point injecting a session from 6 months ago for most applications. Adjust `recall_n` based on how far back is useful.

**Selective injection** — Advanced: embed episode summaries and only inject the k most semantically similar to the current session's opening message. This is episodic memory + vector retrieval — a preview of Experiment 08.

---

## 9. How Many Past Episodes to Recall

`recall_n` is the number of past episodes injected into each session. The right value depends on:

| Use case | Recommended `recall_n` |
|----------|----------------------|
| Simple personal assistant | 3–5 |
| Long-running coding agent | 5–10 |
| Customer support (per user) | 1–3 (recent context usually sufficient) |
| Tutoring / learning system | 10+ (full learning history matters) |
| First session (no history) | 0 (nothing to inject) |

At `recall_n=5` with 50-token summaries: ~250 tokens of past context per session — a trivial cost even against a 200k-token context window.

---

## 10. Failure Modes

**Session not persisted on crash** — If the process is killed before `end_session()`, the session is lost. Mitigation: write turns to the store continuously (append-only log) and summarise asynchronously after the session ends.

**Summary quality degrades with session length** — A 200-turn session summarised into 2 sentences loses a lot. Mitigation: use longer summaries for longer sessions, or store structured facts (entity memory) alongside the free-text summary.

**Growing store, stale episodes** — After years of use, early episodes are rarely relevant but still take up space and potentially get injected. Mitigation: TTL on episodes (auto-delete after N days), or relevance-based injection (only inject episodes similar to the current session).

**Summarisation hallucination** — The LLM summariser may invent facts not present in the session. "User mentioned they prefer Rust" when they never said that. Future sessions then carry this fabricated fact. Mitigation: structured extraction (entity memory) for verifiable facts; treat summaries as approximate context, not ground truth.

**Concurrent sessions** — JSON files have no write locking. Two sessions writing simultaneously can corrupt each other. SQLite handles concurrent writes safely with WAL mode.

---

## 11. Key Principles

> **Principle 1 — In-process memory is temporary; durable memory requires explicit persistence.**
> Every memory strategy in this vertical is useless across sessions unless you write it to disk. Persistence is not an implementation detail — it is the feature.

> **Principle 2 — Store the session; inject the summary.**
> Keep full turn logs for auditability and future re-processing. Inject only summaries into context — they are cheap, focused, and sufficient for cross-session recall.

> **Principle 3 — SQLite is the right default for production episode storage.**
> It is stdlib, transactional, indexed, and handles concurrent writes. JSON files are for development. Reach for SQLite as soon as you need more than a handful of sessions.

> **Principle 4 — `end_session()` must be called reliably.**
> An episode that is never persisted never happened. Design your application to guarantee `end_session()` runs — even on crash (via try/finally, atexit, or continuous turn persistence).

> **Principle 5 — Episodic memory is the persistence layer; it composes with all other memory types.**
> Entity memory tells you what the user's name is. Vector memory retrieves relevant past turns. Episodic memory ensures both survive across sessions. They are layers, not alternatives.

---

## 12. In the Real World

**ChatGPT — chat history**
Every ChatGPT conversation is persisted as an episode. When you start a new conversation, you can scroll through your chat history — these are episodes stored in OpenAI's database. The Memory feature goes further: it extracts structured facts (entity memory) from episodes and injects them as a persistent system prompt — episodic + entity memory combined.

**Anthropic Claude Projects**
Claude Projects persist a system prompt and optionally file attachments across sessions. This is a manually curated episodic memory: you write the "summary" of relevant context yourself and it is injected into every session in that project. Automated episodic memory automates this curation step.

**MemGPT / Letta**
MemGPT's architecture has three memory tiers: in-context (buffer), external (vector store), and archival (episode store). At session boundaries, MemGPT explicitly persists the conversation to archival storage and loads a summary at the start of the next session — the exact lifecycle described in this experiment, with a vector store for within-session retrieval on top.

**Mem0**
Mem0 persists extracted facts from each conversation to a per-user memory store. When a new session starts, it retrieves relevant past memories and injects them. The persistence layer is episodic memory; the retrieval layer is vector search — a hybrid. This is one of the most widely deployed open-source implementations of the pattern in this experiment.

**Microsoft 365 Copilot — meeting memory**
After each Teams meeting, Copilot generates a meeting summary and action items — structured episode summarisation. These summaries are stored and can be referenced in future Copilot interactions: "Based on Monday's meeting, the next steps were..." This is episodic memory at enterprise scale.

**Replit Agent / Devin**
Long-running software engineering agents persist task context across sessions. When you resume a task, the agent loads the prior session's state — what files were modified, what approaches were tried, what is still outstanding. This is episodic memory where the "episode" is a work session on a software task.

**Intercom / Zendesk — customer conversation history**
Customer support platforms persist every customer interaction as an episode in the customer's record. When an agent (human or AI) handles a new ticket, they see prior conversation history — episodic memory for support workflows. The "injection" is the sidebar showing prior conversations; the "summarisation" is the ticket subject line.

---

## 13. Running the Experiment

```bash
# From the project root

# Session 1 — JSON store (default)
uv run python memory/06-episodic-memory/demo.py --mock

# Session 2 — same command, same store — model remembers session 1
uv run python memory/06-episodic-memory/demo.py --mock

# Use SQLite backend instead
uv run python memory/06-episodic-memory/demo.py --mock --store sqlite

# Real mode — Claude summarises each session
ANTHROPIC_API_KEY=sk-... uv run python memory/06-episodic-memory/demo.py --real --store sqlite

# Inspect past episodes during the session
> history

# Wipe the store and start fresh
uv run python memory/06-episodic-memory/demo.py --mock --clear
```

**Suggested exercise sequence:**

**Run 1:**
1. `"My name is Alice and I'm building a distributed cache called Orion."`
2. `"I'm using Redis as the backend."`
3. `stats` — see 0 past episodes
4. `quit` — session saved to disk

**Run 2 (immediately after):**
1. `history` — see session 1 summarised
2. `"Do you remember what I told you last time?"` — model should recall
3. `"What's my name?"` — answered from episode summary
4. `quit`

**Run 3:**
1. `history` — see both prior sessions
2. Notice the model now has two episodes of context
3. Compare JSON files vs. SQLite with `--store sqlite`

Compare `--mock` vs `--real` to see how LLM summarisation produces richer, more useful episode summaries than regex extraction.

---

*Previous: [Entity Memory](../05-entity-memory/) | Next: Knowledge Graph Memory (coming soon)*
