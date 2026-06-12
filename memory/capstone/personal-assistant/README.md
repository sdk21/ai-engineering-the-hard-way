# Capstone: Personal Assistant

**Vertical:** Memory | **Type:** Capstone | **Status:** ✅ Ready

---

## What This Capstone Is

The memory vertical experiments (01–10) each teach one concept in isolation. This capstone synthesises all of them into a single application that behaves like a production memory system:

| Experiment | Concept | Where it appears here |
|-----------|---------|----------------------|
| 01–02 | Conversation buffer | Working memory buffer, restored on resume |
| 03 | Summarisation | Session summariser at `endsession` |
| 04 | Vector / semantic | `PersistentSemanticIndex`, retrieves by similarity |
| 05 | Entity facts | `PersistentFactStore`, confidence-tracked |
| 06 | Episodic | Episodes persisted to SQLite, injected on startup |
| 07 | Knowledge graph | Edge extraction, stored alongside facts |
| 08 | Provenance + confidence | Confidence scores on every fact, suppression |
| 09 | Conflict resolution | Reinforcement / contradiction on upsert |
| 10 | Layered composition | `stable_prefix()` + `dynamic_suffix()` |

In addition it adds four production concerns the experiments deliberately skip:

---

## Production Concerns

### 1. Persistence

All memory survives process restarts.

```
~/.config/ai-hardway/assistant/
    memory.db       ← SQLite: facts + episodes + turns
    semantic.json   ← vector index (all embedded turns)
    session.json    ← current session state (for resume)
```

Quit and restart — your name, employer, project, and past session summaries are all intact. The semantic index preserves every turn embedding across runs.

### 2. Session Resume

If you restart within 4 hours of your last message, the session is resumed: the same session ID, the same conversation buffer. You can pick up mid-conversation as if the process never stopped.

```
╔══ Personal Assistant [REAL] ══╗
  Resumed session a3f2b1c9 (6 turn(s) in buffer).
  2 past episode(s) in memory.
```

After 4 hours (configurable via `resume_window_hours`), a new session is started automatically. The old session is not yet saved to the episodic layer — you need to call `endsession` explicitly to persist it, or the application can do it on clean exit.

### 3. Prompt Caching

The system prompt is split into two parts:

```
stable_prefix()    — base instructions + episodic summaries + structured facts
                     Changes at most once per turn (only when facts update).
                     Pass to Anthropic API with cache_control: {"type": "ephemeral"}

dynamic_suffix()   — semantic hits for the current query
                     Changes every turn. Not suitable for caching.
```

```python
# How to use with prompt caching on the Anthropic API:
system = [
    {
        "type": "text",
        "text": memory.stable_prefix(),
        "cache_control": {"type": "ephemeral"},   # cache this
    },
    {
        "type": "text",
        "text": memory.dynamic_suffix(query),     # don't cache
    },
]
```

**Why this matters:** The stable prefix is typically 200–600 tokens. With prompt caching, you pay full price once per 5-minute cache TTL, then ~10% on subsequent calls. For an active conversation with 20 turns, this is a ~8× reduction in input token cost for the system prompt.

Use the `prefix` command to inspect what would be cached.

### 4. TTL Cleanup

On startup, `ProductionMemory` automatically prunes:
- Facts with confidence < 0.15 last seen more than 90 days ago
- Episodes older than 365 days
- Semantic index entries for pruned sessions

This keeps the stores bounded without manual intervention. Tune the thresholds in `ProductionMemory.__init__()`.

---

## Architecture

```
assistant.py                  ← CLI, chat loop, extractors, summarizers
memory.py                     ← PersistentFactStore, PersistentSemanticIndex,
                                 SessionManager, ProductionMemory

ProductionMemory
    ├── PersistentFactStore   (memory.db)
    │       facts table       entity + attribute + value + confidence
    │       episodes table    session summaries
    │       turns table       raw turn content per session
    │
    ├── PersistentSemanticIndex (semantic.json)
    │       list of SemanticEntry (session_id, turn_index, role, content, embedding)
    │
    └── SessionManager        (session.json)
            SessionState      session_id, started_at, turns[], turn_index
```

---

## System Prompt Assembly

```
get_system_prompt(query)
    │
    ├── stable_prefix()
    │       base_prompt
    │       + "## Previous sessions"   ← last 3 episode summaries from SQLite
    │       + "## Known facts"         ← facts with confidence ≥ 0.40 from SQLite
    │         (truncated to STABLE_BUDGET chars)
    │
    └── dynamic_suffix(query)
            semantic hits               ← top-3 similar past turns from JSON index
            (excluding turns in current buffer, truncated to SEMANTIC_BUDGET chars)
```

---

## Running the Assistant

```bash
# From the project root

# Mock mode — no API key, bag-of-words embeddings
uv run python memory/capstone/personal-assistant/assistant.py --mock

# Real mode — Claude API + sentence-transformers embeddings
ANTHROPIC_API_KEY=sk-... uv run python memory/capstone/personal-assistant/assistant.py --real

# Force a new session (don't resume the last one)
uv run python memory/capstone/personal-assistant/assistant.py --real --new

# Wipe all memory and start completely fresh
uv run python memory/capstone/personal-assistant/assistant.py --mock --clear

# Custom data directory (useful for testing multiple users)
uv run python memory/capstone/personal-assistant/assistant.py --mock \
    --data-dir /tmp/my-assistant
```

---

## Suggested Exercise Sequence

**Run 1 — build up memory:**
```
You: My name is Alice and I'm a software engineer.
You: I work at Acme Corp on project Orion.
You: Orion uses Redis and Kafka.
You: Bob is my colleague on the infrastructure team.
You: memory          ← inspect all four layers
You: prefix          ← see the cacheable system prompt portion
You: endsession      ← persist to episodic layer
You: quit
```

**Run 2 — experience persistence:**
```
(restart the assistant)
You: Do you remember my name?         ← episodic layer recalls session 1
You: What does Orion use?             ← structured KG answers
You: memory                           ← facts still there from run 1
You: prefix                           ← episodic + structured in stable prefix
```

**Run 3 — test conflict and forget:**
```
You: Actually I moved to Beta Corp.   ← confidence drop on employer
You: memory                           ← employer confidence reduced
You: forget user employer             ← manually remove
You: memory                           ← employer gone
```

**Run 4 — observe TTL (fast-forward):**
To see TTL in action, use a custom data dir, manually edit `memory.db`
to set `last_seen` to 100 days ago on a low-confidence fact, then restart.
The fact will be pruned on startup.

---

## Commands

| Command | What it does |
|---------|-------------|
| `memory` | Full memory state: all facts (including suppressed), all recent episodes |
| `prompt` | Full assembled system prompt (stable + dynamic) |
| `prefix` | Stable portion only — what would be sent to prompt cache |
| `forget <entity> <attr>` | Delete a specific fact |
| `endsession` | Summarise current session, save to episodic layer, start fresh |
| `stats` | One-line count of all layer sizes |
| `quit` / `exit` | Exit (session NOT auto-saved — call `endsession` first if needed) |

---

## Differences from Experiment 10

| Concern | Experiment 10 | This Capstone |
|---------|--------------|---------------|
| Storage | In-memory dicts | SQLite + JSON files |
| Persistence | Lost on exit | Survives restarts |
| Session resume | Not supported | 4-hour resume window |
| Prompt caching | Not structured for it | `stable_prefix()` / `dynamic_suffix()` split |
| TTL cleanup | Not implemented | Automatic on startup |
| Multi-session | Single session | Full episode history in SQLite |
| `forget` command | Not available | Deletes individual facts |

---

*Part of the [Memory Vertical](../../) · See also: [Layered Memory experiment](../../10-layered-memory/)*
