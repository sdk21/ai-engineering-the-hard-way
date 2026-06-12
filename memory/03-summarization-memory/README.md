# Lesson: Summarization Memory

**Vertical:** Memory | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [The Problem with Dropping Messages](#1-the-problem-with-dropping-messages)
2. [What Is Summarization Memory?](#2-what-is-summarization-memory)
3. [Architecture](#3-architecture)
4. [How It Works — Step by Step](#4-how-it-works--step-by-step)
5. [The Summarization Prompt](#5-the-summarization-prompt)
6. [Rolling vs. Full Summarization](#6-rolling-vs-full-summarization)
7. [What Gets Lost in Summarization](#7-what-gets-lost-in-summarization)
8. [Triggering Strategy](#8-triggering-strategy)
9. [Cost Model](#9-cost-model)
10. [Failure Modes](#10-failure-modes)
11. [Key Principles](#11-key-principles)
12. [In the Real World](#12-in-the-real-world)
13. [Running the Experiment](#13-running-the-experiment)

---

## 1. The Problem with Dropping Messages

The sliding window (Experiment 02) solves context overflow by dropping old messages. This is simple and cheap — but the dropped information is gone forever. If the user mentioned their name, goals, or constraints in turn 3, and the window is 6 messages, turn 10 has no access to those facts at all.

For many applications, this is unacceptable:

- A personal assistant that forgets your name and preferences after a few exchanges
- A coding agent that forgets the requirements stated at the start of a long session
- A tutoring system that forgets what the student already understands

We need a strategy that **bounds context growth** (like the sliding window) while **preserving information** (unlike the sliding window). Summarization is that strategy.

---

## 2. What Is Summarization Memory?

Summarization memory compresses old conversation turns into a concise text summary rather than dropping them. The summary is injected back into the system prompt, so the model always has access to a condensed view of the full history.

```
Without summarization (sliding window):
  [turn 1] forgotten
  [turn 2] forgotten
  [turn 3] ← window start
  [turn 4]
  [turn 5]
  [turn 6] ← most recent

With summarization:
  [system: "Earlier: user is Alice, a Python engineer in Tokyo..."]  ← summary
  [turn 3]  ← buffer start (same window)
  [turn 4]
  [turn 5]
  [turn 6]  ← most recent
```

The key difference: the dropped turns are not lost — they are distilled. Their essential facts are preserved in the summary at a fraction of the token cost.

---

## 3. Architecture

```
┌─────────────────────────────────────────────┐
│  System Prompt                              │
│  ─────────────                              │
│  Base instructions                          │
│  + Summary of earlier conversation ◄────┐  │
└─────────────────────────────────────────│───┘
                                          │
┌─────────────────────────────────────────│───┐
│  Recent Buffer (fixed size)             │   │
│  ─────────────                          │   │
│  turn N-5 (user)                        │   │
│  turn N-4 (assistant)                   │   │
│  turn N-3 (user)          When overflow─┘   │
│  turn N-2 (assistant)     oldest msgs are   │
│  turn N-1 (user)          summarized +      │
│  turn N   (assistant) ◄── added to summary  │
└─────────────────────────────────────────────┘
```

Two memory regions:
- **Summary** (slow-growing, in system prompt) — compressed history, updated periodically
- **Buffer** (fixed-size) — full verbatim recent turns

On every API call, the model receives: `system(base + summary)` + `messages(buffer)`.

---

## 4. How It Works — Step by Step

```
Setup: buffer_size=4, summarize_after=2

Turns 1-4: buffer fills normally
  buffer: [U1, A1, U2, A2, U3, A3, U4, A4]   ← 8 messages (4 pairs)

Turn 5: buffer hits 8 + 2 = 10 → no trigger yet, just at limit

Turn 6: buffer = 12 messages → 12 > 4+2=6 → TRIGGER
  to_compress: [U1, A1, U2, A2]              ← oldest 4 messages
  buffer after: [U3, A3, U4, A4, U5, A5, U6, A6]
  ↓
  summarizer([U1,A1,U2,A2]) → "User introduced themselves as Alice, a Python engineer
                                in Tokyo. Asked about recursion basics."
  summary = above text
  buffer = [U3, A3, U4, A4, U5, A5, U6, A6]

Turn 7: buffer grows to 10 → trigger again
  to_compress: [U3, A3, U4, A4]
  summarizer(existing_summary, [U3,A3,U4,A4]) → updated summary
  buffer = [U5, A5, U6, A6, U7, A7]
```

The summary grows slowly — each compression adds a few sentences. The buffer stays constant. Total tokens sent per call is now `O(summary_size + buffer_size)` instead of `O(n)`.

---

## 5. The Summarization Prompt

The quality of the summary depends entirely on the summarization prompt. A good prompt:

1. **Passes the existing summary** — so the model can merge rather than restate
2. **Specifies what to preserve** — facts, names, decisions, constraints
3. **Specifies what to drop** — small talk, filler, redundant exchanges
4. **Asks for plain text output** — no formatting, no meta-commentary

```
You are maintaining a running summary of a conversation.

Existing summary:
{existing_summary}

New messages to incorporate:
{new_messages}

Update the summary to include all important facts, decisions, and context.
Merge naturally with the existing summary. Be concise — preserve facts,
names, goals, and decisions. Omit small talk and filler.

Return only the updated summary text.
```

The instruction to **merge** (not restate) is critical. Without it, the model tends to re-summarize the whole thing from scratch each time, which costs more tokens and loses the compounding effect.

---

## 6. Rolling vs. Full Summarization

**Rolling summarization** (this experiment) updates the summary incrementally, merging each batch of dropped messages into the existing summary:

```
Pass 1: summarize(turns 1-4)           → summary_v1
Pass 2: summarize(summary_v1, turns 5-8) → summary_v2
Pass 3: summarize(summary_v2, turns 9-12) → summary_v3
```

**Full re-summarization** discards the old summary and re-summarizes all dropped turns from scratch on each pass:

```
Pass 1: summarize(turns 1-4)       → summary_v1
Pass 2: summarize(turns 1-8)       → summary_v2  (re-processes turns 1-4)
Pass 3: summarize(turns 1-12)      → summary_v3  (re-processes all prior turns)
```

| Approach | Quality | Cost |
|----------|---------|------|
| Rolling | Lower (accumulates compression errors) | O(batch_size) per pass |
| Full re-summarization | Higher (fresh view each time) | O(n) per pass — same as no memory |

Rolling is the only viable option for long conversations. Full re-summarization negates the purpose. The quality tradeoff is managed through batch size: smaller batches compress less aggressively and lose less information.

---

## 7. What Gets Lost in Summarization

Summarization is inherently lossy. Understanding what is lost helps you design around it:

**Exact wording** — Summaries paraphrase. If the model's response to a legal or medical question depends on precise phrasing, summarization will corrupt it.

**Implicit context** — "The bug I mentioned earlier" becomes "the user mentioned a bug." The referent is gone.

**Tone and emotional context** — "The user seemed frustrated about X" is rarely captured in a factual summary.

**Negative facts** — Things the user said they *don't* want ("don't use TypeScript") are easily dropped from summaries because they don't read as facts.

**Temporal ordering** — The summary may say "user asked about A and B" without preserving which came first, losing causal relationships.

**Mitigations:**
- Larger buffer sizes (compress less frequently)
- Prompts that explicitly ask to preserve constraints and negations
- Structured summaries (key-value format: `name: Alice`, `preferences: no TypeScript`)
- Combining with vector memory for fact retrieval when precision matters

---

## 8. Triggering Strategy

When should summarization trigger? Options:

**Message count threshold** (this experiment) — Simplest. Trigger when the buffer exceeds N messages. Predictable, easy to reason about.

**Token count threshold** — Trigger when the buffer exceeds N tokens. More precise cost control, requires token counting.

**Time-based** — Trigger every N minutes in long-running sessions. Natural for agents with long idle periods.

**Semantic shift detection** — Summarize when a topic change is detected (embedding distance between recent turns and older turns exceeds a threshold). Smart but complex.

**Explicit user signals** — Summarize when the user says "let's move on" or starts a new topic. Works well in structured workflows.

The message count threshold is the right starting point. Upgrade to token-based when you need precise cost control.

---

## 9. Cost Model

At steady state with a buffer of `B` messages and a summary of `S` tokens:

- **Per API call:** `S + B × t` tokens (summary + buffer), where `t` = avg tokens per message
- **Per summarization pass:** `S + batch_size × t` input tokens → small summary output

Compare to:
- Conversation buffer (Experiment 01): `n × t` per call — grows without bound
- Sliding window (Experiment 02): `B × t` per call — no summary cost, but no history

Summarization memory is slightly more expensive than pure sliding window (due to the summary tokens) but dramatically cheaper than the full buffer for long conversations, while preserving far more information than the sliding window.

The summary size `S` grows slowly — each compression adds ~50–200 tokens to the summary for a batch of messages. After many summarizations, the summary itself may need to be summarized (meta-summarization). This is the long-term cost management challenge.

---

## 10. Failure Modes

**Summary drift** — Rolling summarization accumulates small errors with each pass. After 20 summarization rounds, the summary may subtly misrepresent the early conversation. Facts get paraphrased into slightly different facts.

**Summary growth without bound** — If each compression adds more to the summary than it compresses, the summary eventually becomes too large. Add a max-summary-length constraint and trigger meta-summarization.

**Lost constraints and negations** — "Please don't recommend TypeScript" is likely to disappear after a few summarization passes because it reads as a negative instruction rather than a fact. Mitigate with structured prompts that explicitly ask for constraints.

**Summarization latency** — Summarization requires an extra LLM call. If triggered mid-conversation, it adds visible latency to the user's turn. Mitigate by running summarization asynchronously or during low-activity periods.

**Poor summarizer quality** — The summary is only as good as the model doing the summarizing. Using a weak summarizer (or a bad prompt) produces summaries that lose important facts, defeating the purpose.

---

## 11. Key Principles

> **Principle 1 — Summarization trades precision for longevity.**
> You keep the facts but lose the exact words, tone, and nuance. This is acceptable for many applications and unacceptable for others. Know which you are building.

> **Principle 2 — The summarization prompt is as important as the memory architecture.**
> A poor prompt that drops constraints, negations, or goals produces a memory that actively misleads the model. Invest in prompt engineering for the summarizer.

> **Principle 3 — Rolling summarization accumulates errors — bound the summary size.**
> Every compression pass introduces small inaccuracies. Don't let the summary grow forever. Trigger meta-summarization or enforce a max token budget for the summary.

> **Principle 4 — Combine with a recent buffer, never replace it.**
> The summary alone is not enough — recent turns need full verbatim context. Summarization memory always has two layers: compressed history + verbatim recent turns.

> **Principle 5 — Async summarization avoids user-visible latency.**
> Summarization is a background concern. In production, trigger it asynchronously after the user's turn, not blocking their next message.

---

## 12. In the Real World

**ChatGPT — Memory feature**
OpenAI's Memory feature for ChatGPT is a form of summarization memory. When you have a long conversation or when you explicitly tell it to remember something, it extracts facts and stores them as a persistent summary attached to your profile. Future conversations are injected with this summary. The extraction step is LLM-as-summarizer applied to conversation history.

**Claude Projects**
Anthropic's Projects feature lets you store persistent context (instructions, facts, documents) that is prepended to every conversation in that project. While not automatic summarization, the pattern is identical: a system-level summary injected before the conversation buffer. Users manually curate this summary; automated summarization systems do it programmatically.

**LangChain — `ConversationSummaryMemory`**
LangChain has a class literally named `ConversationSummaryMemory` which implements rolling summarization. It takes a `llm` argument for the summarizer and maintains a running `buffer` string. The class is documented as "good for longer conversations where keeping the full message history would be too expensive."

**LangChain — `ConversationSummaryBufferMemory`**
A hybrid: maintains full verbatim messages up to a token limit, then switches to summarizing older messages. This is exactly the two-layer architecture (buffer + summary) described in this experiment. It is LangChain's recommended default for production chat applications.

**MemGPT / Letta**
MemGPT (now Letta) introduced a paged memory architecture for LLMs analogous to virtual memory in operating systems. The "main context" (buffer) is limited; a background process summarizes and compresses older content into "archival storage." Summarization is the core mechanism for moving content from hot memory to cold storage.

**Mem0**
Mem0 is an open-source memory layer for AI applications. It extracts structured facts from conversation turns using an LLM and stores them in a searchable memory store. At retrieval time, relevant memories are injected into the system prompt — a hybrid of summarization and retrieval-based memory.

**Microsoft Copilot — conversation context**
Microsoft's enterprise Copilot products maintain conversation context across long multi-turn interactions by periodically summarizing older turns. The summarization happens server-side, invisible to the user, and the summary is injected into subsequent model calls. This is the production-scale deployment of the pattern described in this experiment.

---

## 13. Running the Experiment

```bash
# From the project root

# Mock mode — observe summarization with fake LLM
uv run python memory/03-summarization-memory/demo.py --mock

# Tiny buffer to trigger summarization quickly
uv run python memory/03-summarization-memory/demo.py --mock --buffer 4

# Real mode — Claude summarizes the conversation as it grows
ANTHROPIC_API_KEY=sk-... uv run python memory/03-summarization-memory/demo.py --real
```

**Suggested exercise sequence:**

Start with `--mock --buffer 4` and run this exact conversation:
1. `"My name is Alice and I'm a software engineer."`
2. `"I live in Tokyo."`
3. `"My favourite language is Python."`
4. `"Tell me a random fact."` (filler)
5. `"Tell me another fact."` (filler)
6. Type `summary` — inspect what was captured
7. `"What's my name?"` — does it survive?
8. `"Where do I live?"` — does it survive?
9. Type `stats` — see token estimate vs. a full buffer

Then repeat with `--real` and compare the quality of the LLM-generated summary against the mock's naive extraction.

---

*Previous: [Sliding Window](../02-sliding-window/) | Next: [Vector Memory](../04-vector-memory/) (coming soon)*
