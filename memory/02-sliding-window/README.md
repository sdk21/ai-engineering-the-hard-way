# Lesson: Sliding Window Memory

**Vertical:** Memory | **Difficulty:** Beginner–Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [The Problem with Unbounded Buffers](#1-the-problem-with-unbounded-buffers)
2. [What Is a Sliding Window?](#2-what-is-a-sliding-window)
3. [How It Works — Step by Step](#3-how-it-works--step-by-step)
4. [Choosing the Window Size](#4-choosing-the-window-size)
5. [The Forgetting Problem](#5-the-forgetting-problem)
6. [Variation: Pinned Summary](#6-variation-pinned-summary)
7. [Variation: Token-Based Window](#7-variation-token-based-window)
8. [Cost Model](#8-cost-model)
9. [Failure Modes](#9-failure-modes)
10. [Key Principles](#10-key-principles)
11. [In the Real World](#11-in-the-real-world)
12. [Running the Experiment](#12-running-the-experiment)

---

## 1. The Problem with Unbounded Buffers

The conversation buffer experiment showed that replaying full history works — but it scales badly. Token cost grows as O(n²) over the life of a conversation, and eventually you hit the model's context window limit and the API starts returning errors.

For short conversations this is fine. But consider:

- A customer support bot handling 200-turn sessions
- A coding assistant open all day
- An agent running thousands of steps autonomously

In these cases, an unbounded buffer is either prohibitively expensive, technically broken (context overflow), or both. We need a way to bound the memory footprint.

---

## 2. What Is a Sliding Window?

A **sliding window** keeps only the most recent `k` messages in the buffer. When the buffer exceeds `k` messages, the oldest messages are dropped.

```
Window size k = 4

After turn 1:  [U1, A1]
After turn 2:  [U1, A1, U2, A2]
After turn 3:  [U1, A1, U2, A2, U3, A3] → drop U1, A1 → [U2, A2, U3, A3]
After turn 4:  [U2, A2, U3, A3, U4, A4] → drop U2, A2 → [U3, A3, U4, A4]
```

The buffer size is now constant at `k` messages regardless of how long the conversation runs. Token cost and latency are bounded.

---

## 3. How It Works — Step by Step

```
Turn 1 (window=4):
  sent: [user: "Hi, I'm Alice"]
  kept: [user: "Hi, I'm Alice", asst: "Hello Alice!"]

Turn 2:
  sent: [user: "Hi, I'm Alice", asst: "Hello Alice!", user: "What's the capital of France?"]
  kept: [user: "Hi, I'm Alice", asst: "Hello Alice!", user: "...", asst: "Paris."]

Turn 3:
  sent: [all 4 messages + new user message] → 5 messages, exceeds window
  DROP: [user: "Hi, I'm Alice", asst: "Hello Alice!"]
  kept: [user: "What's the capital...", asst: "Paris.", user: "How far is it from London?", asst: "About 340km."]

Turn 4:
  The model no longer sees "Hi, I'm Alice" — it has been forgotten.
  If asked "What's my name?" it will say "I don't know."
```

This is the fundamental trade-off: **bounded cost, bounded memory**.

---

## 4. Choosing the Window Size

Window size is a design parameter with real consequences:

| Window Size | Effect |
|------------|--------|
| Very small (2–4) | Minimal cost, aggressive forgetting, suitable for stateless Q&A |
| Medium (8–16) | Retains recent task context, forgets older background |
| Large (32–64) | Near-full context for typical conversations, higher cost |
| Full buffer | No forgetting, unbounded cost (equivalent to Experiment 01) |

The right size depends on your use case. A single-question assistant needs almost no window. A multi-step coding task needs enough window to remember the code being discussed. A long-running autonomous agent may need a completely different memory strategy altogether.

---

## 5. The Forgetting Problem

Sliding windows introduce **recency bias**: the model only knows what happened recently. This causes specific failure patterns:

**Identity forgetting** — The user said their name 20 turns ago. The model no longer knows it.

**Goal drift** — The user stated their objective at the start of a long session. As turns accumulate, that goal falls out of the window and the model loses sight of what it was trying to accomplish.

**Broken references** — The user refers to "the document you analyzed earlier." If that analysis happened outside the window, the model has no idea what they mean.

**Instruction loss** — Important constraints ("always respond in Spanish", "never suggest legal advice") stated early in the session may fall out of the window.

These aren't bugs — they're the predictable consequence of bounded memory. The key is to know they will happen and design your system accordingly.

---

## 6. Variation: Pinned Summary

One mitigation is to **summarize dropped messages** and inject the summary back into the system prompt before it is lost.

```
When messages fall out of the window:
  1. Summarize them: "User introduced themselves as Alice, asked about Paris."
  2. Prepend to system prompt: "Earlier in this conversation: [summary]"
  3. Drop the original messages as normal
```

This is a cheap way to preserve high-level facts at low token cost. The summary is lossy — detail is sacrificed — but key facts (names, goals, decisions) can be preserved.

The challenge is that summarization itself requires an LLM call (or a very good heuristic), and deciding what is worth summarizing is non-trivial.

---

## 7. Variation: Token-Based Window

Counting messages is a coarse proxy. A better approach is to count **tokens** and trim until the buffer fits within a target budget.

```python
MAX_TOKENS = 4096

while count_tokens(buffer) > MAX_TOKENS:
    buffer.pop(0)   # drop oldest message
```

This gives precise control over cost regardless of message length variance. Long messages (e.g., a pasted document) are treated the same as short ones — they cost what they cost, and trimming happens when the budget is exceeded.

Token counting requires a tokenizer (or an approximation like `len(text) // 4`). OpenAI's `tiktoken` and Anthropic's API both provide exact token counts.

---

## 8. Cost Model

With a fixed window of `k` messages averaging `t` tokens each, every API call costs approximately `k × t` tokens. This is **O(1)** per call — constant regardless of conversation length.

Compare to the unbounded buffer which was O(n) per call and O(n²) total. The sliding window reduces total token spend from quadratic to linear: `n` turns × `k × t` tokens = `O(n)` total.

The tradeoff: you pay a fixed `k × t` overhead even on turn 1 (when you could have just sent 1 message). The window is both a ceiling and a floor.

---

## 9. Failure Modes

**Off-by-one in turn pairs**
Messages naturally come in user/assistant pairs. If you drop an odd number of messages, you can end up with a window that starts with an `assistant` message, which confuses most models. Always drop complete pairs (2 messages at a time).

**Important context always near the beginning**
System instructions, user identity, and task goals are often stated early. They are the first things to fall out of the window — exactly the opposite of what you want. Mitigation: pin critical messages so they're never dropped.

**Summarization quality degrades at scale**
Chained summaries-of-summaries lose fidelity rapidly. After many rounds of compression, the summary becomes vague and unhelpful.

**Window too small for the task**
Some tasks inherently require long context: editing a long document, debugging a complex codebase, multi-step planning. No amount of clever windowing solves a task that fundamentally requires more context than the window allows.

---

## 10. Key Principles

> **Principle 1 — Forgetting is a feature, not just a bug.**
> Bounded memory forces you to think explicitly about what matters. Systems with unlimited memory often degrade in quality as noise accumulates.

> **Principle 2 — Recency bias is inevitable with a window.**
> Design around it: pin important context, summarize early facts, or use retrieval for distant memories.

> **Principle 3 — Match window size to task requirements.**
> There is no universal right answer. Profile your actual conversations to find the minimum window that preserves task-relevant context.

> **Principle 4 — The sliding window is a building block, not a final answer.**
> Production systems typically combine windowing with summarization, vector retrieval, or structured memory stores to get both bounded cost and high recall.

---

## 11. In the Real World

**OpenAI ChatGPT — context management**
ChatGPT silently truncates conversations that approach the model's context limit, keeping the most recent messages. This is a sliding window under the hood. Users notice this when very old parts of a long conversation stop being referenced correctly.

**LangChain — `ConversationBufferWindowMemory`**
LangChain provides `ConversationBufferWindowMemory(k=5)` which implements exactly this pattern. It is listed as the next step after `ConversationBufferMemory` in their memory documentation, motivated by the same cost concerns described here.

**LangChain — `ConversationTokenBufferMemory`**
A token-based variant of the window: `ConversationTokenBufferMemory(max_token_limit=2000)`. Rather than counting messages, it counts tokens and trims from the oldest end. This is the production-grade version of the token-based window variation described above.

**Anthropic Claude — system prompts with summaries**
Anthropic's own documentation on long-context use cases recommends a pattern of periodically summarizing conversation history and injecting the summary into the system prompt — exactly the "Pinned Summary" variation described in this experiment.

**Microsoft AutoGen**
AutoGen's multi-agent conversations use message history truncation strategies to keep each agent's context within bounds during long orchestration runs. Individual agents receive windowed views of the shared conversation.

**Amazon Lex / Dialogflow**
Enterprise conversational AI platforms typically maintain a short session context (often 5–10 turns) as their built-in memory — a sliding window with session expiry. They supplement it with explicit slot-filling (structured extraction) rather than relying on the LLM to recall arbitrary facts.

**Cursor / GitHub Copilot Chat**
IDE AI assistants use a form of sliding window scoped to the current file and recent edits. They can't maintain infinite conversation history, so they prioritize the most recently touched code and recent chat turns — recency bias applied to a coding context.

---

## 12. Running the Experiment

```bash
# From the project root

# Default window size (4 messages)
uv run python memory/02-sliding-window/demo.py --mock

# Tiny window — watch forgetting happen quickly
uv run python memory/02-sliding-window/demo.py --mock --window 2

# Real mode
ANTHROPIC_API_KEY=sk-... uv run python memory/02-sliding-window/demo.py --real --window 6
```

**Suggested exercises:**
1. Set `--window 2`. Say your name. Chat for 3 more turns. Ask "what's my name?" — observe forgetting.
2. Notice the `[window=X/Y, dropped=Z]` stats printed after each turn.
3. Compare `--window 2` vs `--window 8` on the same conversation — feel the difference in recall.
4. Read `experiment.py` and find where messages are trimmed. What happens if a message is dropped mid-pair?

---

*Previous: [Conversation Buffer](../01-conversation-buffer/) | Next: Summarization Memory (coming soon)*
