# Lesson: Conversation Buffer Memory

**Vertical:** Memory | **Difficulty:** Beginner | **Status:** ✅ Ready

---

## Table of Contents

1. [The Core Problem: LLMs Are Stateless](#1-the-core-problem-llms-are-stateless)
2. [What Is a Conversation Buffer?](#2-what-is-a-conversation-buffer)
3. [How It Works — Step by Step](#3-how-it-works--step-by-step)
4. [The Message Format](#4-the-message-format)
5. [Why This Actually Works](#5-why-this-actually-works)
6. [Cost and Growth Model](#6-cost-and-growth-model)
7. [Failure Modes](#7-failure-modes)
8. [Key Principles](#8-key-principles)
9. [In the Real World](#9-in-the-real-world)
10. [Running the Experiment](#10-running-the-experiment)

---

## 1. The Core Problem: LLMs Are Stateless

Every call to an LLM API is completely independent. The model has no memory of your previous requests. Send the same API call twice in a row and it produces two unrelated responses — it has no idea the first one happened.

This is a fundamental architectural property, not a limitation that will be fixed. LLMs are stateless by design: they are pure functions that map a sequence of tokens to a probability distribution over the next token. There is no hidden state, no session, no persistent memory on the model side.

This creates an immediate problem for conversational applications:

```
Turn 1 — User: "My name is Alice."
         Model: "Nice to meet you, Alice!"

Turn 2 — User: "What's my name?"
         Model: "I don't know your name." ← wrong, but technically correct
```

The model "forgot" because the second API call contained no information about the first. From the model's perspective, Turn 2 was the entire conversation.

---

## 2. What Is a Conversation Buffer?

A **conversation buffer** is the simplest possible solution: maintain a list of every message exchanged, and send the entire list with every API call.

```python
history = []

# Turn 1
history.append({"role": "user", "content": "My name is Alice."})
response = api_call(messages=history)
history.append({"role": "assistant", "content": response})

# Turn 2
history.append({"role": "user", "content": "What's my name?"})
response = api_call(messages=history)  # model now sees all 3 messages
```

The model receives the full conversation history on every call, so it can refer back to anything said earlier.

---

## 3. How It Works — Step by Step

```
API Call 1:
  messages = [
    {role: user,      content: "My name is Alice."}
  ]
  → response: "Nice to meet you, Alice!"

API Call 2:
  messages = [
    {role: user,      content: "My name is Alice."},
    {role: assistant, content: "Nice to meet you, Alice!"},
    {role: user,      content: "What's my name?"}
  ]
  → response: "Your name is Alice."

API Call 3:
  messages = [
    {role: user,      content: "My name is Alice."},
    {role: assistant, content: "Nice to meet you, Alice!"},
    {role: user,      content: "What's my name?"},
    {role: assistant, content: "Your name is Alice."},
    {role: user,      content: "How many letters are in my name?"}
  ]
  → response: "Your name Alice has 5 letters."
```

Each turn, the buffer grows by two messages (user + assistant). Every new API call replays the entire conversation from the beginning.

---

## 4. The Message Format

The multi-turn message format is standardized across all major LLM APIs. Messages are a list of objects with two fields:

| Field | Values | Purpose |
|-------|--------|---------|
| `role` | `"user"`, `"assistant"`, `"system"` | Who sent this message |
| `content` | string | The message text |

The `system` role is special — it sets context and instructions that apply to the whole conversation. It is typically placed first and does not count as a conversational turn.

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user",      "content": "What did I just say?"},
]
```

The model is trained to understand this turn-taking structure and to treat earlier messages as prior context.

---

## 5. Why This Actually Works

The model was trained on data structured as multi-turn conversations. When it sees a list of `[user, assistant, user, assistant, ...]` messages, it has learned to:

- Treat earlier turns as established facts
- Refer back to prior content ("as I mentioned earlier…")
- Maintain persona consistency across turns
- Resolve pronouns and references across turn boundaries

The buffer pattern works so well precisely because it mirrors how the training data was structured. We're not tricking the model — we're giving it exactly the input format it was designed to consume.

---

## 6. Cost and Growth Model

The buffer has a critical scaling property: **cost grows linearly with conversation length**.

If each turn averages `t` tokens, then at turn `n` you're sending approximately `n × t` tokens per API call. The total tokens billed across an entire `n`-turn conversation is:

```
Total tokens ≈ t × (1 + 2 + 3 + ... + n) = t × n(n+1)/2  →  O(n²)
```

This means a 100-turn conversation costs roughly 50× more in tokens than a 10-turn conversation (not 10×). Long conversations become expensive fast.

Additionally, every LLM has a **context window** — a hard maximum on the total tokens it can process in one call (input + output). As of 2025 this ranges from ~16k tokens (older models) to 200k+ tokens (Claude, Gemini). Once your buffer exceeds the context window, API calls start failing.

---

## 7. Failure Modes

**Context window overflow**
The buffer grows without bound. Eventually it exceeds the model's maximum context length and the API returns an error. You must handle this — either by truncating, summarizing, or switching to a windowed strategy.

**Repetition and noise**
Early turns often contain small talk, corrections, and tangents that are no longer relevant. Including them costs tokens and can actually confuse the model by introducing noise into the context.

**Cost surprises**
Developers building chat apps often notice their costs grow much faster than expected. The quadratic total token cost is the culprit — the buffer pattern is convenient but expensive at scale.

**No persistence across sessions**
The in-memory buffer is lost when the process ends. Separate sessions have no shared memory. This is usually handled at the application layer (e.g., serialize to a database), not the buffer itself.

---

## 8. Key Principles

> **Principle 1 — Statefulness is your responsibility.**
> The LLM is stateless. Your application is responsible for maintaining and transmitting all conversational context.

> **Principle 2 — The model only knows what you send it.**
> If a fact isn't in the current message list, the model cannot know it. There are no hidden channels.

> **Principle 3 — More context is not always better.**
> Longer histories cost more, increase latency, and can introduce noise. Every memory architecture is a trade-off between completeness and cost.

> **Principle 4 — The buffer is the baseline.**
> Every more sophisticated memory technique (sliding window, summarization, vector retrieval) is solving a specific failure mode of the naive buffer. Understanding the buffer is prerequisite to understanding why anything else exists.

---

## 9. In the Real World

The conversation buffer pattern is the foundation of nearly every LLM-based chat product. Here is how real systems use it:

**ChatGPT (OpenAI)**
ChatGPT maintains a full conversation buffer per session and persists it to a database so you can resume conversations later. OpenAI's API accepts a `messages` array that is exactly a conversation buffer. ChatGPT's memory features layer on top of this baseline.

**Claude.ai (Anthropic)**
Claude's chat interface maintains the full conversation buffer within a session. Claude's 200k token context window means the buffer can hold very long conversations before hitting limits — this is a deliberate product decision to minimize the need for complex memory management.

**LangChain — `ConversationBufferMemory`**
LangChain has a class literally named `ConversationBufferMemory` that implements this exact pattern. It is listed as the simplest memory type in their documentation and is the default starting point for building chains with memory.

**LlamaIndex — Chat Engine**
LlamaIndex's `SimpleChatEngine` maintains a conversation buffer as its default memory strategy. Their documentation uses it as the baseline before introducing more sophisticated retrieval-based approaches.

**Vercel AI SDK**
The Vercel AI SDK's `useChat` hook maintains client-side conversation state as a message array and sends the full array on each request — a conversation buffer implemented in React state.

**Slack / Microsoft Teams bots**
Enterprise chat bots built on frameworks like Microsoft Bot Framework maintain a conversation buffer per thread, serialized to storage (CosmosDB, Redis) so the buffer survives process restarts.

---

## 10. Running the Experiment

```bash
# From the project root

# Mock mode — no API key needed, learn the structure
uv run python memory/01-conversation-buffer/demo.py --mock

# Real mode — calls Claude API
ANTHROPIC_API_KEY=sk-... uv run python memory/01-conversation-buffer/demo.py --real
```

**Suggested exercises:**
1. Start a conversation, tell the model your name, chat for a few turns, then ask "what's my name?" — confirm it remembers.
2. Type `stats` to see the token estimate grow with each turn.
3. Type `clear` to reset the buffer and ask again — observe it has forgotten.
4. Read `experiment.py` and trace exactly which messages are sent on each API call.

---

*Next experiment: [Sliding Window Memory](../02-sliding-window/) — what happens when the buffer gets too large.*
