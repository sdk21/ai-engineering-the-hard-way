# Lesson: Tool Chaining

**Vertical:** Tools | **Difficulty:** Beginner–Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [What Is Tool Chaining?](#1-what-is-tool-chaining)
2. [Why Chains Form Naturally](#2-why-chains-form-naturally)
3. [The Dependency Graph](#3-the-dependency-graph)
4. [How the Model Manages the Chain](#4-how-the-model-manages-the-chain)
5. [Designing Tools for Chaining](#5-designing-tools-for-chaining)
6. [Conversation State Is the Chain State](#6-conversation-state-is-the-chain-state)
7. [Chain Termination and Safety](#7-chain-termination-and-safety)
8. [Failure Modes in Chains](#8-failure-modes-in-chains)
9. [Key Principles](#9-key-principles)
10. [In the Real World](#10-in-the-real-world)
11. [Running the Experiment](#11-running-the-experiment)

---

## 1. What Is Tool Chaining?

Tool chaining is what happens when the output of one tool call becomes an input (or informs the arguments) of the next. It is not a special API feature — it is a natural consequence of the agentic loop plus dependent tool designs.

**The chain in this experiment:**

```
look_up_flight(origin, destination)
    → flight_id, price, duration

get_flight_details(flight_id)           ← uses flight_id from step 1
    → airline, departure, seat_options, base_price

check_seat_availability(flight_id, seat_class)   ← uses flight_id + seat_options from step 2
    → seats_available, status

calculate_total_cost(base_price, seat_class, n)  ← uses base_price from step 2
    → total_cost
```

The user asks: *"What will it cost to fly NYC→LAX in business for 2?"*
The model issues 4 tool calls, each depending on earlier results, before answering.

---

## 2. Why Chains Form Naturally

The model reasons over the conversation history at each step. After calling `look_up_flight` and receiving a flight ID, the model sees:

```
Assistant turn 1: [tool_use: look_up_flight → {flight_id: "F001", price: 320}]
User turn 2:      [tool_result: flight_id: F001, base_price: 320, ...]
```

The next tool call — `get_flight_details(flight_id="F001")` — requires no special framework. The model simply reads the flight ID from the conversation context and uses it. The chain forms because:

1. The full conversation (including all tool results) is sent with every API call
2. The model can read any prior result and reference it in the next tool call
3. The model keeps calling tools until it has everything it needs to answer

---

## 3. The Dependency Graph

Not all tool chains are linear. Some tasks create tree-shaped or fan-in structures:

**Linear chain (this experiment):**
```
A → B → C → D → answer
```

**Parallel fan-out (next experiment):**
```
       ┌─ B ─┐
A ────→│     ├─→ answer
       └─ C ─┘
```

**Mixed (realistic agents):**
```
A ─→ B ─→ D ─┐
              ├─→ answer
A ─→ C ──────┘
```

When tools are independent, the model may issue multiple `tool_use` blocks in a single response — parallel calls. When one tool depends on another's output, the model issues them sequentially across turns. The model infers this dependency from your tool descriptions.

---

## 4. How the Model Manages the Chain

The model does not have an explicit plan or chain graph. It reasons step by step:

1. **After receiving the user's message:** "I need a flight ID before I can get details. I'll call `look_up_flight` first."
2. **After receiving the flight list:** "I have flight IDs. The user wants business class — I should check if F001 offers that. I'll call `get_flight_details`."
3. **After receiving flight details:** "F001 offers business class. Let me confirm seats are available before quoting a price."
4. **After confirming availability:** "Three business seats available. Now I can calculate the final cost."
5. **After receiving the cost:** "I have everything. I'll give the user the answer."

This is emergent chain reasoning — no explicit chain graph, no framework, just the model reading context and deciding what to do next.

---

## 5. Designing Tools for Chaining

For chaining to work, your tool designs must make dependencies explicit in descriptions.

**Good — describes inputs clearly and what downstream tools need:**
```python
{
    "name": "look_up_flight",
    "description": (
        "Search for available flights. Returns flight IDs and prices. "
        "Use the returned flight_id with get_flight_details for more info."
    ),
}
```

**Bad — no hint that output flows to other tools:**
```python
{
    "name": "look_up_flight",
    "description": "Get flight info.",
}
```

The model uses descriptions to reason about the tool graph. Explicit cross-references ("use the returned X with tool Y") help the model plan the chain correctly.

**Return consistent IDs.** If your tools form a chain, the ID that links them (here, `flight_id`) must appear in the output of the upstream tool and be accepted by the downstream tool. Inconsistent naming causes the model to hallucinate IDs or give up.

---

## 6. Conversation State Is the Chain State

In a chained tool-use scenario, the conversation messages array IS the chain state:

```python
messages = [
    {"role": "user",      "content": "NYC to LAX business for 2"},
    {"role": "assistant", "content": [tool_use: look_up_flight]},
    {"role": "user",      "content": [tool_result: flight_id=F001, price=320]},
    {"role": "assistant", "content": [tool_use: get_flight_details(F001)]},
    {"role": "user",      "content": [tool_result: airline=SkyLine, base_price=320, ...]},
    # ...
]
```

The model has access to the entire prior chain at every step. This means:
- No state variable needed in your code — the chain state lives in messages
- You can inspect the full chain at any point by printing messages
- Long chains accumulate tokens — a 10-step chain with large results can get expensive

---

## 7. Chain Termination and Safety

**How chains end:** The model stops calling tools when it has enough information to answer the user. It then produces a final text response with `stop_reason == "end_turn"`.

**Infinite loop risk:** A buggy or unhelpful tool that always returns errors can trap the model in a loop. Always add a max-iterations guard:

```python
MAX_ITERATIONS = 10
iterations = 0

while True:
    if iterations >= MAX_ITERATIONS:
        raise RuntimeError("Tool loop exceeded max iterations")
    iterations += 1
    # ...
```

**Premature termination:** If a tool returns an error early in the chain, the model may abandon the chain and answer with "I couldn't find that information." Make your error messages actionable — tell the model what to try instead.

---

## 8. Failure Modes in Chains

**ID propagation failure**
The model passes the wrong ID to a downstream tool. Usually caused by ambiguous output formats (multiple IDs in the result, inconsistent field names).
*Fix:* Return exactly one ID per result when downstream tools need it. Name it consistently.

**Premature chain termination**
The model stops early and answers from partial information. Usually caused by a tool error that looks like a dead end.
*Fix:* Return errors as informative strings that suggest alternatives: `"No business seats on F001. Try economy or a different flight."

**Over-calling**
The model calls a tool again unnecessarily because it "forgot" the earlier result. Usually caused by a very long conversation where early results are far from the current position.
*Fix:* Keep tool result strings concise. Summaries beat raw JSON.

**Wrong chain order**
The model tries `get_flight_details` before `look_up_flight`. Usually caused by weak descriptions that don't indicate ordering.
*Fix:* Use phrases like "you must call X first" in descriptions for strict dependencies.

---

## 9. Key Principles

> **Principle 1 — Chain state lives in the conversation, not in your code.**
> Every prior tool result is available to the model via message history. You don't need explicit state variables to pass data between tool calls.

> **Principle 2 — Descriptions carry the dependency graph.**
> The model infers ordering from your descriptions. "Use the flight_id returned by look_up_flight" is a hard dependency made visible to the model. Write it explicitly.

> **Principle 3 — Each tool should do exactly one thing.**
> Avoid combining look-up + detail-fetch + availability-check into one tool. Small, composable tools give the model flexibility to chain them appropriately. They're also easier to test and maintain.

> **Principle 4 — Long chains are expensive and brittle.**
> Every step adds latency and tokens. If a task reliably requires a 5-step chain, consider merging some steps into a single tool. Optimize the common case.

> **Principle 5 — Always cap chain depth.**
> A max-iterations limit on your agentic loop is non-optional in production. Models can loop. Loops cost money and time.

---

## 10. In the Real World

**Stripe / Payment APIs**
A payment agent chains tool calls: `create_customer` → `attach_payment_method(customer_id)` → `create_invoice(customer_id)` → `finalize_invoice(invoice_id)`. Each step depends on IDs from the previous.

**GitHub Agents**
Code automation chains: `search_repository` → `read_file(path)` → `create_pull_request(base, head, diff)`. Reading the right file requires knowing which repo and path first.

**LangChain SQL Agent**
LangChain's SQL agent chains: `list_tables` → `get_table_schema(table_name)` → `execute_query(sql)`. The schema must be known before a valid query can be written.

**OpenAI Function Calling in GPT**
Any multi-step form completion, booking flow, or wizard UI built on function calling is a tool chain. Each form step populates fields that subsequent steps require.

**Computer Use (Claude, Operator)**
Screen automation chains hundreds of tool calls: `screenshot` → `click(coordinates)` → `type(text)` → `screenshot` → `read_element(...)`. Each screenshot informs the next interaction.

---

## 11. Running the Experiment

```bash
# From the project root

# Mock mode — see the 4-step chain without an API key
uv run python tools/02-tool-chaining/demo.py --mock

# Real mode — watch Claude reason through the chain
ANTHROPIC_API_KEY=sk-... uv run python tools/02-tool-chaining/demo.py --real
```

**Suggested queries:**
- `"I want to fly from NYC to LAX in business class for 2 people. What will it cost?"` — full 4-step chain
- `"Find me a flight from SFO to ORD and check first class availability."` — 3-step chain
- `"What's the cheapest NYC to LAX option with available economy seats?"` — chain with selection logic

**Suggested exercises:**
1. Add a 5th tool: `reserve_seat(flight_id, seat_class, passenger_name) → confirmation_code`. Observe how the model extends the chain.
2. Make `check_seat_availability` return an error for a sold-out flight and observe how the model backtracks to try another flight.
3. Remove the dependency hints from `get_flight_details` description and observe what changes.
4. Add a `max_iterations=6` guard to `real_chain()` and verify it fires correctly.

---

*Previous: [Basic Function Calling](../01-basic-function-calling/) · Next: [Parallel Tool Calls](../03-parallel-tool-calls/)*
