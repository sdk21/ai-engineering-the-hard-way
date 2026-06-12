# Lesson: Dynamic Tool Selection

**Vertical:** Tools | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [The Problem with Large Tool Sets](#1-the-problem-with-large-tool-sets)
2. [The Registry Pattern](#2-the-registry-pattern)
3. [Relevance Scoring](#3-relevance-scoring)
4. [Two-Stage Architecture](#4-two-stage-architecture)
5. [The Search-Tools Meta-Tool](#5-the-search-tools-meta-tool)
6. [Category Filtering](#6-category-filtering)
7. [Choosing top-k](#7-choosing-top-k)
8. [Tool Discovery During Execution](#8-tool-discovery-during-execution)
9. [Key Principles](#9-key-principles)
10. [In the Real World](#10-in-the-real-world)
11. [Running the Experiment](#11-running-the-experiment)

---

## 1. The Problem with Large Tool Sets

The basic function calling experiments pass all available tools to the model on every turn. This works fine with 2–5 tools. It breaks down at scale:

**Token cost:** Each tool schema is 50–150 tokens. 100 tools = 5,000–15,000 tokens per API call. On a long conversation with 50 turns, that's 250k–750k wasted tokens just for unused tool definitions.

**Performance degradation:** Studies show that model function calling accuracy drops significantly above ~20 tools. With 100+ tools, the model struggles to select the right one and may call the wrong tool or none at all.

**Attention dilution:** The model's attention is finite. Presenting 100 tool descriptions dilutes attention from the tools that actually matter for the current query.

The solution: don't send all tools — send only the relevant ones.

---

## 2. The Registry Pattern

A tool registry separates tool storage from tool execution:

```
┌─────────────────────────────────────────────────────┐
│ Tool Registry                                        │
│                                                      │
│  ToolDef(name, description, category, tags, schema, fn) │
│  ToolDef(...)                                        │
│  ... (30–1000 tools)                                 │
│                                                      │
│  search(query, top_k) → [ToolDef, ...]               │
│  get(name) → ToolDef                                 │
│  dispatch(name, inputs) → str                        │
└─────────────────────────────────────────────────────┘
         ↑                        ↑
   retrieval                 execution
   (per query)               (per tool call)
```

The registry holds all tools. Before each API call, you query it for the relevant subset. The model receives only those tools — but the registry can execute any tool by name when the model calls it.

---

## 3. Relevance Scoring

The simplest effective approach is TF-IDF cosine similarity between the user query and each tool's description + tags:

```python
def bow_similarity(query: str, doc: str, idf: dict) -> float:
    q_vec = tf_idf_vector(query, idf)
    d_vec = tf_idf_vector(doc, idf)
    return cosine_similarity(q_vec, d_vec)
```

**Better approach for production:** dense embeddings. Embed tool descriptions once at startup, embed the query at runtime, compute cosine similarity in vector space:

```python
# At startup (once)
tool_embeddings = {tool.name: embed(tool.description) for tool in catalogue}

# Per query
query_embedding = embed(user_query)
scores = {name: cosine(query_embedding, emb) for name, emb in tool_embeddings.items()}
top_k = sorted(scores, key=lambda n: -scores[n])[:k]
```

This experiment uses bag-of-words to avoid requiring an embedding model. In production, use sentence-transformers or an embedding API.

---

## 4. Two-Stage Architecture

```
Stage 1 — Retrieval (your code, before API call)
  User query → registry.search(query, top_k=5) → [selected_tools]
  Cost: ~1ms for BoW, ~10-50ms for embedding model
  Output: small set of relevant tool schemas

Stage 2 — Execution (standard agentic loop)
  API call with [selected_tools] → tool_use blocks → execute → tool_result
  Model operates on a focused, relevant tool set
  If it needs a tool not in the set → use search_tools meta-tool
```

The two stages are independent. You can swap the retrieval strategy without touching the execution logic, and vice versa.

---

## 5. The Search-Tools Meta-Tool

Always include one special tool: `search_tools`. This gives the model a way to discover tools it wasn't given in the initial selection:

```python
{
    "name": "search_tools",
    "description": (
        "Search the tool catalogue for tools relevant to a task. "
        "Use this when the initially provided tools don't cover what you need."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "category": {"type": "string"}
        },
        "required": ["query"]
    }
}
```

When the model calls `search_tools`, your code:
1. Runs another registry search
2. Returns the matching tool names and descriptions as a result
3. Adds those tools to the active schema set for subsequent turns

This creates a progressive discovery pattern: the model gets a starter set, and can expand it if needed.

---

## 6. Category Filtering

Before similarity scoring, filter by category when the query clearly belongs to one domain:

```python
def select_tools(query: str, top_k: int = 5) -> list[ToolDef]:
    # Coarse filter (fast, keyword-based)
    category = detect_category(query)
    if category:
        candidates = [t for t in catalogue if t.category == category]
    else:
        candidates = catalogue

    # Fine ranking (similarity-based)
    return rank_by_similarity(query, candidates, top_k=top_k)
```

Category detection can be as simple as checking for keywords:
```python
CATEGORY_SIGNALS = {
    "finance": ["stock", "price", "market", "invest", "earnings"],
    "weather": ["weather", "forecast", "temperature", "rain"],
    "travel": ["travel", "flight", "hotel", "visa", "trip"],
}
```

Category filtering reduces the search space by 5–6x before similarity scoring, making it faster and more accurate.

---

## 7. Choosing top-k

| top-k | Behavior |
|-------|---------|
| 3 | Very focused. Risk: missing needed tools. Good for narrow single-intent queries. |
| 5 | Balanced. Covers the most likely tools without adding noise. **Good default.** |
| 8–10 | Broad. Useful for multi-intent queries. Starts to dilute attention. |
| 15+ | Approaches "give model all tools" behavior. Use only with strong retrieval. |

**Adaptive top-k:** Use a lower k for simple queries (single entity, single action) and higher k for complex queries (multiple entities, multiple actions, ambiguous intent).

A rough heuristic:
- Single action query → k=3
- Multi-step query → k=5
- Open-ended/ambiguous → k=8

---

## 8. Tool Discovery During Execution

A common pattern: the model calls `search_tools`, gets results, then immediately calls one of the discovered tools — all in the same conversation. This means you need to update the active tool set mid-conversation:

```python
if block.name == "search_tools":
    result = search_tools_fn(block.input["query"])
    # Parse the result to extract tool names
    discovered = registry.search(block.input["query"], top_k=3)
    for tool in discovered:
        if tool.name not in active_tool_names:
            active_schemas.append(tool.to_schema())
            active_tool_names.add(tool.name)
```

The next API call includes the newly added tools, and the model can immediately call them.

---

## 9. Key Principles

> **Principle 1 — Never send all tools when you have more than ~15.**
> Past ~15 tools, model accuracy degrades. Dynamic selection keeps the model operating in its effective range regardless of catalogue size.

> **Principle 2 — Tool descriptions are search documents.**
> Good retrieval depends on good descriptions. Each tool's description should mention the problem it solves (not just what it does), the entities it works with, and the type of output it produces.

> **Principle 3 — Always include a meta search tool.**
> The initial selection will sometimes miss a needed tool. `search_tools` is the model's escape hatch — it lets the model expand its own tool set rather than giving up.

> **Principle 4 — Separate storage from execution.**
> The registry holds all tools; the API call receives a subset; the execution path handles any tool by name. This decoupling lets you scale the catalogue without changing the loop.

> **Principle 5 — Category filtering before similarity scoring is faster and often better.**
> If you can identify the domain (finance, travel, etc.) from the query, filter to that category first. This reduces noise and improves top-k precision.

---

## 10. In the Real World

**Salesforce Einstein**
Salesforce's AI platform has hundreds of CRM actions (create contact, update opportunity, schedule task, etc.). Dynamic tool selection filters to the relevant action set based on the current view and user intent.

**Microsoft Copilot Studio**
Microsoft's agent builder lets organizations define thousands of custom actions. The runtime selects relevant actions per query using embeddings over action descriptions — exactly the two-stage pattern described here.

**LangChain Tool Selection**
LangChain's `VectorStoreToolkit` wraps a vector store as a tool retriever. Given a query, it returns the most relevant tool objects. This is the retrieval stage of the two-stage architecture.

**Amazon Bedrock Agents**
Bedrock Agents support tool groups (called "action groups"). You can associate different action groups with different knowledge bases, and the runtime selects which groups to activate per request.

**Plugin Systems (ChatGPT Plugins)**
ChatGPT's plugin system selected relevant plugins before the deprecation based on user intent and plugin descriptions. This was dynamic tool selection at the plugin level — exactly the registry + retrieval architecture.

**Zapier Central**
Zapier's AI features use tool selection to present only the relevant Zapier integrations from their 5000+ app catalogue. The selection is based on the current workflow context and user query.

---

## 11. Running the Experiment

```bash
# From the project root

# Mock mode — see which tools are selected for different queries
uv run python tools/07-dynamic-tool-selection/demo.py --mock

# Real mode — full two-stage retrieval + execution
ANTHROPIC_API_KEY=sk-... uv run python tools/07-dynamic-tool-selection/demo.py --real

# Real mode with custom top-k
ANTHROPIC_API_KEY=sk-... uv run python tools/07-dynamic-tool-selection/demo.py --real --top-k 3
```

**Suggested queries:**
- `"What's the weather forecast for Paris this week?"` — weather category selected
- `"Search for Python packages for data visualization on GitHub."` — development + knowledge
- `"What do I need to know about traveling to Japan?"` — travel category
- `"What's the stock price for NVDA and what do analysts think?"` — finance category

**Suggested exercises:**
1. Replace `bow_similarity` with real sentence-transformer embeddings. Install `sentence-transformers` and use `SentenceTransformer("all-MiniLM-L6-v2")`.
2. Add 10 more tools to the catalogue and verify that relevance still works correctly.
3. Set `--top-k 2` and ask a multi-intent query. Observe how often the model calls `search_tools` to discover missing tools.
4. Implement `detect_category(query)` to filter by category before similarity scoring and measure the improvement in precision.

---

*Previous: [Human-in-the-Loop](../06-human-in-the-loop/) · Next: [Programmatic Tool Generation](../08-programmatic-tool-generation/)*
