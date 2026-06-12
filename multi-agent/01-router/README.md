# Lesson: Agent Router

**Vertical:** Multi-Agent | **Difficulty:** Beginner | **Status:** 🔜 Coming Soon

---

## Table of Contents

1. [The Problem: One Model Can't Do Everything Well](#1-the-problem-one-model-cant-do-everything-well)
2. [What Is an Agent Router?](#2-what-is-an-agent-router)
3. [The Orchestrator / Subagent Pattern](#3-the-orchestrator--subagent-pattern)
4. [Routing Strategies](#4-routing-strategies)
5. [Rule-Based Routing](#5-rule-based-routing)
6. [LLM-Based Routing](#6-llm-based-routing)
7. [Embedding-Based Routing](#7-embedding-based-routing)
8. [Designing Subagents](#8-designing-subagents)
9. [Failure Modes](#9-failure-modes)
10. [Key Principles](#10-key-principles)
11. [In the Real World](#11-in-the-real-world)
12. [Running the Experiment](#12-running-the-experiment)

---

## 1. The Problem: One Model Can't Do Everything Well

A single general-purpose LLM handles diverse tasks adequately. But "adequately" is often not good enough for production systems. Consider a customer support platform that needs to:

- Answer questions about billing (requires access to account data)
- Troubleshoot technical issues (requires deep product knowledge + tool use)
- Handle refund requests (requires policy knowledge + action execution)
- Route complex complaints to humans (requires escalation logic)

You could handle all of these in a single agent with a massive system prompt — but that agent would be mediocre at all of them and excellent at none. The system prompt would be so bloated it degrades performance. The tools would be cluttered. The instructions would conflict.

The solution is specialization: build multiple focused agents, each excellent at one thing, and route incoming requests to the right one.

---

## 2. What Is an Agent Router?

An agent router is a component that classifies an incoming request and dispatches it to the appropriate specialized agent or handler.

```
User message
     │
     ▼
  [Router]  ← classifies intent
     │
     ├── billing_agent      → specialized for account/billing questions
     ├── technical_agent    → specialized for product troubleshooting
     ├── refund_agent       → specialized for refund policy + execution
     └── human_escalation   → hands off to a human agent
```

The router itself is typically lightweight. The heavy work is done by the subagents.

---

## 3. The Orchestrator / Subagent Pattern

The router is the simplest instance of the **orchestrator / subagent** pattern, which is the foundational architecture of multi-agent systems:

- **Orchestrator** — Understands the high-level task, decomposes it, and delegates subtasks to subagents. It doesn't necessarily execute tasks itself.
- **Subagent** — Specialized in a narrow domain. Receives a clear task from the orchestrator, executes it using its own tools and context, and returns a result.

```
Orchestrator
  ├── Receives: "Research quantum computing and write a blog post about it"
  ├── Decomposes: [research task] + [writing task]
  ├── Delegates: research_agent ← "Find key facts about quantum computing"
  ├── Delegates: writing_agent ← "Write a blog post using these facts: [facts]"
  └── Returns: assembled blog post
```

A router is a degenerate orchestrator that doesn't decompose tasks — it just picks which subagent handles the whole request.

---

## 4. Routing Strategies

There are three main approaches to routing, each with different accuracy/cost/complexity tradeoffs:

| Strategy | Accuracy | Cost | Latency | Complexity |
|----------|---------|------|---------|-----------|
| Rule-based | Low–Medium | None | Near-zero | Low |
| LLM-based | High | Medium | 200–500ms | Low–Medium |
| Embedding-based | Medium–High | Low | 10–50ms | Medium |

The right choice depends on the number of routes, the variability of inputs, and latency requirements.

---

## 5. Rule-Based Routing

The simplest router uses keywords, regex, or structured metadata:

```python
def route(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["invoice", "billing", "charge", "payment"]):
        return "billing_agent"
    if any(w in msg for w in ["broken", "not working", "error", "bug"]):
        return "technical_agent"
    if any(w in msg for w in ["refund", "return", "cancel"]):
        return "refund_agent"
    return "general_agent"
```

**Pros:** Zero latency, zero cost, fully deterministic, easy to audit.

**Cons:** Brittle. Misses paraphrases ("I was overcharged" doesn't contain "billing"). Requires ongoing maintenance as language patterns evolve. Falls apart for ambiguous inputs.

Rule-based routing works well when:
- Categories are well-separated with clear vocabulary
- Volume is high and even small misrouting costs matter
- Routing happens before an LLM call, and you can't afford the extra latency

---

## 6. LLM-Based Routing

Use a (small, fast) model to classify the intent:

```python
ROUTER_PROMPT = """
You are a classifier. Given a user message, output exactly one of:
  billing | technical | refund | general

User message: {message}
Output:
"""

def route(message: str) -> str:
    response = fast_llm_call(ROUTER_PROMPT.format(message=message))
    return response.strip().lower()
```

The key design choices:
- **Use a small, fast, cheap model** for the router. You don't need GPT-4 to classify intent. A small model (Haiku, GPT-4o-mini) handles routing at 1/10th the cost and 2× the speed.
- **Force structured output.** Ask for a single word or a JSON object with a fixed schema. Don't ask for prose.
- **Include edge cases.** What should the router do when the message is ambiguous? Provide a "default" or "clarify" category.

LLM-based routing handles paraphrases, code-switching, typos, and novel phrasing far better than keyword rules.

---

## 7. Embedding-Based Routing

Represent each route as a set of example queries. At routing time, embed the incoming message and find the nearest route by cosine similarity:

```python
# Offline: build route embeddings
route_embeddings = {
    "billing":   embed(["How do I update my credit card?", "I was charged twice", ...]),
    "technical": embed(["The app crashes on login", "I can't export my data", ...]),
    "refund":    embed(["I want my money back", "Can I cancel my subscription?", ...]),
}

# Online: route by similarity
def route(message: str) -> str:
    query_embedding = embed(message)
    return argmax_cosine_similarity(query_embedding, route_embeddings)
```

**Pros:** Fast (embedding lookup is cheap), handles paraphrases, no LLM call at routing time.

**Cons:** Requires example queries per route (curation work), less accurate than LLM routing for edge cases, needs an embedding model.

Embedding routing is the standard approach when:
- You have many routes (10+) and keyword rules are unmanageable
- Latency is critical (embedding lookup << LLM call)
- You can invest in curating quality example sets per route

---

## 8. Designing Subagents

Each subagent should be designed for focus, not flexibility:

**Narrow system prompt** — The subagent should only know what it needs. A billing agent doesn't need to know about technical troubleshooting. Narrow prompts reduce confusion and improve accuracy.

**Specialized tools** — Give each subagent only the tools it needs. The billing agent gets `lookup_invoice` and `apply_credit`. The technical agent gets `check_service_status` and `search_knowledge_base`. Don't share a bloated tool set.

**Clear handoff protocol** — Define how subagents signal completion, failure, or the need to escalate. A common pattern: subagents return a structured response with a `status` field (`completed | failed | escalate`) and a `result`.

**Stateless where possible** — Subagents that don't maintain their own conversation history are easier to reason about, test, and scale. The orchestrator manages state; the subagent just processes one request.

---

## 9. Failure Modes

**Misrouting** — The most common failure. A billing question gets sent to the technical agent. Mitigation: add a "confidence" score and route to a general agent for low-confidence cases.

**Route proliferation** — Teams keep adding new specialized agents until the routing problem becomes as hard as the original task. Periodically audit whether route consolidation would be better.

**Silent failures** — A subagent fails but the orchestrator doesn't detect it and presents a partial answer as complete. Always check for error states in subagent responses.

**State management complexity** — When subagents need to share state (e.g., "the user mentioned their account number in a previous turn handled by a different agent"), managing that context becomes hard. Design your state passing protocol carefully.

**Routing loops** — An escalation from subagent A triggers routing back to subagent A. Add cycle detection.

---

## 10. Key Principles

> **Principle 1 — Specialization beats generalization for depth.**
> A focused agent with a narrow prompt and targeted tools outperforms a general agent on its specific domain. Use routing to get the benefits of specialization without sacrificing breadth.

> **Principle 2 — The router is on the critical path.**
> Every user request passes through the router. A slow or inaccurate router degrades the entire system. Make it fast, accurate, and fallback-safe.

> **Principle 3 — Match routing complexity to the problem.**
> Don't use an LLM to route if five keywords cover 95% of cases. Don't use keywords if your inputs are diverse and paraphrased. Start simple and upgrade when evidence shows it's needed.

> **Principle 4 — Multi-agent systems amplify both capability and failure modes.**
> When things work, you get specialized excellence. When things fail, you get complex cascades that are hard to debug. Invest in observability from the start.

---

## 11. In the Real World

**Intercom / Zendesk AI**
Customer support platforms use intent classification (a routing layer) to direct incoming tickets to the right automated workflow or human queue. This is the most commercially deployed form of agent routing — millions of tickets per day routed by ML classifiers.

**OpenAI Swarm (experimental)**
OpenAI's Swarm framework is built entirely around the agent handoff pattern. Agents call `transfer_to_X()` functions that route control to specialized agents. The routing is LLM-driven: the current agent decides whether to handle the request or hand off based on instructions in its system prompt.

**Anthropic — multi-agent guidance**
Anthropic's documentation explicitly recommends the orchestrator/subagent pattern for complex tasks, with Claude as both orchestrator and as individual subagents. They note that specialized subagents with focused prompts outperform single-agent approaches for multi-domain tasks.

**LangGraph**
LangGraph's "conditional edges" in the graph abstraction are a formalization of routing. Each node (agent) in the graph can return a routing decision that determines which node handles the next step. Complex multi-agent workflows are expressed as graphs with conditional routing.

**LlamaIndex — Router Query Engine**
LlamaIndex provides `RouterQueryEngine` that routes queries to different index types (vector, keyword, summary) based on query classification. The routing prompt asks the model which index is best suited for the query — LLM-based routing applied to retrieval.

**AWS Bedrock — Agents with Action Groups**
AWS Bedrock's agent framework routes to "action groups" (function groups) based on the user's request. The routing is handled by the model choosing which action group's tools to use — a form of implicit routing via tool selection.

**Sierra (customer service AI)**
Sierra's AI agents use a routing architecture where a triage agent classifies customer issues and routes to specialized agents (billing, technical, returns). The specialized agents have deep integrations with back-end systems and narrow, task-specific prompts.

---

## 12. Running the Experiment

```bash
# From the project root (experiment coming soon)

uv run python multi-agent/01-router/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python multi-agent/01-router/demo.py --real
```

**Planned exercises:**
1. Implement all three routing strategies (rule, LLM, embedding) for the same set of routes and compare accuracy on a test set.
2. Add a "confidence" threshold below which the router sends the request to a general fallback agent.
3. Instrument the router to log misrouting — build intuition for where each strategy fails.
4. Design a simple 3-agent system (router + 2 specialists) and trace a full request end-to-end.

---

*Next experiment: Parallel Agent Execution (coming soon)*
