# Lesson: Basic Function Calling

**Vertical:** Tools | **Difficulty:** Beginner | **Status:** ✅ Ready

---

## Table of Contents

1. [The Problem: LLMs Can't Act](#1-the-problem-llms-cant-act)
2. [What Is Function Calling?](#2-what-is-function-calling)
3. [The Tool-Call Loop — Step by Step](#3-the-tool-call-loop--step-by-step)
4. [Defining Tools as JSON Schemas](#4-defining-tools-as-json-schemas)
5. [The Agentic Loop](#5-the-agentic-loop)
6. [How the Model Decides When to Call a Tool](#6-how-the-model-decides-when-to-call-a-tool)
7. [Tool Results and Multi-Step Calls](#7-tool-results-and-multi-step-calls)
8. [Safety and Sandboxing](#8-safety-and-sandboxing)
9. [Failure Modes](#9-failure-modes)
10. [Key Principles](#10-key-principles)
11. [In the Real World](#11-in-the-real-world)
12. [Running the Experiment](#12-running-the-experiment)

---

## 1. The Problem: LLMs Can't Act

LLMs are text in, text out. They can reason, write, summarize, and plan — but they cannot:

- Look up live data (weather, stock prices, search results)
- Execute code
- Read or write files
- Call APIs
- Control a computer

This limits them to tasks where knowledge baked in at training time is sufficient. For anything requiring current information or side effects in the real world, a pure LLM is useless.

Function calling (also called "tool use") is the mechanism that closes this gap.

---

## 2. What Is Function Calling?

Function calling gives the model a way to request that your code runs a function and returns the result. The model doesn't run code itself — it emits a structured request, and you (the developer) execute the actual function.

The flow:

```
1. You define tools as JSON schemas (name, description, parameters)
2. You send those tool definitions alongside the user's message
3. The model decides whether to answer directly OR call a tool
4. If the model calls a tool, it returns a structured tool_use block
5. You execute the function with the provided arguments
6. You send the result back to the model as a tool_result
7. The model incorporates the result and produces a final response
```

This loop can repeat multiple times if the model needs to call several tools to answer a single question.

---

## 3. The Tool-Call Loop — Step by Step

```
User: "What's the weather in Tokyo and how many days until New Year?"

──────────────────────────────────────────────────────
API Call 1
  input:  [user message] + [tool definitions]
  output: tool_use → get_weather(city="Tokyo")
──────────────────────────────────────────────────────
  Your code executes: get_weather("Tokyo") → "22°C, sunny"
──────────────────────────────────────────────────────
API Call 2
  input:  [user message, assistant tool_use, tool_result: "22°C, sunny"] + [tool definitions]
  output: tool_use → days_until(date="January 1")
──────────────────────────────────────────────────────
  Your code executes: days_until("January 1") → "183 days"
──────────────────────────────────────────────────────
API Call 3
  input:  [full history including both tool results] + [tool definitions]
  output: text → "It's 22°C and sunny in Tokyo. New Year is 183 days away."
──────────────────────────────────────────────────────
```

The model orchestrates the calls. You execute them. Together you get a useful answer.

---

## 4. Defining Tools as JSON Schemas

Tools are described to the model using JSON Schema. Each tool definition has three parts:

```python
{
    "name": "get_weather",
    "description": "Get the current weather conditions for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g. 'Tokyo' or 'New York'."
            }
        },
        "required": ["city"]
    }
}
```

**`name`** — The identifier your code uses to dispatch to the right function. Keep it lowercase with underscores.

**`description`** — The most important field. The model reads this to decide whether to use the tool. A vague description leads to wrong or missed calls. Be specific about what the tool does, what it returns, and when it should be used.

**`input_schema`** — JSON Schema describing the parameters. Include descriptions on each property — the model uses them to construct the correct arguments.

The quality of your tool descriptions is a primary driver of function calling accuracy.

---

## 5. The Agentic Loop

The **agentic loop** is the core control structure of any tool-using agent:

```python
messages = [{"role": "user", "content": user_input}]

while True:
    response = api_call(messages=messages, tools=tool_definitions)
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        # Model is done — return the text response
        return extract_text(response)

    if response.stop_reason == "tool_use":
        # Model wants to call one or more tools
        tool_results = []
        for tool_call in response.tool_use_blocks:
            result = dispatch(tool_call.name, tool_call.input)
            tool_results.append(make_tool_result(tool_call.id, result))
        messages.append({"role": "user", "content": tool_results})
        # Loop — give the model the results and let it continue
```

The loop terminates when the model produces a final text response (`end_turn`) rather than another tool call. Without this loop, the model can only make one tool call per user message. With it, the model can chain as many calls as needed to answer the question.

---

## 6. How the Model Decides When to Call a Tool

The model uses the tool descriptions to make a judgment call at each step:

- If it can answer from its training knowledge alone → direct text response
- If it needs current data or capabilities it lacks → tool call
- If the task requires multiple pieces of information → sequential or parallel tool calls

This decision is not deterministic and can be influenced by:

- **Description clarity**: vague descriptions cause missed or incorrect tool use
- **Examples in the description**: showing sample inputs/outputs helps calibrate the model
- **Tool count**: too many tools can overwhelm the model's selection ability (>20 tools starts to degrade)
- **System prompt instructions**: you can instruct the model to always use a tool, or never use it for certain inputs

A useful mental model: the model treats tool descriptions the way a programmer treats documentation. Clear, accurate docs lead to correct usage; poor docs lead to misuse.

---

## 7. Tool Results and Multi-Step Calls

Tool results are returned as a special message type, not a regular user message:

```python
# Anthropic format
{
    "type": "tool_result",
    "tool_use_id": "toolu_abc123",   # must match the tool_use id
    "content": "22°C, sunny"
}
```

The `tool_use_id` ties the result back to the specific tool call. This matters when the model calls multiple tools in parallel — each result is matched to its corresponding request by ID.

**Parallel tool calls** happen when the model determines that multiple tools can be called independently:

```
User: "What's the weather in Tokyo and Paris?"
  → Model emits two tool_use blocks in one response: get_weather("Tokyo") AND get_weather("Paris")
  → You execute both, return both results
  → Model combines them in a single final response
```

This reduces round-trips and latency when tasks are independent.

---

## 8. Safety and Sandboxing

Tool use is a code execution surface. The model can call any tool you expose, with any arguments it chooses. This has security implications:

**Validate inputs** — The model might pass unexpected argument values. Don't trust them blindly. Validate types, ranges, and formats before executing.

**Limit scope** — Only expose tools the model actually needs. A model with access to `delete_file` will eventually call it when you didn't intend it to.

**Sandbox side effects** — Tools that write to databases, send emails, or call external APIs should have rate limits, confirmation steps, or dry-run modes.

**Prompt injection** — If tool results contain untrusted content (e.g., web search results), a malicious page could embed instructions telling the model to call a dangerous tool. Treat tool results from external sources the same as user input.

This experiment's `calculate` tool uses `ast.parse` + a safe operator whitelist rather than `eval()` for exactly this reason — arbitrary code execution via tool arguments is a real attack surface.

---

## 9. Failure Modes

**Hallucinated tool calls**
The model calls a tool with syntactically correct but semantically wrong arguments (e.g., `get_weather(city="a city that doesn't exist")`). Your implementation must handle this gracefully.

**Infinite tool loops**
If a tool always returns an error or an unsatisfying result, a model may keep calling it in a loop. Add a max-iterations limit to your agentic loop.

**Wrong tool selection**
With many tools, the model may choose the wrong one. Improve descriptions, add examples, or reduce the tool count.

**Partial tool result integration**
The model may ignore a tool result and answer from prior knowledge anyway, especially if the tool result contradicts its training data. This is particularly common with time-sensitive information.

**Latency multiplication**
Each tool call adds at least one round-trip to the API. A 3-step tool chain has 3x the latency of a direct response. Design tool granularity to minimize unnecessary hops.

---

## 10. Key Principles

> **Principle 1 — The model orchestrates; you execute.**
> The model is the brain deciding what to do. Your code is the hands that do it. This separation is fundamental to the tool use architecture.

> **Principle 2 — Tool descriptions are the API contract.**
> The model reads descriptions to decide how and when to use tools. Investing in clear, accurate descriptions pays off directly in reliability.

> **Principle 3 — Tool use transforms LLMs from knowledge bases into agents.**
> A model without tools can only retrieve baked-in knowledge. Tools give it the ability to act in and on the world.

> **Principle 4 — Validate everything that comes back from the model.**
> Tool arguments are model outputs. They can be malformed, out of range, or adversarially crafted if prompt injection is possible. Treat them like untrusted input.

> **Principle 5 — The agentic loop is the unit of work.**
> A single user message may require many tool calls to answer. The loop is what makes multi-step reasoning possible.

---

## 11. In the Real World

**OpenAI — Function Calling / Tools API**
OpenAI introduced function calling in June 2023. It was one of the most impactful API changes in LLM history, enabling the first generation of production AI agents. The tool definition format (name, description, JSON Schema parameters) they introduced has become an informal standard, adopted by Anthropic, Google, and others.

**Anthropic — Tool Use**
Anthropic's tool use API (used in this experiment) follows the same conceptual model. Claude is trained to be particularly careful about tool use — it will ask for clarification rather than guess at ambiguous arguments, and it tends not to call tools when it can answer from knowledge. This is a deliberate safety behavior.

**LangChain — Tools and Agents**
LangChain's entire agent abstraction is built on function calling. Tools are first-class objects with names, descriptions, and `run()` methods. Their `AgentExecutor` implements the agentic loop described above. The majority of LangChain tutorials are essentially teaching function calling with wrapper abstractions.

**LlamaIndex — Query Tools**
LlamaIndex wraps RAG pipelines as tools that agents can call. A vector index becomes a tool: `search_knowledge_base(query: str) → str`. This is the standard pattern for giving agents access to large document collections.

**OpenAI Assistants API**
The Assistants API is function calling with persistence: thread history is managed server-side, and built-in tools (code interpreter, file search) are implemented as function calls under the hood.

**Cursor / GitHub Copilot**
IDE AI assistants use tool calls to read files, search code, run terminals, and apply edits. When Cursor "reads your codebase," it is running tool calls like `read_file(path)` and `search_code(query)` in a loop.

**Zapier AI Actions / Make.com**
Workflow automation platforms expose their 5000+ integrations as tool definitions that an LLM can call. This is function calling scaled to the entire SaaS ecosystem — the model picks which integration to invoke based on natural language intent.

**Replit Agent / Devin / SWE-agent**
Autonomous software engineering agents are entirely built on function calling. Their tool sets include: `read_file`, `write_file`, `run_command`, `search_web`, `create_pr`. The agent loops through tool calls until the task is done. The quality of the tool set is the primary differentiator between these products.

---

## 12. Running the Experiment

```bash
# From the project root

# Mock mode — see the tool call loop without an API key
uv run python tools/01-basic-function-calling/demo.py --mock

# Real mode — watch Claude decide which tool to call
ANTHROPIC_API_KEY=sk-... uv run python tools/01-basic-function-calling/demo.py --real
```

**Suggested queries to try:**
- `"What's the weather in London?"` — single tool call
- `"What's 2 to the power of 16?"` — calculator tool
- `"Is it warmer in Sydney or Paris?"` — two tool calls, one question
- `"Hello, how are you?"` — observe the model answering without a tool call

**Suggested exercises:**
1. Add a third tool: `get_time(city: str) → str`. Give it a description and a fake implementation.
2. Deliberately write a bad tool description and observe how it affects the model's selection.
3. Read `experiment.py` and trace the `dispatch_tool` function — this is where your code takes control back from the model.
4. Add a `max_iterations` guard to the agentic loop in `demo.py` to prevent infinite loops.

---

*Next experiment: Tool Chaining (coming soon)*
