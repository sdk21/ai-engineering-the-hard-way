# Lesson: Tool Error Handling

**Vertical:** Tools | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [Why Error Handling Is Different for Tool Use](#1-why-error-handling-is-different-for-tool-use)
2. [The Four Error Types](#2-the-four-error-types)
3. [Actionable vs. Silent Errors](#3-actionable-vs-silent-errors)
4. [The is_error Flag](#4-the-is_error-flag)
5. [Error Recovery Patterns](#5-error-recovery-patterns)
6. [Validation: Catching Bad Arguments](#6-validation-catching-bad-arguments)
7. [Rate Limits and Retries](#7-rate-limits-and-retries)
8. [Graceful Degradation](#8-graceful-degradation)
9. [Key Principles](#9-key-principles)
10. [In the Real World](#10-in-the-real-world)
11. [Running the Experiment](#11-running-the-experiment)

---

## 1. Why Error Handling Is Different for Tool Use

In normal programming, errors propagate to the caller via exceptions or return codes. With tool use, the model is the caller — and the model cannot catch exceptions. It only sees what you put in the `tool_result` message.

This changes the design goal: **errors must be informative enough that the model can decide what to do next.**

```
Without actionable errors:
  User: "What's the weather in Tokio?"
  Tool: Error: city not found
  Model: I'm sorry, I couldn't find weather for Tokio.  ← dead end

With actionable errors:
  User: "What's the weather in Tokio?"
  Tool: Error: city 'Tokio' not found. Did you mean: tokyo?
  Model: [retries with city="Tokyo"]
  Model: The weather in Tokyo is 22°C and sunny.  ← recovered
```

The error message is the model's only recovery signal. Invest in it.

---

## 2. The Four Error Types

**Validation errors** — The model passed arguments that don't meet the tool's constraints. The model can fix these by re-reading the description and trying again.

```python
# Bad unit
{"error": "Invalid unit 'kelvin'. Must be 'celsius' or 'fahrenheit'.", "error_type": "validation"}

# Negative amount
{"error": "Amount must be positive, got -50.", "error_type": "validation"}
```

**Resource errors** — The requested resource doesn't exist or is temporarily unavailable. Often recoverable with a different input.

```python
# Unknown city
{"error": "City 'Tokio' not found. Did you mean: tokyo?", "error_type": "not_found"}

# Unknown currency pair
{"error": "Unsupported pair USD→CNY. Supported: USD→EUR, USD→GBP, ...", "error_type": "unsupported"}
```

**Permission errors** — The model requested something it's not authorized to access. Usually not recoverable — the model should explain this to the user.

```python
{"error": "Access denied: user U0042 is in the restricted admin range.", "error_type": "permission"}
```

**Execution errors** — Unexpected exceptions during the tool's logic. These indicate bugs or edge cases.

```python
{"error": "Cannot divide by zero. Please provide a non-zero divisor.", "error_type": "execution"}
```

---

## 3. Actionable vs. Silent Errors

The quality of an error message is measured by whether the model can act on it:

| Error | Actionable? | Why |
|-------|------------|-----|
| `"Error: not found"` | No | The model doesn't know what's valid |
| `"City 'Tokio' not found. Did you mean: tokyo?"` | Yes | The model can retry with the suggestion |
| `"Error: rate limit"` | Partially | The model knows to stop but not when to retry |
| `"Rate limit. Retry after 60 seconds."` | Yes | The model can inform the user accurately |
| `"Unsupported pair. Supported: USD→EUR, USD→GBP"` | Yes | The model can suggest alternatives |
| `"Permission denied"` | No | The model doesn't know why or what to do |
| `"Access denied for admin IDs (U0001–U0999). Use U1000+"` | Yes | The model knows what range works |

**Rule of thumb:** If a human reading the error message would know what to try next, it's actionable. If not, improve it.

---

## 4. The is_error Flag

The Anthropic API accepts an optional `is_error` boolean on tool results:

```python
# Normal result
{
    "type": "tool_result",
    "tool_use_id": "tu_abc",
    "content": "temperature: 22\ncondition: sunny",
}

# Error result
{
    "type": "tool_result",
    "tool_use_id": "tu_abc",
    "content": "City 'Tokio' not found. Did you mean: tokyo?",
    "is_error": True,
}
```

**With `is_error: True`:** The model is told this is a failure. It's more likely to:
- Acknowledge the failure to the user
- Attempt recovery if the error message is actionable
- Not treat the error message as factual data

**Without `is_error`:** The model treats the content as a normal result. A string like "Error: not found" will still be read as an error (the model understands English), but it's less explicit.

**Recommendation:** Use `is_error: True` for genuine tool failures. Omit it for partial or degraded results where you still have useful data to return.

---

## 5. Error Recovery Patterns

**Pattern 1 — Suggest-and-retry**
Include the closest valid alternative in the error. The model will usually retry automatically.

```python
# Good error for a typo
return {
    "error": f"City '{city}' not found.",
    "hint": f"Did you mean: {closest_match}?",
}
```

**Pattern 2 — Enumerate valid options**
For constrained inputs, list what's valid.

```python
return {
    "error": f"Unsupported currency pair: {from_c} to {to_c}.",
    "supported_pairs": ["USD→EUR", "USD→GBP", "USD→JPY"],
}
```

**Pattern 3 — Retry-after for rate limits**
Include the wait time so the model can inform the user accurately.

```python
return {
    "error": "Rate limit exceeded.",
    "retry_after_seconds": 60,
}
```

**Pattern 4 — Partial success**
Return what you have and describe what's missing.

```python
return {
    "data": partial_results,
    "warning": "Data for MSFT is unavailable. Showing AAPL only.",
}
```

---

## 6. Validation: Catching Bad Arguments

The model constructs tool arguments from its understanding of your description. It can make mistakes:
- Typos in string values ("kelvin" instead of "celsius")
- Wrong numeric ranges (negative amounts, out-of-bound indices)
- Wrong ID formats (passing "Alice" where "U1042" is expected)
- Wrong types if your parser is lenient

Your tool should validate all inputs before execution:

```python
def get_weather(city: str, unit: str = "celsius") -> dict:
    # Validate at the boundary
    if unit not in ("celsius", "fahrenheit"):
        return {"error": f"Invalid unit '{unit}'. Use 'celsius' or 'fahrenheit'."}

    # Now safe to use
    data = fetch_weather(city, unit)
    ...
```

**Never trust model-provided arguments.** Treat them like user input from an untrusted source.

---

## 7. Rate Limits and Retries

If your tool calls external APIs, you will hit rate limits. How you handle them determines whether the model gives up or waits:

```python
import time

def call_with_retry(fn, *args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fn(*args)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                return {"error": f"Rate limit exceeded after {max_retries} retries. Try again later."}
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            time.sleep(wait)
```

**Key decision:** Should retries happen inside the tool (invisible to the model) or should the tool return an error that the model surfaces to the user?

- **Inside the tool** — for transient errors (network blips, brief rate limits). The model gets a clean result.
- **Return an error** — for persistent limits or when the user should know. The model can tell the user to try later.

---

## 8. Graceful Degradation

When a tool partially fails, returning partial data is usually better than returning an error:

```python
def get_multi_city_weather(cities: list[str]) -> dict:
    results = {}
    errors = []

    for city in cities:
        try:
            results[city] = fetch_weather(city)
        except Exception as e:
            errors.append(f"{city}: {e}")

    response = {"results": results}
    if errors:
        response["warnings"] = errors  # report failures without hiding successes
    return response
```

The model can incorporate both the results and the warnings into its response, giving the user maximum information even when some calls fail.

---

## 9. Key Principles

> **Principle 1 — The error message is the model's recovery signal.**
> The model can only act on what you put in the tool_result. An error that doesn't suggest a path forward is a dead end. Every error should answer: "what should I try instead?"

> **Principle 2 — Validate all tool arguments.**
> Model-provided arguments are untrusted input. Validate types, ranges, and formats before executing. Return validation errors with the corrected format as a hint.

> **Principle 3 — Use is_error for real failures, not degraded results.**
> Reserve `is_error: True` for genuine tool failures. If you have partial data, return it as a normal result with a warning field.

> **Principle 4 — Let rate limit errors surface to the user.**
> Don't silently sleep inside your tool for long periods. Return a rate limit error with a retry time — the model will tell the user, which is better than silent hangs.

> **Principle 5 — Execution errors indicate bugs in your tool.**
> If your tool throws unexpected exceptions, you have a bug. Catch them, log them, and return a clean error to the model. Don't let raw stack traces reach the model.

---

## 10. In the Real World

**OpenAI Function Calling**
OpenAI's documentation explicitly recommends returning errors as tool results rather than raising exceptions: "if the function encounters an error, return a description of the error as the result." This is universal advice, not OpenAI-specific.

**LangChain Tool Error Handling**
LangChain's `@tool` decorator catches exceptions and returns them as error strings by default. Production applications override this to add validation and actionable error formatting.

**Stripe API Design**
Stripe's error objects always include a `code` (machine-readable), `message` (human-readable), and `param` (which parameter was wrong). This is the gold standard for actionable errors — the same principles apply to tool results.

**Zapier AI Error Recovery**
When an automated workflow fails a step, the Zapier agent generates a "here's what went wrong and what to try" message using the error content. Agents that process errors as text produce better recovery suggestions than agents that only receive codes.

**GitHub Copilot Workspace**
When a shell command fails in Copilot Workspace, the agent receives the full stderr output, which typically includes enough context to fix the issue. Returning raw error output — rather than a generic "command failed" — is what enables autonomous error correction.

---

## 11. Running the Experiment

```bash
# From the project root

# Mock mode — runs scripted scenarios showing each error type
uv run python tools/04-tool-error-handling/demo.py --mock

# Real mode — interactive, observe Claude recover from errors
ANTHROPIC_API_KEY=sk-... uv run python tools/04-tool-error-handling/demo.py --real
```

**Suggested queries for real mode:**
- `"What's the weather in Tokio?"` — typo causes not_found error; model retries with "Tokyo"
- `"Convert -50 USD to EUR."` — validation error; model explains the constraint
- `"What's 100 divided by 0?"` — execution error; model explains why this fails
- `"Look up user U0042."` — permission error; model tells user what range is allowed
- `"Look up user ABC then get their weather."` — format error; model corrects and chains

**Suggested exercises:**
1. Change the `get_weather` error to just `"Error: not found"` (remove the hint). Observe whether the model can still recover.
2. Make `convert_currency` always return a rate limit error and observe how the model explains this to the user.
3. Add a `fuzzy_match` function that finds the closest city name using edit distance, and incorporate it into the error message.
4. Add a `max_retries=2` guard so the model doesn't loop endlessly on persistent errors.

---

*Previous: [Parallel Tool Calls](../03-parallel-tool-calls/) · Next: [Tool Result Context](../05-tool-result-context/)*
