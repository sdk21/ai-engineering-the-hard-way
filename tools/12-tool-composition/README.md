# Lesson: Tool Composition

**Vertical:** Tools | **Difficulty:** Advanced | **Status:** ✅ Ready

---

## What This Teaches

Higher-order tools: tools whose implementations call other tools, hiding multi-step logic behind a single interface.

**Three patterns:**

**Pipeline** — chain tools A → B:
```python
def get_weather_in_unit(city, unit):
    raw = get_weather(city)       # tool A
    return convert_units(raw, unit)  # tool B
```

**Aggregate** — fan-out A, B, C then merge:
```python
def compare_weather(cities):
    results = [get_weather(c) for c in cities]  # parallel A calls
    return format_comparison(results)
```

**Conditional** — branch based on intermediate result:
```python
def weather_advisory(city):
    data = get_weather(city)
    if "rain" in data["condition"]:
        advice = "bring umbrella"
    elif data["temp"] > 30:
        advice = "stay hydrated"
    return f"{data} — Advisory: {advice}"
```

**Why compose?** Reduces the model's decision space. Instead of calling `get_weather` then deciding to call `convert_temperature`, the model calls one composite tool. The internal complexity is invisible. The model's job is simpler; errors are less likely.

---

## When to Compose

Compose when:
- A combination of tools is reliably called together
- The intermediate results aren't useful to the model independently
- The composition logic is deterministic (always the same sequence)

Don't compose when:
- The model needs to see intermediate results to decide what to do next
- Different queries require different combinations of the same base tools
- You need to parallelize the underlying calls across different composite tools

---

## Running the Experiment

```bash
uv run python tools/12-tool-composition/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python tools/12-tool-composition/demo.py --real
```

**Suggested exercises:**
1. Build a composite `research_topic(topic)` tool that calls `wikipedia_search`, `get_definition`, and formats the combined result.
2. Implement a tool factory that creates differently-configured versions of the same base tool.
3. Measure whether the model calls fewer total tools when composite tools are available vs. when only atomic tools are provided.

---

*Previous: [Multi-Tool Agent](../11-multi-tool-agent/) · Next: [Capstone: CLI Agent](../capstone/cli-agent/)*
