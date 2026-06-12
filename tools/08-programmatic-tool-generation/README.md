# Lesson: Programmatic Tool Generation

**Vertical:** Tools | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [The Problem with Hand-Written Schemas](#1-the-problem-with-hand-written-schemas)
2. [Generating Schemas from Type Annotations](#2-generating-schemas-from-type-annotations)
3. [The @tool Decorator](#3-the-tool-decorator)
4. [Parsing Docstrings for Descriptions](#4-parsing-docstrings-for-descriptions)
5. [Generating Tools from a Class](#5-generating-tools-from-a-class)
6. [Generating Tools from an OpenAPI Spec](#6-generating-tools-from-an-openapi-spec)
7. [Keeping Schemas in Sync](#7-keeping-schemas-in-sync)
8. [Limitations](#8-limitations)
9. [Key Principles](#9-key-principles)
10. [In the Real World](#10-in-the-real-world)
11. [Running the Experiment](#11-running-the-experiment)

---

## 1. The Problem with Hand-Written Schemas

Every tool in the basic function calling experiment requires a hand-written JSON Schema:

```python
{
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}
```

This works for 2–3 tools. It doesn't scale:
- **Drift:** When you rename a parameter in Python, the schema silently goes stale
- **Duplication:** The parameter list exists in both the function signature and the schema
- **Friction:** Adding a new tool requires writing both implementation and schema
- **Maintenance:** A 50-tool suite means 50 schemas to keep synchronized manually

The solution: generate schemas from source — the function signature is already the ground truth.

---

## 2. Generating Schemas from Type Annotations

Python type hints map directly to JSON Schema types:

| Python Type | JSON Schema |
|-------------|------------|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `Literal["a", "b"]` | `{"type": "string", "enum": ["a", "b"]}` |
| `Optional[str]` | `{"type": "string"}` (required=False) |

The `inspect` and `typing` modules expose everything needed:

```python
import inspect
from typing import get_type_hints, get_origin, get_args, Literal

def signature_to_schema(fn):
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "return"):
            continue
        tp = hints.get(name, str)
        properties[name] = type_to_json_schema(tp)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}
```

---

## 3. The @tool Decorator

The `@tool` decorator wraps a Python function and generates its schema automatically:

```python
@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """
    Get current weather for a city.

    Args:
        city: The city name to get weather for.
        unit: Temperature unit, either celsius or fahrenheit.
    """
    ...

# Access the generated schema
get_weather.api_schema
# {
#   "name": "get_weather",
#   "description": "Get current weather for a city.",
#   "input_schema": {
#     "type": "object",
#     "properties": {
#       "city": {"type": "string", "description": "The city name to get weather for."},
#       "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit..."}
#     },
#     "required": ["city"]
#   }
# }

# Call it normally
result = get_weather(city="Tokyo", unit="celsius")
```

The schema is a derived artifact — it's always consistent with the implementation.

---

## 4. Parsing Docstrings for Descriptions

The tool description and parameter descriptions come from the docstring. Google-style docstrings are the most common format:

```python
@tool
def send_email(to: str, subject: str, body: str, cc: str = None) -> str:
    """
    Send an email to one or more recipients.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
        cc: Optional CC recipient.
    """
```

The parser extracts:
- Function description from the first paragraph
- Parameter descriptions from the Args section
- These are injected into the JSON Schema `description` fields

Without docstrings, the generated schema will have empty descriptions — which degrades model performance. Write good docstrings; they serve double duty as both human documentation and model guidance.

---

## 5. Generating Tools from a Class

`tools_from_class` wraps every public method of a class:

```python
class WeatherService:
    def current(self, city: str) -> dict:
        """Get current weather. Args: city: City name."""
        ...

    def forecast(self, city: str, days: int = 7) -> dict:
        """Get forecast. Args: city: City name. days: Number of days (1-10)."""
        ...

    def historical(self, city: str, date: str) -> dict:
        """Get historical weather. Args: city: City name. date: ISO date string."""
        ...

service = WeatherService()
tools = tools_from_class(service)
# → [ToolDefinition(current), ToolDefinition(forecast), ToolDefinition(historical)]
```

This is powerful for wrapping SDK clients. Instead of manually defining tools for every API endpoint, you generate them from the client class:

```python
import anthropic
# Hypothetical: generate tools from Anthropic client methods
tools = tools_from_class(anthropic.Anthropic())
```

---

## 6. Generating Tools from an OpenAPI Spec

`tools_from_openapi` converts API operations to tool schemas:

```python
import json

spec = json.load(open("api_spec.json"))
tools = tools_from_openapi(spec)
# One tool per API operation, parameters extracted from path params + request body
```

This is especially useful for:
- Wrapping third-party REST APIs (no Python SDK available)
- Building agents that can call your own API
- Generating tools for microservices in a service mesh

The limitation: you get schemas but no Python implementations. You need a generic HTTP dispatch function that calls the API endpoint based on the operation ID and arguments.

---

## 7. Keeping Schemas in Sync

With programmatic generation, the schema is always in sync because it IS the function signature. The only way to break it is to:

1. Remove a type annotation (schema falls back to `{"type": "string"}`)
2. Change a parameter name (schema updates, model may break if prompt was hardcoded)
3. Add required parameters without updating callers

**Best practice:** add schema generation to your test suite:

```python
def test_tool_schemas_valid():
    for td in ALL_TOOLS:
        schema = td.api_schema
        assert "name" in schema
        assert "description" in schema
        assert schema["description"]  # non-empty description
        for param in schema["input_schema"]["properties"].values():
            assert "description" in param  # all params have descriptions
```

---

## 8. Limitations

**Complex types:** Nested dataclasses, Union types with 3+ options, TypedDict — these require more sophisticated schema generation than this experiment implements. In production, use Pydantic's `model_json_schema()` for complex types.

**Return type:** The return type annotation (`-> str`) is not part of the tool schema (the API doesn't use it). But it's still useful for static type checking in your own code.

**Dynamic descriptions:** Sometimes a tool's description should include runtime context (e.g., the current user's name). The `@tool` decorator produces static descriptions. For dynamic descriptions, generate the schema dict directly at call time.

---

## 9. Key Principles

> **Principle 1 — The function signature is the ground truth.**
> Don't maintain schemas separately from implementations. Generate them from the source. Divergence between schema and implementation is a class of bug that programmatic generation eliminates.

> **Principle 2 — Docstrings are model documentation.**
> The `description` field on a tool and its parameters is what the model reads to decide how to use the tool. Treat docstrings as first-class API documentation — not optional comments.

> **Principle 3 — Type annotations should be specific.**
> `Literal["celsius", "fahrenheit"]` generates a better schema than `str`. Use the narrowest type that correctly describes the parameter. This helps the model call your tools correctly.

> **Principle 4 — Test schema validity.**
> Add assertions that all tools have non-empty descriptions and that all required parameters are annotated. Catch schema regressions the same way you catch other bugs.

---

## 10. In the Real World

**LangChain @tool Decorator**
LangChain's `@tool` decorator does exactly what this experiment implements — it reads the function signature and docstring and generates the tool schema. This is the production pattern for building tool suites in Python.

**Pydantic Function Tools**
Pydantic's `model_json_schema()` can generate complex JSON Schema from dataclass-like models. LangChain and other frameworks use Pydantic models as tool argument validators.

**FastAPI → Agent Tools**
FastAPI automatically generates OpenAPI specs from route function signatures. You can run `tools_from_openapi` on a FastAPI app's spec to turn any REST API into a tool suite, zero hand-writing required.

**Cursor / Windsurf IDE Tools**
IDE AI agents define their tools (read_file, write_file, run_terminal) programmatically from internal system schemas rather than hand-written JSON. The schemas stay in sync with implementation as the tools evolve.

---

## 11. Running the Experiment

```bash
# From the project root

# Mock mode — see generated schemas
uv run python tools/08-programmatic-tool-generation/demo.py --mock

# Real mode — use generated tools in a real API call
ANTHROPIC_API_KEY=sk-... uv run python tools/08-programmatic-tool-generation/demo.py --real
```

**Suggested exercises:**
1. Add a `list[str]` parameter to one of the example tools and verify the schema updates correctly.
2. Define a `Pydantic` model as a tool argument type and implement schema generation for it.
3. Run `tools_from_openapi` on a real public OpenAPI spec (e.g., Petstore) and inspect the generated schemas.
4. Add a test that asserts all `@tool`-decorated functions have non-empty docstrings and parameter descriptions.

---

*Previous: [Dynamic Tool Selection](../07-dynamic-tool-selection/) · Next: [Streaming with Tools](../09-streaming-tools/)*
