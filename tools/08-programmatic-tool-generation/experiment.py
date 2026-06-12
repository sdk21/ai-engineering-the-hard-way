"""
Programmatic Tool Generation
-----------------------------
Instead of hand-writing tool schemas, you generate them automatically
from Python function signatures, type annotations, and docstrings.

Key concepts:
- Extracting JSON Schema from type annotations (str, int, float, bool, list, Literal)
- Parsing docstrings for parameter descriptions
- Decorators that turn any Python function into a tool
- Generating entire tool suites from a data model or API spec

This removes the burden of keeping schemas in sync with implementations.
When you change a function signature, the schema updates automatically.

Three generation approaches:
  1. @tool decorator — annotate any function to make it a tool
  2. from_class — generate tools from all public methods of a class
  3. from_openapi — generate tools from an OpenAPI spec (simulated)
"""

import inspect
import json
import re
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints


# ---------------------------------------------------------------------------
# Type annotation → JSON Schema conversion
# ---------------------------------------------------------------------------

def _type_to_schema(tp) -> dict:
    """Convert a Python type annotation to a JSON Schema fragment."""
    origin = get_origin(tp)
    args = get_args(tp)

    # Literal["a", "b"] → {"type": "string", "enum": ["a", "b"]}
    if origin is Literal:
        # All values same type
        first = args[0]
        js_type = _py_type_to_json(type(first))
        return {"type": js_type, "enum": list(args)}

    # list[X] → {"type": "array", "items": {...}}
    if origin is list:
        item_schema = _type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    # Optional[X] == Union[X, None] → same as X (required will be False)
    if origin is type(None):
        return {"type": "null"}

    # Union[X, Y] — simplified: use first non-None type
    if hasattr(origin, "__name__") and "Union" in str(origin):
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _type_to_schema(non_none[0])

    # Primitive types
    return {"type": _py_type_to_json(tp)}


def _py_type_to_json(tp) -> str:
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return mapping.get(tp, "string")


# ---------------------------------------------------------------------------
# Docstring parser — extracts parameter descriptions
# ---------------------------------------------------------------------------

def _parse_docstring(fn: Callable) -> tuple[str, dict[str, str]]:
    """
    Returns (function_description, {param_name: param_description}).
    Supports Google-style and NumPy-style docstrings.
    """
    doc = inspect.getdoc(fn) or ""
    if not doc:
        return "", {}

    # Split into function description and args section
    lines = doc.splitlines()
    func_desc_lines = []
    param_descs = {}

    in_args = False
    current_param = None

    for line in lines:
        stripped = line.strip()

        # Google-style: "Args:" or "Parameters:" section header
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_args = True
            continue

        # Google-style: "    param_name (type): description"
        if in_args:
            param_match = re.match(r'\s{2,4}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)', line)
            if param_match:
                current_param = param_match.group(1)
                param_descs[current_param] = param_match.group(2).strip()
                continue
            # Continuation of previous param description
            if current_param and line.startswith("        "):
                param_descs[current_param] += " " + stripped
                continue
            # Empty line or new section
            if not stripped or stripped.endswith(":"):
                current_param = None
            continue

        func_desc_lines.append(stripped)

    func_desc = " ".join(l for l in func_desc_lines if l).strip()
    return func_desc, param_descs


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

class ToolDefinition:
    """A tool generated from a Python function."""

    def __init__(self, fn: Callable, name: str = None, description: str = None):
        self.fn = fn
        self.name = name or fn.__name__
        self._schema = self._generate_schema(description)

    def _generate_schema(self, override_description: str = None) -> dict:
        hints = get_type_hints(self.fn)
        sig = inspect.signature(self.fn)
        func_desc, param_descs = _parse_docstring(self.fn)

        description = override_description or func_desc or f"Call {self.name}"
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls", "return"):
                continue

            tp = hints.get(param_name, str)
            schema_fragment = _type_to_schema(tp)

            # Add description from docstring if available
            if param_name in param_descs:
                schema_fragment["description"] = param_descs[param_name]

            properties[param_name] = schema_fragment

            # Required if no default value
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": self.name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                **({"required": required} if required else {}),
            },
        }

    @property
    def api_schema(self) -> dict:
        return self._schema

    def __call__(self, **kwargs) -> Any:
        return self.fn(**kwargs)


def tool(fn: Callable = None, *, name: str = None, description: str = None):
    """
    Decorator that turns a Python function into a ToolDefinition.

    Usage:
        @tool
        def my_fn(x: int, label: str = "default") -> str:
            ...

        @tool(name="custom_name", description="Custom description")
        def my_fn(...):
            ...
    """
    if fn is not None:
        # Called as @tool (no arguments)
        return ToolDefinition(fn, name=name, description=description)

    # Called as @tool(...) — return decorator
    def decorator(f: Callable) -> ToolDefinition:
        return ToolDefinition(f, name=name, description=description)
    return decorator


# ---------------------------------------------------------------------------
# from_class: generate tools from all public methods of a class
# ---------------------------------------------------------------------------

def tools_from_class(cls) -> list[ToolDefinition]:
    """
    Generate a ToolDefinition for every public method of a class instance.
    Useful for wrapping an API client or service class.
    """
    definitions = []
    for method_name in dir(cls):
        if method_name.startswith("_"):
            continue
        method = getattr(cls, method_name)
        if callable(method):
            # Bind the method (strip self)
            bound = method
            td = ToolDefinition(bound, name=method_name)
            definitions.append(td)
    return definitions


# ---------------------------------------------------------------------------
# from_openapi: generate tools from an OpenAPI spec (simulated)
# ---------------------------------------------------------------------------

def tools_from_openapi(spec: dict) -> list[dict]:
    """
    Generate tool schemas from an OpenAPI 3.x spec.
    Returns raw API schemas (not ToolDefinition — no Python fn to bind).
    """
    tools = []
    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        for method, operation in path_item.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue

            op_id = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
            description = operation.get("summary") or operation.get("description", "")

            properties = {}
            required = []

            # Path parameters
            for param in operation.get("parameters", []):
                pname = param["name"]
                schema = param.get("schema", {"type": "string"})
                schema["description"] = param.get("description", "")
                properties[pname] = schema
                if param.get("required", False):
                    required.append(pname)

            # Request body (POST/PUT/PATCH)
            body = operation.get("requestBody", {})
            content = body.get("content", {}).get("application/json", {})
            body_schema = content.get("schema", {})
            for prop_name, prop_schema in body_schema.get("properties", {}).items():
                properties[prop_name] = prop_schema
                if prop_name in body_schema.get("required", []):
                    required.append(prop_name)

            tools.append({
                "name": op_id,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    **({"required": required} if required else {}),
                },
            })

    return tools


# ---------------------------------------------------------------------------
# Example tools generated via @tool decorator
# ---------------------------------------------------------------------------

@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """
    Get current weather for a city.

    Args:
        city: The city name to get weather for.
        unit: Temperature unit, either celsius or fahrenheit.
    """
    temps = {"tokyo": 22, "london": 15, "paris": 17}
    temp = temps.get(city.lower(), 20)
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)
    return f"{city}: {temp}°{'C' if unit == 'celsius' else 'F'}"


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression: A Python arithmetic expression, e.g. '2 ** 10' or '(3 + 4) * 5'.
    """
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}

    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
        raise ValueError(f"Unsupported: {type(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(eval_node(tree.body))
    except Exception as e:
        return f"Error: {e}"


@tool(name="find_user", description="Look up a user by ID or email address.")
def find_user(user_id: str = None, email: str = None) -> str:
    """Find a user in the system."""
    if user_id:
        return f"User {user_id}: Alice Chen, alice@example.com"
    if email:
        return f"User matching {email}: Bob Patel, U1234"
    return "Error: provide user_id or email"


# Example tools generated from @tool — collect all
EXAMPLE_TOOLS = [get_weather, calculate, find_user]


# ---------------------------------------------------------------------------
# Sample OpenAPI spec for demo
# ---------------------------------------------------------------------------

SAMPLE_OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Pet Store API", "version": "1.0.0"},
    "paths": {
        "/pets": {
            "get": {
                "operationId": "list_pets",
                "summary": "List all pets in the store.",
                "parameters": [
                    {"name": "limit", "in": "query", "required": False,
                     "description": "Maximum number of results to return",
                     "schema": {"type": "integer"}},
                    {"name": "species", "in": "query", "required": False,
                     "description": "Filter by species (cat, dog, bird)",
                     "schema": {"type": "string"}},
                ],
            },
            "post": {
                "operationId": "create_pet",
                "summary": "Add a new pet to the store.",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Pet's name"},
                                    "species": {"type": "string", "description": "Species: cat, dog, bird"},
                                    "age": {"type": "integer", "description": "Age in years"},
                                },
                                "required": ["name", "species"],
                            }
                        }
                    }
                },
            },
        },
        "/pets/{pet_id}": {
            "get": {
                "operationId": "get_pet",
                "summary": "Get a specific pet by ID.",
                "parameters": [
                    {"name": "pet_id", "in": "path", "required": True,
                     "description": "The pet's unique ID",
                     "schema": {"type": "string"}},
                ],
            },
            "delete": {
                "operationId": "delete_pet",
                "summary": "Remove a pet from the store.",
                "parameters": [
                    {"name": "pet_id", "in": "path", "required": True,
                     "description": "The pet's unique ID",
                     "schema": {"type": "string"}},
                ],
            },
        },
    },
}
