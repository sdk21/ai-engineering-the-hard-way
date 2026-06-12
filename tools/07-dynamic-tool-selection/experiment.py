"""
Dynamic Tool Selection
----------------------
When you have many tools (dozens to hundreds), you cannot send all of them
to the model every turn — token budget and performance degrade.

Instead, you select a relevant subset before each API call.

Key concepts:
- Tool registries: a catalogue of all available tools with metadata
- Relevance search: vector similarity or keyword matching against tool descriptions
- Dynamic injection: select N most relevant tools per query
- Two-stage architecture: retrieval pass + execution pass
- Tool tags and categories for coarse filtering before similarity search

This experiment implements a tool registry with 30+ tools across 6 categories.
For each user query, it retrieves the top-k most relevant tools and passes
only those to the model.

The selection process:
  1. Embed or bag-of-words the user query
  2. Score each tool by description similarity
  3. Filter by category if query hints are present
  4. Return top-k tools + one always-on "search_tools" meta-tool
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    name: str
    description: str
    category: str
    tags: list[str]
    input_schema: dict
    fn: Callable
    always_on: bool = False  # always included regardless of query


# ---------------------------------------------------------------------------
# Bag-of-words similarity (no ML dependencies needed for mock)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _idf(corpus: list[str]) -> dict[str, float]:
    """Compute inverse document frequency over a corpus of strings."""
    N = len(corpus)
    df: Counter = Counter()
    for doc in corpus:
        for term in set(_tokenize(doc)):
            df[term] += 1
    return {term: math.log(N / (1 + count)) for term, count in df.items()}


def bow_similarity(query: str, doc: str, idf: dict[str, float]) -> float:
    """TF-IDF cosine similarity between query and doc."""
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(doc)

    q_tf = Counter(q_tokens)
    d_tf = Counter(d_tokens)

    terms = set(q_tokens) | set(d_tokens)
    q_vec = {t: q_tf.get(t, 0) * idf.get(t, 0) for t in terms}
    d_vec = {t: d_tf.get(t, 0) * idf.get(t, 0) for t in terms}

    dot = sum(q_vec[t] * d_vec[t] for t in terms)
    q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values())) or 1
    d_norm = math.sqrt(sum(v ** 2 for v in d_vec.values())) or 1

    return dot / (q_norm * d_norm)


# ---------------------------------------------------------------------------
# Fake tool implementations (stubs — content doesn't matter for this lesson)
# ---------------------------------------------------------------------------

def _stub(name: str) -> Callable:
    def _fn(**kwargs) -> str:
        return f"[{name}] called with {kwargs}"
    _fn.__name__ = name
    return _fn


# ---------------------------------------------------------------------------
# Tool catalogue — 30 tools across 6 categories
# ---------------------------------------------------------------------------

_CATALOGUE_RAW = [
    # ── Finance ──────────────────────────────────────────────────────────
    ("get_stock_price",       "Get the current stock price for a ticker symbol.",                      "finance",    ["stock", "price", "market"]),
    ("get_market_summary",    "Get a summary of major market indices (S&P 500, NASDAQ, Dow Jones).",   "finance",    ["market", "indices", "summary"]),
    ("get_earnings_report",   "Retrieve the latest earnings report for a company.",                    "finance",    ["earnings", "revenue", "profit"]),
    ("get_analyst_consensus", "Get analyst buy/hold/sell consensus and price target for a ticker.",    "finance",    ["analyst", "rating", "target"]),
    ("convert_currency",      "Convert an amount between two currency codes (e.g. USD to EUR).",       "finance",    ["currency", "exchange", "convert"]),

    # ── Weather ───────────────────────────────────────────────────────────
    ("get_current_weather",   "Get the current weather conditions for a city.",                        "weather",    ["weather", "temperature", "forecast"]),
    ("get_forecast",          "Get a 7-day weather forecast for a city.",                              "weather",    ["forecast", "week", "future"]),
    ("get_air_quality",       "Get the air quality index (AQI) for a city.",                          "weather",    ["air", "quality", "pollution", "AQI"]),
    ("get_uv_index",          "Get the UV index and sun safety advice for a location.",               "weather",    ["UV", "sun", "sunscreen"]),
    ("get_historical_weather","Get historical weather data for a city and date range.",               "weather",    ["historical", "past", "climate"]),

    # ── Travel ────────────────────────────────────────────────────────────
    ("search_flights",        "Search for available flights between two cities on a given date.",      "travel",     ["flight", "airplane", "booking"]),
    ("get_hotel_rates",       "Get hotel availability and rates for a city and date range.",           "travel",     ["hotel", "accommodation", "lodging"]),
    ("get_visa_requirements", "Get visa requirements for a passport holder entering a country.",       "travel",     ["visa", "passport", "immigration"]),
    ("get_travel_advisory",   "Get the official travel advisory level for a country.",                "travel",     ["advisory", "safety", "travel", "warning"]),
    ("get_exchange_offices",  "Find currency exchange offices near a given location.",                "travel",     ["currency", "exchange", "location"]),

    # ── Productivity ──────────────────────────────────────────────────────
    ("create_calendar_event", "Create a new event on the user's calendar.",                           "productivity",["calendar", "event", "schedule", "meeting"]),
    ("search_calendar",       "Search calendar events by keyword or date range.",                     "productivity",["calendar", "search", "event"]),
    ("create_task",           "Create a to-do task with optional due date and priority.",             "productivity",["task", "todo", "reminder"]),
    ("send_email",            "Send an email to one or more recipients.",                             "productivity",["email", "send", "message"]),
    ("search_email",          "Search emails by keyword, sender, or date range.",                     "productivity",["email", "search", "inbox"]),

    # ── Knowledge ─────────────────────────────────────────────────────────
    ("wikipedia_search",      "Search Wikipedia and return a summary for a topic.",                   "knowledge",  ["wikipedia", "search", "information", "summary"]),
    ("get_definition",        "Get the dictionary definition of a word.",                             "knowledge",  ["definition", "dictionary", "meaning", "word"]),
    ("translate_text",        "Translate text between languages.",                                    "knowledge",  ["translate", "language", "text"]),
    ("calculate",             "Evaluate a mathematical expression.",                                  "knowledge",  ["math", "calculate", "arithmetic", "expression"]),
    ("unit_convert",          "Convert between units of measurement (length, weight, temperature).",  "knowledge",  ["unit", "convert", "measure", "length", "weight"]),

    # ── Development ───────────────────────────────────────────────────────
    ("run_python",            "Execute a Python code snippet and return stdout.",                     "development",["python", "code", "run", "execute"]),
    ("search_github",         "Search GitHub repositories by keyword.",                              "development",["github", "code", "repository", "search"]),
    ("get_npm_package",       "Get information about an npm package.",                               "development",["npm", "package", "javascript", "node"]),
    ("validate_json",         "Validate a JSON string and return any syntax errors.",                "development",["json", "validate", "syntax"]),
    ("format_code",           "Format a code snippet using the appropriate formatter.",              "development",["format", "code", "pretty", "lint"]),
]

# Build ToolDef objects
CATALOGUE: list[ToolDef] = []
for name, desc, category, tags in _CATALOGUE_RAW:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Input for this tool"}},
        "required": ["query"],
    }
    CATALOGUE.append(ToolDef(
        name=name,
        description=desc,
        category=category,
        tags=tags,
        input_schema=schema,
        fn=_stub(name),
    ))


# ---------------------------------------------------------------------------
# Tool registry with dynamic selection
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self, catalogue: list[ToolDef]):
        self._catalogue = catalogue
        self._by_name = {t.name: t for t in catalogue}

        # Pre-compute IDF over all descriptions for similarity scoring
        descriptions = [t.description + " " + " ".join(t.tags) for t in catalogue]
        self._idf = _idf(descriptions)

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[ToolDef]:
        """
        Return the top-k most relevant tools for the given query.
        Optionally filter to a specific category first.
        """
        candidates = self._catalogue
        if category_filter:
            candidates = [t for t in candidates if t.category == category_filter]

        scored = []
        query_text = query
        for tool in candidates:
            doc = tool.description + " " + " ".join(tool.tags)
            score = bow_similarity(query_text, doc, self._idf)
            scored.append((score, tool))

        scored.sort(key=lambda x: -x[0])
        return [tool for _, tool in scored[:top_k]]

    def get(self, name: str) -> ToolDef | None:
        return self._by_name.get(name)

    def to_api_schemas(self, tools: list[ToolDef]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def dispatch(self, name: str, inputs: dict) -> str:
        tool = self._by_name.get(name)
        if not tool:
            return f"Unknown tool: {name}"
        return tool.fn(**inputs)

    @property
    def all_tools(self) -> list[ToolDef]:
        return list(self._catalogue)


REGISTRY = ToolRegistry(CATALOGUE)


# ---------------------------------------------------------------------------
# Meta-tool: search_tools (always included so model can discover tools)
# ---------------------------------------------------------------------------

SEARCH_TOOLS_SCHEMA = {
    "name": "search_tools",
    "description": (
        "Search the tool catalogue for tools relevant to a task. "
        "Use this when you're not sure which tools are available or when the "
        "initially provided tools don't cover what you need. "
        "Returns a list of matching tool names and descriptions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Describe what you need to do"},
            "category": {
                "type": "string",
                "description": "Optional category filter: finance, weather, travel, productivity, knowledge, development",
            },
        },
        "required": ["query"],
    },
}


def search_tools_fn(query: str, category: str | None = None) -> str:
    results = REGISTRY.search(query, top_k=5, category_filter=category)
    lines = [f"Top {len(results)} tools matching '{query}':"]
    for tool in results:
        lines.append(f"  {tool.name} [{tool.category}]: {tool.description}")
    return "\n".join(lines)
