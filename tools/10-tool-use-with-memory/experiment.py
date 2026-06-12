"""
Tool Use with Memory
---------------------
Combines the tool vertical with the memory vertical: tool results are
stored in memory and retrieved in future turns, enabling an agent that
builds up knowledge from its own tool-use history.

Key concepts:
- Tool results as memory: every tool call and its result is stored
- Memory-augmented tool dispatch: check memory before calling a tool
- Observation injection: past tool results inform future tool arguments
- Tool-derived entity extraction: extract facts from tool results
- Reducing redundant API calls via cached tool results

Two memory strategies for tool results:
  1. Semantic cache — embed the tool call signature and result; retrieve
     by similarity on future calls with similar arguments
  2. Fact extraction — parse tool results to extract structured facts
     (entity → attribute → value) that can be injected into prompts

This prevents the agent from re-fetching the same data repeatedly
and allows it to accumulate domain knowledge over a session.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tool result cache (semantic caching simplified with exact + fuzzy match)
# ---------------------------------------------------------------------------

@dataclass
class CachedResult:
    tool_name: str
    arguments: dict
    result: str
    timestamp: float
    call_count: int = 1  # how many times this result was reused

    @property
    def cache_key(self) -> str:
        """Deterministic key for exact match."""
        return f"{self.tool_name}:{json.dumps(self.arguments, sort_keys=True)}"

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class ToolResultCache:
    """
    Caches tool results keyed by (tool_name, arguments).
    Supports TTL-based expiry.
    """

    DEFAULT_TTL = {
        "get_weather": 300,       # weather: 5 minutes
        "get_stock_price": 60,    # stocks: 1 minute
        "convert_currency": 120,  # fx: 2 minutes
        "wikipedia_lookup": 3600, # wiki: 1 hour
        "calculate": None,        # calculations: never expire
    }

    def __init__(self):
        self._store: dict[str, CachedResult] = {}

    def get(self, tool_name: str, arguments: dict) -> CachedResult | None:
        key = self._make_key(tool_name, arguments)
        entry = self._store.get(key)
        if entry is None:
            return None

        ttl = self.DEFAULT_TTL.get(tool_name)
        if ttl is not None and entry.age_seconds > ttl:
            del self._store[key]
            return None

        entry.call_count += 1
        return entry

    def set(self, tool_name: str, arguments: dict, result: str) -> None:
        key = self._make_key(tool_name, arguments)
        self._store[key] = CachedResult(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            timestamp=time.time(),
        )

    def _make_key(self, tool_name: str, arguments: dict) -> str:
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"

    def stats(self) -> dict:
        return {
            "total_entries": len(self._store),
            "tools": list({e.tool_name for e in self._store.values()}),
        }


# ---------------------------------------------------------------------------
# Tool-derived fact store (structured memory from tool results)
# ---------------------------------------------------------------------------

@dataclass
class ToolFact:
    entity: str
    attribute: str
    value: str
    source_tool: str
    timestamp: float = field(default_factory=time.time)


class ToolFactStore:
    """
    Extracts and stores facts derived from tool results.
    These facts can be injected into the system prompt.
    """

    def __init__(self):
        self._facts: list[ToolFact] = []

    def extract_and_store(self, tool_name: str, arguments: dict, result: str) -> list[ToolFact]:
        """Parse a tool result and extract structured facts."""
        new_facts = self._extract_facts(tool_name, arguments, result)
        for fact in new_facts:
            # Upsert — replace existing fact for same entity+attribute
            self._facts = [
                f for f in self._facts
                if not (f.entity == fact.entity and f.attribute == fact.attribute)
            ]
            self._facts.append(fact)
        return new_facts

    def _extract_facts(self, tool_name: str, arguments: dict, result: str) -> list[ToolFact]:
        facts = []

        if tool_name == "get_weather":
            city = arguments.get("city", "")
            if "error" not in result.lower():
                temp_match = re.search(r'(\d+)°([CF])', result)
                condition_match = re.search(r'(sunny|cloudy|rainy|overcast|clear|rain|snow)', result, re.I)
                if temp_match:
                    facts.append(ToolFact(city, "temperature", temp_match.group(0), tool_name))
                if condition_match:
                    facts.append(ToolFact(city, "weather_condition", condition_match.group(0), tool_name))

        elif tool_name == "get_stock_price":
            ticker = arguments.get("ticker", "")
            price_match = re.search(r'\$[\d.]+', result)
            if price_match:
                facts.append(ToolFact(ticker, "stock_price", price_match.group(0), tool_name))

        elif tool_name == "wikipedia_lookup":
            topic = arguments.get("topic", "")
            # Extract the first sentence as a summary fact
            first_sentence = result.split(".")[0].strip()
            if first_sentence and len(first_sentence) < 200:
                facts.append(ToolFact(topic, "description", first_sentence, tool_name))

        return facts

    def get_facts_for_entity(self, entity: str) -> list[ToolFact]:
        return [f for f in self._facts if f.entity.lower() == entity.lower()]

    def get_all_facts(self) -> list[ToolFact]:
        return list(self._facts)

    def to_context_string(self) -> str:
        if not self._facts:
            return ""
        lines = ["## Facts from previous tool calls"]
        for f in self._facts:
            age = time.time() - f.timestamp
            age_str = f"{int(age)}s ago" if age < 60 else f"{int(age//60)}m ago"
            lines.append(f"  {f.entity} / {f.attribute}: {f.value} (from {f.source_tool}, {age_str})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Memory-augmented tool executor
# ---------------------------------------------------------------------------

class MemoryAugmentedExecutor:
    """
    Wraps a tool dispatch function with caching + fact extraction.
    """

    def __init__(self, tool_fn_map: dict, cache_ttl_override: dict = None):
        self._fns = tool_fn_map
        self.cache = ToolResultCache()
        self.facts = ToolFactStore()
        self._call_log: list[dict] = []

    def execute(
        self,
        tool_name: str,
        arguments: dict,
        bypass_cache: bool = False,
    ) -> tuple[str, bool]:
        """
        Returns (result, cache_hit).
        """
        # Check cache
        if not bypass_cache:
            cached = self.cache.get(tool_name, arguments)
            if cached:
                self._log(tool_name, arguments, cached.result, cache_hit=True)
                return cached.result, True

        # Execute
        fn = self._fns.get(tool_name)
        if fn is None:
            result = f"Unknown tool: {tool_name}"
        else:
            try:
                result = fn(**arguments)
            except Exception as e:
                result = f"Error: {e}"

        # Store in cache
        self.cache.set(tool_name, arguments, result)

        # Extract facts
        self.facts.extract_and_store(tool_name, arguments, result)

        self._log(tool_name, arguments, result, cache_hit=False)
        return result, False

    def _log(self, tool_name: str, arguments: dict, result: str, cache_hit: bool) -> None:
        self._call_log.append({
            "tool": tool_name,
            "arguments": arguments,
            "result": result[:100],
            "cache_hit": cache_hit,
            "timestamp": time.time(),
        })

    def memory_context(self) -> str:
        return self.facts.to_context_string()

    def stats(self) -> dict:
        total = len(self._call_log)
        hits = sum(1 for e in self._call_log if e["cache_hit"])
        return {
            "total_calls": total,
            "cache_hits": hits,
            "cache_miss": total - hits,
            "hit_rate": f"{hits/total:.0%}" if total else "0%",
            **self.cache.stats(),
        }


# ---------------------------------------------------------------------------
# Example tools (same as other experiments)
# ---------------------------------------------------------------------------

FAKE_WEATHER = {
    "tokyo": "22°C, sunny, humidity 65%",
    "london": "15°C, overcast, humidity 80%",
    "paris": "17°C, light rain, humidity 85%",
    "sydney": "28°C, clear, humidity 55%",
}

FAKE_STOCKS = {
    "AAPL": "$189.30 (+0.64%)",
    "MSFT": "$415.80 (-0.50%)",
    "NVDA": "$875.40 (+2.64%)",
}

FAKE_WIKI = {
    "tokyo": "Tokyo is the capital and most populous city of Japan, with a population of approximately 14 million in the city proper.",
    "python": "Python is a high-level, general-purpose programming language known for its readability and versatility.",
}


def get_weather(city: str) -> str:
    return FAKE_WEATHER.get(city.lower(), f"No weather data for '{city}'")


def get_stock_price(ticker: str) -> str:
    return FAKE_STOCKS.get(ticker.upper(), f"No data for ticker '{ticker}'")


def wikipedia_lookup(topic: str) -> str:
    return FAKE_WIKI.get(topic.lower(), f"No Wikipedia entry for '{topic}'")


def calculate(expression: str) -> str:
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}
    def ev(n):
        if isinstance(n, ast.Constant): return n.n
        if isinstance(n, ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(f"Unsupported: {n}")
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


TOOL_FN_MAP = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "wikipedia_lookup": wikipedia_lookup,
    "calculate": calculate,
}

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "wikipedia_lookup",
        "description": "Look up a topic on Wikipedia and return a summary.",
        "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression.",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
]
