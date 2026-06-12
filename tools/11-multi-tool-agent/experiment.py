"""
Multi-Tool Agent
-----------------
A complete agent that wields a large, diverse tool set to accomplish
open-ended tasks requiring planning, tool selection, chaining, and synthesis.

Key concepts:
- Planning before acting: generating a sequence of steps before any tool calls
- ReAct pattern: Reason → Act → Observe → Reason → ... loop
- Tool use across different categories (data, compute, I/O, knowledge)
- Task decomposition: breaking complex requests into sub-tasks
- Result synthesis: combining multiple tool outputs into a coherent answer

The agent in this experiment is a "research assistant" that can:
  - Look up facts (Wikipedia, definitions)
  - Perform calculations
  - Fetch real-time data (weather, stocks)
  - Manage a note-taking scratchpad
  - Search a local knowledge base

What makes this an "agent" vs. just "tool use":
  - It decomposes multi-step requests autonomously
  - It decides when it has enough information to answer
  - It handles failures and retries gracefully
  - It maintains a running plan and revises it when results change expectations
"""

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Scratchpad tool — agent can write notes to itself
# ---------------------------------------------------------------------------

_SCRATCHPAD: list[str] = []


def scratchpad_write(note: str) -> str:
    """Write a note to the agent's scratchpad."""
    _SCRATCHPAD.append(f"[{int(time.time())}] {note}")
    return f"Note added. Scratchpad has {len(_SCRATCHPAD)} notes."


def scratchpad_read() -> str:
    """Read all notes from the scratchpad."""
    if not _SCRATCHPAD:
        return "Scratchpad is empty."
    return "\n".join(_SCRATCHPAD)


def scratchpad_clear() -> str:
    """Clear all notes from the scratchpad."""
    count = len(_SCRATCHPAD)
    _SCRATCHPAD.clear()
    return f"Scratchpad cleared ({count} notes removed)."


# ---------------------------------------------------------------------------
# Knowledge base tool — local "documents"
# ---------------------------------------------------------------------------

KB = {
    "company_overview": "AcmeCorp was founded in 2010. We build enterprise software. HQ is in San Francisco. CEO: Jane Smith.",
    "product_roadmap": "Q1: AI features launch. Q2: Mobile app redesign. Q3: Enterprise API. Q4: International expansion.",
    "team_structure": "Engineering: 45 people. Product: 12 people. Sales: 30 people. Total: 150 employees.",
    "financial_summary": "2023 ARR: $42M. Growth: 35% YoY. Burn rate: $800K/month. Runway: 24 months.",
}


def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for relevant documents."""
    query_lower = query.lower()
    results = []
    for doc_name, content in KB.items():
        if any(word in content.lower() for word in query_lower.split()):
            results.append(f"[{doc_name}]: {content}")
    if not results:
        return f"No documents found matching '{query}'"
    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# Data tools
# ---------------------------------------------------------------------------

WEATHER_DATA = {
    "san francisco": "16°C, foggy, humidity 85%",
    "tokyo": "22°C, sunny, humidity 65%",
    "london": "14°C, cloudy, humidity 78%",
    "new york": "18°C, partly cloudy, humidity 60%",
}


def get_weather(city: str) -> str:
    return WEATHER_DATA.get(city.lower(), f"No weather data for '{city}'")


def get_stock_price(ticker: str) -> str:
    prices = {"AAPL": "$189.30", "MSFT": "$415.80", "NVDA": "$875.40", "GOOG": "$175.50"}
    return prices.get(ticker.upper(), f"Ticker '{ticker}' not found")


def calculate(expression: str) -> str:
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}
    def ev(n):
        if isinstance(n, ast.Constant): return n.n
        if isinstance(n, ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(f"Unsupported: {n}")
    try:
        result = ev(ast.parse(expression, mode="eval").body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


WIKI = {
    "san francisco": "San Francisco is a city in California, USA, known for the Golden Gate Bridge and tech industry.",
    "tokyo": "Tokyo is the capital of Japan and the world's most populous metropolitan area.",
    "artificial intelligence": "Artificial intelligence (AI) is the simulation of human intelligence in machines.",
    "python": "Python is a high-level programming language emphasizing readability and versatility.",
}


def wikipedia_search(topic: str) -> str:
    result = WIKI.get(topic.lower().strip())
    if result:
        return result
    return f"No Wikipedia entry found for '{topic}'"


def get_definition(word: str) -> str:
    defs = {
        "algorithm": "A step-by-step procedure for solving a problem or accomplishing a task.",
        "heuristic": "A practical problem-solving approach that is not guaranteed to be optimal.",
        "latency": "The time delay between a cause and its effect in a system.",
    }
    return defs.get(word.lower(), f"No definition found for '{word}'")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "scratchpad_write",
        "description": "Write a note to your scratchpad. Use this to record your plan, intermediate results, or any information you want to remember across tool calls.",
        "input_schema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]},
    },
    {
        "name": "scratchpad_read",
        "description": "Read all notes from your scratchpad.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "scratchpad_clear",
        "description": "Clear all notes from your scratchpad.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the internal company knowledge base. Contains company overview, product roadmap, team structure, and financial summary.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "get_stock_price",
        "description": "Get current stock price for a ticker.",
        "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression.",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
    {
        "name": "wikipedia_search",
        "description": "Search Wikipedia for a topic.",
        "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
    },
    {
        "name": "get_definition",
        "description": "Get the definition of a word.",
        "input_schema": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
    },
]

TOOL_FN = {
    "scratchpad_write": scratchpad_write,
    "scratchpad_read": lambda: scratchpad_read(),
    "scratchpad_clear": lambda: scratchpad_clear(),
    "search_knowledge_base": search_knowledge_base,
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "calculate": calculate,
    "wikipedia_search": wikipedia_search,
    "get_definition": get_definition,
}


def dispatch_tool(name: str, inputs: dict) -> str:
    fn = TOOL_FN.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        return fn(**inputs)
    except Exception as e:
        return f"Error in {name}: {e}"


SYSTEM_PROMPT = """You are a research assistant with access to multiple tools.

For complex requests, use your scratchpad to:
1. Write down your plan before acting
2. Record key findings as you go
3. Synthesize everything into a final answer

Available tool categories:
- scratchpad_* : your personal notepad
- search_knowledge_base : internal company documents
- get_weather, get_stock_price : real-time data
- calculate : math
- wikipedia_search, get_definition : knowledge lookup

Work methodically. Use the scratchpad for planning. Synthesize results into clear, complete answers."""
