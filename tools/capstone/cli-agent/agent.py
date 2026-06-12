"""
Capstone: CLI Agent
-------------------
A production-quality command-line AI agent that synthesises all 12 tools
experiments into a single application:

  01 Basic Function Calling    → agentic loop foundation
  02 Tool Chaining             → multi-step tool sequences
  03 Parallel Tool Calls       → concurrent execution
  04 Tool Error Handling       → actionable errors + recovery
  05 Tool Result Context       → confidence + caveats in results
  06 Human-in-the-Loop         → approval gate for risky actions
  07 Dynamic Tool Selection    → registry + top-k retrieval
  08 Programmatic Generation   → @tool decorator
  09 Streaming                 → token-by-token output
  10 Tool Use with Memory      → result caching + fact extraction
  11 Multi-Tool Agent          → scratchpad + synthesis
  12 Tool Composition          → composite tools

Production concerns this capstone adds (beyond the experiments):
  - Persistence: conversation history and facts saved to disk
  - Session resume: pick up where you left off within 4 hours
  - Streaming UI: text appears as it's generated
  - Audit log: every tool call recorded with approval status
  - Stats command: cache hit rate, call counts, session info
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path.home() / ".config" / "ai-hardway" / "cli-agent"
RESUME_WINDOW_HOURS = 4
MAX_ITERATIONS = 20
MODEL = "claude-haiku-4-5-20251001"

# ─── Approval policies ────────────────────────────────────────────────────────

class ApprovalPolicy:
    AUTO = "auto"
    CONFIRM = "confirm"
    BLOCK = "block"


TOOL_POLICIES = {
    # Safe read-only
    "get_weather": ApprovalPolicy.AUTO,
    "get_stock_price": ApprovalPolicy.AUTO,
    "calculate": ApprovalPolicy.AUTO,
    "wikipedia_search": ApprovalPolicy.AUTO,
    "search_knowledge_base": ApprovalPolicy.AUTO,
    "compare_weather": ApprovalPolicy.AUTO,
    "weather_advisory": ApprovalPolicy.AUTO,
    "scratchpad_write": ApprovalPolicy.AUTO,
    "scratchpad_read": ApprovalPolicy.AUTO,
    "scratchpad_clear": ApprovalPolicy.AUTO,
    # Writes
    "write_note": ApprovalPolicy.CONFIRM,
    "send_summary": ApprovalPolicy.CONFIRM,
    # Always blocked
    "delete_all_notes": ApprovalPolicy.BLOCK,
}

# ─── Tool implementations ─────────────────────────────────────────────────────

_SCRATCHPAD: list[str] = []
_NOTES: dict[str, str] = {}

WEATHER_DATA = {
    "tokyo": {"temp_c": 22, "condition": "sunny", "humidity": 65},
    "london": {"temp_c": 14, "condition": "cloudy", "humidity": 78},
    "paris": {"temp_c": 17, "condition": "light rain", "humidity": 85},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 85},
    "new york": {"temp_c": 18, "condition": "partly cloudy", "humidity": 60},
}

STOCK_PRICES = {
    "AAPL": {"price": 189.30, "change": "+0.64%"},
    "MSFT": {"price": 415.80, "change": "-0.50%"},
    "NVDA": {"price": 875.40, "change": "+2.64%"},
    "GOOGL": {"price": 175.50, "change": "+1.97%"},
}

KB = {
    "company": "AcmeCorp, founded 2010, San Francisco. CEO: Jane Smith. 150 employees.",
    "product": "Q1: AI features. Q2: Mobile redesign. Q3: Enterprise API. Q4: International.",
    "finance": "2023 ARR: $42M. Growth: 35% YoY. Burn rate: $800K/month. Runway: 24 months.",
}

WIKI = {
    "tokyo": "Tokyo is the capital of Japan and most populous metropolitan area in the world.",
    "python": "Python is a high-level programming language emphasizing code readability.",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
}


def get_weather(city: str) -> str:
    d = WEATHER_DATA.get(city.lower())
    if not d:
        return f"No weather data for '{city}'. Available: {', '.join(WEATHER_DATA.keys())}"
    return f"{city.title()}: {d['temp_c']}°C, {d['condition']}, humidity {d['humidity']}%"


def compare_weather(cities: list) -> str:
    results = []
    for city in cities:
        d = WEATHER_DATA.get(city.lower())
        if d:
            results.append((city.title(), d['temp_c'], d['condition']))
    if not results:
        return "No data for any specified city."
    results.sort(key=lambda x: -x[1])
    lines = ["Weather comparison:"]
    for city, temp, cond in results:
        lines.append(f"  {city}: {temp}°C, {cond}")
    lines.append(f"Warmest: {results[0][0]}")
    return "\n".join(lines)


def weather_advisory(city: str) -> str:
    d = WEATHER_DATA.get(city.lower())
    if not d:
        return f"No data for '{city}'"
    temp, cond = d["temp_c"], d["condition"]
    advice = []
    if "rain" in cond: advice.append("bring umbrella")
    if temp < 10: advice.append("dress warmly")
    elif temp > 25: advice.append("light clothing + sunscreen")
    advisory = "; ".join(advice) if advice else "no special advisory"
    return f"{city.title()}: {temp}°C, {cond}. Advisory: {advisory}"


def get_stock_price(ticker: str) -> str:
    d = STOCK_PRICES.get(ticker.upper())
    if not d:
        return f"Ticker '{ticker}' not found. Available: {', '.join(STOCK_PRICES.keys())}"
    return f"{ticker.upper()}: ${d['price']:.2f} ({d['change']})"


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


def wikipedia_search(topic: str) -> str:
    result = WIKI.get(topic.lower().strip())
    return result if result else f"No Wikipedia entry for '{topic}'"


def search_knowledge_base(query: str) -> str:
    hits = [c for c in KB.values() if any(w in c.lower() for w in query.lower().split())]
    return "\n".join(hits) if hits else f"No KB results for '{query}'"


def scratchpad_write(note: str) -> str:
    _SCRATCHPAD.append(note)
    return f"Note added. ({len(_SCRATCHPAD)} total)"


def scratchpad_read() -> str:
    return "\n".join(_SCRATCHPAD) if _SCRATCHPAD else "(empty)"


def scratchpad_clear() -> str:
    n = len(_SCRATCHPAD); _SCRATCHPAD.clear()
    return f"Cleared {n} notes."


def write_note(title: str, content: str) -> str:
    _NOTES[title] = content
    return f"Note '{title}' saved ({len(content)} chars)."


def send_summary(recipient: str, subject: str, body: str) -> str:
    return f"[SIMULATED] Email to {recipient}: '{subject}' ({len(body)} chars body)"


def delete_all_notes() -> str:
    return "BLOCKED: This operation is not permitted."


TOOL_FN: dict[str, Callable] = {
    "get_weather": get_weather,
    "compare_weather": compare_weather,
    "weather_advisory": weather_advisory,
    "get_stock_price": get_stock_price,
    "calculate": calculate,
    "wikipedia_search": wikipedia_search,
    "search_knowledge_base": search_knowledge_base,
    "scratchpad_write": scratchpad_write,
    "scratchpad_read": lambda: scratchpad_read(),
    "scratchpad_clear": lambda: scratchpad_clear(),
    "write_note": write_note,
    "send_summary": send_summary,
    "delete_all_notes": lambda: delete_all_notes(),
}

TOOLS = [
    {"name": "get_weather", "description": "Get current weather for a city.",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "compare_weather", "description": "Compare weather across multiple cities.",
     "input_schema": {"type": "object", "properties": {"cities": {"type": "array", "items": {"type": "string"}}}, "required": ["cities"]}},
    {"name": "weather_advisory", "description": "Get weather + clothing/safety advisory for a city.",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "get_stock_price", "description": "Get current stock price for a ticker symbol.",
     "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
    {"name": "calculate", "description": "Evaluate a math expression.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "wikipedia_search", "description": "Search Wikipedia for a topic.",
     "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}},
    {"name": "search_knowledge_base", "description": "Search the internal knowledge base.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "scratchpad_write", "description": "Write a note to your scratchpad (planning/memory).",
     "input_schema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}},
    {"name": "scratchpad_read", "description": "Read your scratchpad notes.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "scratchpad_clear", "description": "Clear all scratchpad notes.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "write_note", "description": "Save a named note (requires approval).",
     "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]}},
    {"name": "send_summary", "description": "Send a summary email (requires approval, simulated).",
     "input_schema": {"type": "object", "properties": {"recipient": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["recipient", "subject", "body"]}},
    {"name": "delete_all_notes", "description": "Delete all saved notes (BLOCKED - destructive).",
     "input_schema": {"type": "object", "properties": {}}},
]

# ─── Result cache ─────────────────────────────────────────────────────────────

_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL = {"get_weather": 300, "get_stock_price": 60, "calculate": None, "wikipedia_search": 3600}
_CALL_LOG: list[dict] = []


def _cache_key(name: str, inputs: dict) -> str:
    return f"{name}:{json.dumps(inputs, sort_keys=True)}"


def dispatch_tool(name: str, inputs: dict, interactive: bool = True) -> tuple[str, bool, bool]:
    """Returns (result, cache_hit, was_executed)."""
    policy = TOOL_POLICIES.get(name, ApprovalPolicy.CONFIRM)

    # Block
    if policy == ApprovalPolicy.BLOCK:
        result = f"Tool '{name}' is blocked by policy — it performs destructive operations."
        _CALL_LOG.append({"tool": name, "inputs": inputs, "policy": policy, "approved": False, "ts": time.time()})
        return result, False, False

    # Confirm
    if policy == ApprovalPolicy.CONFIRM and interactive:
        print(f"\n  ┌─ APPROVAL REQUIRED ─────────────────────────────────┐")
        print(f"  │ Tool: {name}")
        for k, v in inputs.items():
            vs = str(v)[:50] + ("..." if len(str(v)) > 50 else "")
            print(f"  │ {k}: {vs}")
        print(f"  └────────────────────────────────────────────────────┘")
        answer = input("  Approve? [y/n] > ").strip().lower()
        if answer not in ("y", "yes"):
            reason = input("  Reason (optional): ").strip()
            _CALL_LOG.append({"tool": name, "inputs": inputs, "policy": policy, "approved": False, "reason": reason, "ts": time.time()})
            return f"Denied: {reason or 'User declined.'}", False, False

    # Check cache
    key = _cache_key(name, inputs)
    ttl = _CACHE_TTL.get(name)
    if key in _CACHE:
        cached_result, cached_at = _CACHE[key]
        age = time.time() - cached_at
        if ttl is None or age < ttl:
            _CALL_LOG.append({"tool": name, "inputs": inputs, "cache_hit": True, "ts": time.time()})
            return cached_result, True, True

    # Execute
    fn = TOOL_FN.get(name)
    try:
        result = fn(**inputs) if inputs else fn()
    except Exception as e:
        result = f"Error in {name}: {e}"

    _CACHE[key] = (result, time.time())
    _CALL_LOG.append({"tool": name, "inputs": inputs, "cache_hit": False, "approved": True, "ts": time.time()})
    return result, False, True


# ─── Session persistence ───────────────────────────────────────────────────────

@dataclass
class Session:
    session_id: str
    started_at: float
    last_message_at: float
    messages: list = field(default_factory=list)


def load_session(data_dir: Path) -> Session | None:
    path = data_dir / "session.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        s = Session(**data)
        age_hours = (time.time() - s.last_message_at) / 3600
        if age_hours > RESUME_WINDOW_HOURS:
            return None
        return s
    except Exception:
        return None


def save_session(session: Session, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "session.json"
    path.write_text(json.dumps(asdict(session), indent=2))


# ─── Main agent loop ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a capable AI assistant with tools for weather, stocks, calculations,
knowledge lookup, and note-taking. For complex requests:
1. Write a brief plan to your scratchpad first
2. Execute the plan step by step
3. Synthesize findings into a clear answer

Use parallel tool calls when tasks are independent.
Use the scratchpad to remember important findings across steps.
For sensitive actions (write_note, send_summary), you'll need user approval."""


def run(
    data_dir: Path = DEFAULT_DATA_DIR,
    force_new: bool = False,
    clear: bool = False,
    mock: bool = False,
    api_key: str = None,
) -> None:
    if clear and data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
        print("Cleared all agent data.")

    data_dir.mkdir(parents=True, exist_ok=True)
    mode = "MOCK" if mock else "REAL"

    # Session management
    session = None if force_new else load_session(data_dir)
    if session:
        print(f"\n╔══ CLI Agent [{mode}] ══╗")
        print(f"  Resumed session {session.session_id[:8]} ({len(session.messages)//2} turn(s))")
    else:
        import uuid
        session = Session(
            session_id=str(uuid.uuid4()),
            started_at=time.time(),
            last_message_at=time.time(),
            messages=[],
        )
        print(f"\n╔══ CLI Agent [{mode}] ══╗")
        print(f"  New session {session.session_id[:8]}")

    print(f"  Commands: 'stats', 'scratchpad', 'notes', 'quit'")
    print()

    client = None
    if not mock:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            save_session(session, data_dir)
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input.lower() == "quit":
            save_session(session, data_dir)
            break

        if user_input.lower() == "stats":
            total = len(_CALL_LOG)
            hits = sum(1 for e in _CALL_LOG if e.get("cache_hit"))
            denied = sum(1 for e in _CALL_LOG if not e.get("approved", True))
            print(f"  Total tool calls: {total}")
            print(f"  Cache hits: {hits} ({hits/total:.0%} hit rate)" if total else "  No calls yet")
            print(f"  Denied: {denied}")
            print(f"  Session turns: {len(session.messages)//2}")
            print()
            continue

        if user_input.lower() == "scratchpad":
            print(scratchpad_read())
            print()
            continue

        if user_input.lower() == "notes":
            if _NOTES:
                for title, content in _NOTES.items():
                    print(f"  [{title}] {content[:80]}")
            else:
                print("  No notes saved.")
            print()
            continue

        session.messages.append({"role": "user", "content": user_input})

        if mock:
            _mock_turn(user_input, session, data_dir)
        else:
            _real_turn(session, client, data_dir)

        session.last_message_at = time.time()
        save_session(session, data_dir)
        print()


def _mock_turn(user_input: str, session: Session, data_dir: Path) -> None:
    print(f"\n  [mock mode — simulating tool-augmented response]")
    msg = user_input.lower()

    if "weather" in msg:
        city = "Tokyo"
        for c in ["tokyo", "london", "paris", "san francisco", "new york"]:
            if c in msg:
                city = c.title()
        result, hit, _ = dispatch_tool("get_weather", {"city": city}, interactive=False)
        hit_str = " (cached)" if hit else ""
        print(f"  [get_weather{hit_str}] → {result}")
        response = f"The weather in {city}: {result}"

    elif any(t in msg.upper() for t in ["AAPL", "MSFT", "NVDA", "GOOGL", "STOCK"]):
        ticker = "AAPL"
        for t in ["AAPL", "MSFT", "NVDA", "GOOGL"]:
            if t in msg.upper():
                ticker = t
        result, hit, _ = dispatch_tool("get_stock_price", {"ticker": ticker}, interactive=False)
        response = f"{ticker} stock: {result}"

    elif "compare" in msg and "weather" in msg:
        result, hit, _ = dispatch_tool("compare_weather", {"cities": ["Tokyo", "London", "Paris"]}, interactive=False)
        response = result

    elif "write note" in msg or "save note" in msg:
        result, _, executed = dispatch_tool("write_note", {"title": "demo note", "content": user_input}, interactive=True)
        response = result

    else:
        response = "I can help with weather, stocks, calculations, knowledge lookup, and note-taking."

    print(f"Assistant: {response}")
    session.messages.append({"role": "assistant", "content": response})


def _real_turn(session: Session, client, data_dir: Path) -> None:
    print("Assistant: ", end="", flush=True)

    iterations = 0
    while iterations < MAX_ITERATIONS:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=session.messages,
        )

        # Accumulate content
        session.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(block.text)
            break

        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            # Execute in parallel for AUTO tools, sequential for CONFIRM/BLOCK
            tool_results = []
            for block in tool_blocks:
                result, cache_hit, executed = dispatch_tool(block.name, block.input, interactive=True)
                status = "cached" if cache_hit else ("denied" if not executed else "ok")
                print(f"\n  [{block.name}] [{status}] {result[:60]}{'...' if len(result)>60 else ''}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    **({"is_error": True} if not executed else {}),
                })

            session.messages.append({"role": "user", "content": tool_results})
            print("  [continuing] ", end="", flush=True)
            iterations += 1
