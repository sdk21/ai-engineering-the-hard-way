"""
Parallel Tool Calls
-------------------
When the model needs multiple independent pieces of information, it can
emit several tool_use blocks in a single response. Your code executes
them (potentially concurrently) and returns all results in one message.

Key concepts:
- The model decides when tools are independent and issues them together
- Each tool_use block has a unique ID; each tool_result references that ID
- Your code returns all results as a list before the model continues
- Parallel calls reduce round-trips from N to 1 for independent lookups

This experiment uses a market research scenario:
  - get_stock_price(ticker)      → current price and change
  - get_company_info(ticker)     → name, sector, employees
  - get_analyst_rating(ticker)   → buy/hold/sell consensus
  - get_news_headlines(ticker)   → recent news items

A user asking "Give me a full report on AAPL and MSFT" can trigger
the model to call all tools for both tickers in parallel.
"""

import concurrent.futures
import time

# ---------------------------------------------------------------------------
# Fake market data
# ---------------------------------------------------------------------------

STOCKS = {
    "AAPL": {"price": 189.30, "change": +1.2, "change_pct": +0.64},
    "MSFT": {"price": 415.80, "change": -2.1, "change_pct": -0.50},
    "GOOGL": {"price": 175.50, "change": +3.4, "change_pct": +1.97},
    "AMZN": {"price": 198.10, "change": +0.8, "change_pct": +0.41},
    "NVDA": {"price": 875.40, "change": +22.5, "change_pct": +2.64},
}

COMPANIES = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "employees": 164_000, "founded": 1976},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "employees": 228_000, "founded": 1975},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "employees": 182_000, "founded": 1998},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "employees": 1_541_000, "founded": 1994},
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology", "employees": 29_600, "founded": 1993},
}

RATINGS = {
    "AAPL": {"consensus": "Buy", "buy": 28, "hold": 9, "sell": 3, "avg_target": 210.00},
    "MSFT": {"consensus": "Strong Buy", "buy": 35, "hold": 5, "sell": 1, "avg_target": 460.00},
    "GOOGL": {"consensus": "Buy", "buy": 30, "hold": 7, "sell": 2, "avg_target": 200.00},
    "AMZN": {"consensus": "Strong Buy", "buy": 38, "hold": 3, "sell": 0, "avg_target": 230.00},
    "NVDA": {"consensus": "Strong Buy", "buy": 40, "hold": 2, "sell": 1, "avg_target": 1000.00},
}

NEWS = {
    "AAPL": [
        "Apple unveils new AI features for iPhone lineup",
        "App Store revenue beats estimates in Q4",
        "Apple expands manufacturing partnerships in India",
    ],
    "MSFT": [
        "Microsoft Copilot adoption accelerates across enterprise",
        "Azure cloud revenue grows 28% year-over-year",
        "Microsoft acquires AI startup for $1.2B",
    ],
    "GOOGL": [
        "Google Search integrates Gemini AI across all results",
        "YouTube ad revenue rebounds after two quarters of decline",
        "Waymo expands autonomous taxi service to three new cities",
    ],
    "AMZN": [
        "Amazon AWS introduces new GPU cluster for AI workloads",
        "Prime membership reaches record 230 million globally",
        "Amazon Fresh expands to 50 new markets",
    ],
    "NVDA": [
        "NVIDIA announces next-generation Blackwell Ultra GPUs",
        "Data center revenue surpasses $30B quarterly for first time",
        "NVIDIA partners with every major cloud provider for AI infrastructure",
    ],
}


# ---------------------------------------------------------------------------
# Tool implementations (with artificial latency to show parallel speedup)
# ---------------------------------------------------------------------------

def get_stock_price(ticker: str, _simulate_latency: bool = False) -> dict:
    if _simulate_latency:
        time.sleep(0.3)
    ticker = ticker.upper()
    data = STOCKS.get(ticker)
    if not data:
        return {"error": f"No price data for ticker '{ticker}'"}
    sign = "+" if data["change"] >= 0 else ""
    return {
        "ticker": ticker,
        "price": f"${data['price']:.2f}",
        "change": f"{sign}{data['change']:.2f}",
        "change_pct": f"{sign}{data['change_pct']:.2f}%",
    }


def get_company_info(ticker: str, _simulate_latency: bool = False) -> dict:
    if _simulate_latency:
        time.sleep(0.3)
    ticker = ticker.upper()
    data = COMPANIES.get(ticker)
    if not data:
        return {"error": f"No company data for ticker '{ticker}'"}
    return {
        "ticker": ticker,
        "name": data["name"],
        "sector": data["sector"],
        "employees": f"{data['employees']:,}",
        "founded": data["founded"],
    }


def get_analyst_rating(ticker: str, _simulate_latency: bool = False) -> dict:
    if _simulate_latency:
        time.sleep(0.3)
    ticker = ticker.upper()
    data = RATINGS.get(ticker)
    if not data:
        return {"error": f"No analyst data for ticker '{ticker}'"}
    return {
        "ticker": ticker,
        "consensus": data["consensus"],
        "buy_ratings": data["buy"],
        "hold_ratings": data["hold"],
        "sell_ratings": data["sell"],
        "avg_price_target": f"${data['avg_target']:.2f}",
    }


def get_news_headlines(ticker: str, _simulate_latency: bool = False) -> dict:
    if _simulate_latency:
        time.sleep(0.3)
    ticker = ticker.upper()
    headlines = NEWS.get(ticker)
    if not headlines:
        return {"error": f"No news for ticker '{ticker}'"}
    return {
        "ticker": ticker,
        "headlines": headlines,
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_stock_price",
        "description": (
            "Get the current stock price and daily change for a ticker symbol. "
            "Returns the latest price, change in dollars, and change percentage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_company_info",
        "description": (
            "Get background information about a company by ticker symbol. "
            "Returns company name, sector, number of employees, and founding year."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_analyst_rating",
        "description": (
            "Get the Wall Street analyst consensus rating for a stock. "
            "Returns buy/hold/sell counts, consensus label, and average price target."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_news_headlines",
        "description": (
            "Get the latest news headlines for a company by ticker symbol. "
            "Returns the 3 most recent headlines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
]

TOOL_FN = {
    "get_stock_price": get_stock_price,
    "get_company_info": get_company_info,
    "get_analyst_rating": get_analyst_rating,
    "get_news_headlines": get_news_headlines,
}


def dispatch_tool(name: str, inputs: dict) -> str:
    fn = TOOL_FN.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    result = fn(**inputs)
    if isinstance(result, dict):
        if "error" in result:
            return f"Error: {result['error']}"
        lines = []
        for k, v in result.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)
    return str(result)


def dispatch_tool_parallel(tool_calls: list[tuple[str, dict]]) -> list[str]:
    """Execute multiple tool calls concurrently using a thread pool."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(dispatch_tool, name, inputs) for name, inputs in tool_calls]
        return [f.result() for f in futures]
