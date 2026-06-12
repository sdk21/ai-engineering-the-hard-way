"""
Tool Result Context
-------------------
Demonstrates how tool results can carry structured context — metadata,
confidence scores, provenance, caveats, and follow-up suggestions —
that shapes the model's subsequent reasoning without explicit instructions.

Key concepts:
- Results are not just values: they can carry warnings, confidence, sources
- The model reads and reasons over all fields in a result, not just the answer
- Well-structured results guide the model to give better, more honest answers
- Context fields turn a raw result into a reasoning artifact

Compare three result styles for the same underlying data:
  1. Bare   — just the value, no context
  2. Simple — value + confidence + source
  3. Rich   — value + confidence + source + caveats + suggested_followups

The model's response quality improves significantly from bare to rich,
even though the core data is identical.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Result quality levels
# ---------------------------------------------------------------------------

class ResultStyle:
    BARE = "bare"        # just the value
    SIMPLE = "simple"    # value + confidence + source
    RICH = "rich"        # value + confidence + source + caveats + follow-ups


# ---------------------------------------------------------------------------
# Contextual result structure
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """
    Wraps a tool result with reasoning-relevant metadata.
    The model reads all of this — every field influences its response.
    """
    value: Any                              # The actual answer
    confidence: float = 1.0                # 0.0 = unreliable, 1.0 = certain
    source: str = ""                        # Where this data came from
    freshness: str = ""                     # How current the data is
    caveats: list[str] = field(default_factory=list)   # Known limitations
    suggested_followups: list[str] = field(default_factory=list)  # What to ask next
    data_type: str = ""                     # Machine-readable type hint

    def to_string(self, style: str = ResultStyle.RICH) -> str:
        """Serialize to a string the model will read as a tool result."""
        if style == ResultStyle.BARE:
            return str(self.value)

        if style == ResultStyle.SIMPLE:
            lines = [str(self.value)]
            if self.source:
                lines.append(f"source: {self.source}")
            if self.confidence < 1.0:
                lines.append(f"confidence: {self.confidence:.0%}")
            return "\n".join(lines)

        # RICH: full context
        lines = [str(self.value)]
        if self.source:
            lines.append(f"source: {self.source}")
        if self.freshness:
            lines.append(f"data_freshness: {self.freshness}")
        if self.confidence < 1.0:
            lines.append(f"confidence: {self.confidence:.0%} — interpret with caution")
        elif self.confidence == 1.0:
            lines.append(f"confidence: high")
        if self.caveats:
            lines.append("caveats:")
            for c in self.caveats:
                lines.append(f"  - {c}")
        if self.suggested_followups:
            lines.append("suggested_followups:")
            for s in self.suggested_followups:
                lines.append(f"  - {s}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool implementations returning ToolContext
# ---------------------------------------------------------------------------

def get_population(city: str) -> ToolContext:
    """City population data — varying confidence and freshness."""
    data = {
        "tokyo": {
            "value": "Population: 13.96 million (city proper), 37.4 million (greater metro)",
            "confidence": 0.95,
            "source": "Statistics Bureau of Japan",
            "freshness": "2023 census estimate",
            "caveats": [
                "City proper boundary excludes many effectively urban areas",
                "Metro population definition varies by source (±5M)",
            ],
            "followups": [
                "get_population('osaka') — for comparison",
                "get_urban_density('tokyo') — for density per km²",
            ],
        },
        "springfield": {
            "value": "Population: 114,230 (city) — Note: multiple US cities named Springfield",
            "confidence": 0.40,
            "source": "US Census Bureau",
            "freshness": "2020 census",
            "caveats": [
                "Ambiguous: there are 34 Springfields in the US; data is for Springfield, IL",
                "Which Springfield did you mean? MO (167k), OH (58k), MA (154k) also exist",
            ],
            "followups": [
                "Clarify which state's Springfield you mean",
                "get_population('springfield, mo') — for Missouri",
            ],
        },
        "silicon valley": {
            "value": "No single population figure — Silicon Valley is a region, not a city",
            "confidence": 0.10,
            "source": "N/A",
            "freshness": "N/A",
            "caveats": [
                "Silicon Valley spans Santa Clara County + parts of San Mateo County",
                "Estimated regional population: ~3-4 million depending on boundary definition",
            ],
            "followups": [
                "get_population('san jose') — the largest city in the region",
                "get_population('palo alto') — the cultural center",
            ],
        },
    }

    key = city.lower().strip()
    entry = data.get(key)

    if not entry:
        return ToolContext(
            value=f"No population data found for '{city}'",
            confidence=0.0,
            source="internal database",
            caveats=["City not in database"],
            suggested_followups=["Try a major city name", "Check spelling"],
        )

    return ToolContext(
        value=entry["value"],
        confidence=entry["confidence"],
        source=entry["source"],
        freshness=entry["freshness"],
        caveats=entry["caveats"],
        suggested_followups=entry["followups"],
    )


def get_stock_info(ticker: str) -> ToolContext:
    """Stock data — real-time simulated, with staleness and disclaimers."""
    import random
    random.seed(hash(ticker) % 1000)  # deterministic for demo

    stocks = {
        "NVDA": {"price": 875.40, "pe": 65.2, "sector": "Technology"},
        "GME":  {"price": 14.20,  "pe": None, "sector": "Consumer Cyclical"},
        "BYND": {"price": 7.10,   "pe": None, "sector": "Consumer Staples"},
    }

    ticker = ticker.upper()
    entry = stocks.get(ticker)

    if not entry:
        return ToolContext(
            value=f"Ticker '{ticker}' not found",
            confidence=0.0,
            caveats=["Verify the ticker symbol is correct", "Delisted stocks are not included"],
        )

    price = entry["price"]
    pe = entry["pe"]
    sector = entry["sector"]

    value_lines = [
        f"ticker: {ticker}",
        f"price: ${price:.2f}",
        f"sector: {sector}",
    ]
    if pe:
        value_lines.append(f"P/E ratio: {pe}")
    else:
        value_lines.append("P/E ratio: N/A (company not profitable)")

    # Build context based on the stock's characteristics
    caveats = ["This is simulated data for educational purposes"]
    confidence = 0.9

    if ticker in ("GME", "BYND"):
        confidence = 0.5
        caveats += [
            "This stock has high volatility and speculative characteristics",
            "Fundamental analysis (P/E) is not applicable — company is unprofitable",
            "Price can be heavily influenced by social media sentiment",
        ]

    followups = [
        f"get_analyst_rating('{ticker}') — for Wall Street consensus",
        f"get_earnings_history('{ticker}') — for profitability trend",
    ]

    return ToolContext(
        value="\n".join(value_lines),
        confidence=confidence,
        source="Simulated market data feed",
        freshness="Delayed 15 minutes (simulated)",
        caveats=caveats,
        suggested_followups=followups,
    )


def translate_text(text: str, target_language: str) -> ToolContext:
    """Translation — with back-translation confidence and cultural notes."""
    translations = {
        ("hello", "japanese"): {
            "value": "こんにちは (Konnichiwa)",
            "confidence": 0.98,
            "caveats": [
                "Konnichiwa is appropriate for daytime greetings (afternoon)",
                "Ohayou gozaimasu is used in the morning",
                "Konbanwa is for evening",
            ],
            "followups": ["translate_text('good morning', 'japanese')", "translate_text('goodbye', 'japanese')"],
        },
        ("hello", "arabic"): {
            "value": "مرحبا (Marhaba) or السلام عليكم (As-salamu alaykum)",
            "confidence": 0.85,
            "caveats": [
                "Marhaba is secular and informal",
                "As-salamu alaykum is the Islamic greeting, widely used but has religious connotation",
                "The appropriate choice depends on the social context",
                "Arabic has significant dialectal variation — this is Modern Standard Arabic",
            ],
            "followups": ["translate_text('thank you', 'arabic')"],
        },
    }

    key = (text.lower().strip(), target_language.lower().strip())
    entry = translations.get(key)

    if not entry:
        return ToolContext(
            value=f"[Translation of '{text}' to {target_language} — not in demo dataset]",
            confidence=0.0,
            caveats=["This demo only includes a small set of example translations"],
            suggested_followups=["Try 'hello' in 'japanese' or 'arabic'"],
        )

    return ToolContext(
        value=entry["value"],
        confidence=entry["confidence"],
        source="Simulated translation engine",
        freshness="Static",
        caveats=entry["caveats"],
        suggested_followups=entry["followups"],
    )


# ---------------------------------------------------------------------------
# Tool registry — supports style parameter for experimentation
# ---------------------------------------------------------------------------

_RESULT_STYLE = ResultStyle.RICH  # default; change to see the difference


def set_result_style(style: str):
    global _RESULT_STYLE
    _RESULT_STYLE = style


TOOLS = [
    {
        "name": "get_population",
        "description": (
            "Get population data for a city. Returns population figures along with "
            "confidence level, data source, freshness, caveats about the data quality, "
            "and suggested follow-up queries. Read all fields before answering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'Springfield'"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_stock_info",
        "description": (
            "Get stock information for a ticker symbol. Returns price, sector, P/E ratio, "
            "data freshness, confidence level, and caveats. "
            "High-volatility stocks will include specific risk caveats — surface these to the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g. 'NVDA', 'GME'"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "translate_text",
        "description": (
            "Translate a word or phrase to a target language. Returns the translation "
            "along with cultural context, usage caveats, and alternative forms. "
            "Include relevant cultural notes in your response."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "target_language": {"type": "string", "description": "Target language, e.g. 'japanese', 'arabic'"},
            },
            "required": ["text", "target_language"],
        },
    },
]

TOOL_FN = {
    "get_population": get_population,
    "get_stock_info": get_stock_info,
    "translate_text": translate_text,
}


def dispatch_tool(name: str, inputs: dict, style: str = None) -> str:
    fn = TOOL_FN.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    ctx = fn(**inputs)
    return ctx.to_string(style or _RESULT_STYLE)
