"""
Agent Router
-------------
A router classifies an incoming request and dispatches it to the appropriate
specialized agent. The simplest form of multi-agent orchestration.

Three routing strategies:
  1. Rule-based: keyword matching — zero latency, brittle
  2. LLM-based: model classifies intent — accurate, flexible, one extra call
  3. Embedding-based: cosine similarity to example queries — fast, scalable

Each subagent has:
  - A narrow system prompt focused on its domain
  - Only the tools it needs (simulated here)
  - A clear task scope

Key insight:
  A focused agent with a narrow prompt outperforms a general agent on its
  specific domain. Routing gets you specialization without sacrificing breadth.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re


class Route(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    REFUND = "refund"
    GENERAL = "general"


@dataclass
class RoutingResult:
    route: Route
    confidence: str = "high"          # "high", "medium", "low"
    reasoning: str = ""
    strategy: str = ""                 # "rule", "llm"


@dataclass
class AgentResponse:
    route: Route
    agent_name: str
    response: str
    routing_result: Optional[RoutingResult] = None


# ---------------------------------------------------------------------------
# Subagent system prompts
# ---------------------------------------------------------------------------

AGENTS = {
    Route.BILLING: {
        "name": "Billing Agent",
        "system": """You are a billing specialist. You help customers with:
- Invoice questions and payment history
- Subscription plan changes
- Charge disputes and billing errors
- Payment method updates

Be concise and factual. If you need account information you don't have, say so.""",
    },
    Route.TECHNICAL: {
        "name": "Technical Support Agent",
        "system": """You are a technical support specialist. You help customers with:
- Product bugs and unexpected behavior
- Integration and API issues
- Performance problems
- Feature how-to questions

Provide clear troubleshooting steps. Ask clarifying questions if needed.""",
    },
    Route.REFUND: {
        "name": "Refund Agent",
        "system": """You are a refund and returns specialist. You help customers with:
- Refund eligibility and policy
- Processing refund requests
- Return instructions
- Cancellation requests

Be empathetic. State policy clearly but fairly.""",
    },
    Route.GENERAL: {
        "name": "General Agent",
        "system": """You are a helpful customer support generalist. Handle questions
that don't fit billing, technical, or refund categories. If a question belongs
to a specialist, say which team can better help.""",
    },
}

# ---------------------------------------------------------------------------
# Routing strategy 1: Rule-based
# ---------------------------------------------------------------------------

BILLING_KEYWORDS = ["invoice", "billing", "charge", "payment", "subscription", "plan", "overcharged", "fee", "cost", "price"]
TECHNICAL_KEYWORDS = ["broken", "error", "bug", "crash", "not working", "issue", "problem", "api", "integration", "slow", "timeout"]
REFUND_KEYWORDS = ["refund", "return", "cancel", "cancellation", "money back", "reimburse", "reimbursement"]


def rule_based_route(message: str) -> RoutingResult:
    msg = message.lower()
    scores = {
        Route.BILLING: sum(1 for kw in BILLING_KEYWORDS if kw in msg),
        Route.TECHNICAL: sum(1 for kw in TECHNICAL_KEYWORDS if kw in msg),
        Route.REFUND: sum(1 for kw in REFUND_KEYWORDS if kw in msg),
    }
    best_route = max(scores, key=scores.get)
    best_score = scores[best_route]
    if best_score == 0:
        return RoutingResult(route=Route.GENERAL, confidence="high",
                             reasoning="No keywords matched", strategy="rule")
    confidence = "high" if best_score >= 2 else "medium" if best_score == 1 else "low"
    return RoutingResult(route=best_route, confidence=confidence,
                         reasoning=f"Matched {best_score} keyword(s)", strategy="rule")


# ---------------------------------------------------------------------------
# Routing strategy 2: LLM-based
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = """You are a customer support request classifier. Classify the request into exactly one category.

Categories:
- billing: questions about invoices, charges, payments, subscription plans, pricing
- technical: bugs, errors, crashes, API issues, performance problems, how-to questions
- refund: refund requests, cancellations, returns, money-back requests
- general: anything that doesn't clearly fit the above

Return JSON: {"route": "billing|technical|refund|general", "confidence": "high|medium|low", "reasoning": "one sentence"}"""


def llm_router_prompt(message: str) -> str:
    return f"Customer message: {message}"


def parse_routing_result(json_text: str, strategy: str = "llm") -> RoutingResult:
    import json
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    route_str = data.get("route", "general").lower()
    route = Route(route_str) if route_str in [r.value for r in Route] else Route.GENERAL
    return RoutingResult(
        route=route,
        confidence=data.get("confidence", "medium"),
        reasoning=data.get("reasoning", ""),
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Mock responses
# ---------------------------------------------------------------------------

MOCK_CONVERSATIONS = [
    ("I was charged twice for my subscription last month.",
     Route.BILLING, "Billing Agent",
     "I'm sorry to hear about the double charge. This typically happens when a payment retry is triggered. I'll look into your account and process a refund for the duplicate charge within 3-5 business days."),
    ("The API keeps returning 504 errors when I call the /export endpoint.",
     Route.TECHNICAL, "Technical Support Agent",
     "504 errors on /export usually indicate a timeout — the export job is taking longer than the gateway allows. Try: (1) paginating your export request with smaller date ranges, (2) using the async export endpoint /export/async which returns a job ID you can poll."),
    ("I'd like a refund for my last month's charge, the product didn't meet my expectations.",
     Route.REFUND, "Refund Agent",
     "I understand your frustration. Our refund policy allows full refunds within 30 days of charge. Since you're within the window, I can process a full refund to your original payment method. It will appear in 5-7 business days."),
    ("What integrations do you support?",
     Route.GENERAL, "General Agent",
     "We support integrations with Slack, Zapier, GitHub, Jira, and all major CRM platforms via our REST API. Full documentation is at docs.example.com/integrations."),
]


def mock_route_and_respond(message: str) -> AgentResponse:
    for msg, route, agent_name, response in MOCK_CONVERSATIONS:
        if message == msg:
            routing = rule_based_route(message)
            routing.strategy = "mock"
            return AgentResponse(route=route, agent_name=agent_name,
                                 response=response, routing_result=routing)
    # Default
    routing = rule_based_route(message)
    return AgentResponse(route=routing.route,
                         agent_name=AGENTS[routing.route]["name"],
                         response="[Mock response not available for this input]",
                         routing_result=routing)


EXAMPLE_MESSAGES = [
    "I was charged twice for my subscription last month.",
    "The API keeps returning 504 errors when I call the /export endpoint.",
    "I'd like a refund for my last month's charge, the product didn't meet my expectations.",
    "What integrations do you support?",
]
