"""
Debate
-------
Two agents argue opposing positions on a question. A judge agent evaluates
the arguments and delivers a verdict.

Pattern:
  1. Topic is stated with two positions (pro/con, option A/B)
  2. Agent A opens with their argument
  3. Agent B rebuts and presents counter-argument
  4. Agent A responds to the rebuttal (optional additional rounds)
  5. Judge evaluates all arguments and gives a reasoned verdict

Why debate?
  - Forces exploration of both sides before a decision
  - Rebuttals expose weaknesses in arguments that a single agent misses
  - The judge must weigh evidence rather than just accepting the first answer
  - Useful for: technology choices, policy decisions, product tradeoffs

Compared to consensus voting (exp 07):
  Debate: adversarial, explicit argument/rebuttal, judge decides
  Consensus: independent opinions, no argumentation, majority rules
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import re


@dataclass
class DebateTurn:
    agent: str                # "pro", "con", or "judge"
    position: str             # what position this agent holds
    argument: str             # the argument text


@dataclass
class DebateSession:
    topic: str
    pro_position: str
    con_position: str
    turns: list[DebateTurn] = field(default_factory=list)
    verdict: str = ""
    winner: str = ""           # "pro", "con", or "neither"

    def display(self) -> None:
        print(f"\n  Topic: {self.topic}")
        print(f"  PRO: {self.pro_position}")
        print(f"  CON: {self.con_position}")
        for i, turn in enumerate(self.turns, 1):
            label = f"[{turn.agent.upper()}]"
            print(f"\n  Turn {i} {label} ({turn.position}):")
            print(f"    {turn.argument[:200]}")
        if self.verdict:
            print(f"\n  Judge's Verdict [{self.winner.upper()}]:\n  {self.verdict}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PRO_SYSTEM = """You are arguing IN FAVOR of the given position. Make the strongest possible case.
Use specific evidence, examples, and logic. Acknowledge weaknesses only to rebut them.
Keep arguments focused: 3-5 sentences."""

CON_SYSTEM = """You are arguing AGAINST the given position (taking the opposing view). Make the strongest possible case.
Use specific evidence, examples, and logic. Acknowledge weaknesses only to rebut them.
Keep arguments focused: 3-5 sentences."""

JUDGE_SYSTEM = """You are an impartial judge evaluating a debate. Read all arguments carefully.

Evaluate:
- Strength of evidence presented
- Quality of rebuttals
- Logical consistency
- Practical relevance

Return JSON:
{
  "winner": "pro|con|neither",
  "verdict": "2-3 sentence reasoned conclusion",
  "pro_strengths": "what pro argued well",
  "con_strengths": "what con argued well"
}

Be fair. The winner is whoever made the stronger overall case, not necessarily who is 'right'."""


def pro_opening_prompt(topic: str, position: str) -> str:
    return f"Topic: {topic}\nYour position: {position}\n\nPresent your opening argument."


def con_rebuttal_prompt(topic: str, position: str, pro_argument: str) -> str:
    return f"Topic: {topic}\nYour position: {position}\n\nOpponent's argument:\n{pro_argument}\n\nPresent your rebuttal and counter-argument."


def pro_response_prompt(topic: str, position: str, con_argument: str) -> str:
    return f"Topic: {topic}\nYour position: {position}\n\nOpponent's rebuttal:\n{con_argument}\n\nRespond to their rebuttal and strengthen your case."


def judge_prompt(session: DebateSession) -> str:
    lines = [f"Topic: {session.topic}", ""]
    for turn in session.turns:
        lines.append(f"[{turn.agent.upper()}] ({turn.position}):\n{turn.argument}")
        lines.append("")
    lines.append("Evaluate the debate and deliver your verdict.")
    return "\n".join(lines)


def parse_verdict(json_text: str) -> tuple[str, str]:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    return data.get("winner", "neither"), data.get("verdict", "")


# ---------------------------------------------------------------------------
# Predefined debates
# ---------------------------------------------------------------------------

DEBATES = [
    {
        "topic": "Should a startup use microservices from day one?",
        "pro": "Yes — microservices enable independent scaling and deployment",
        "con": "No — start with a monolith and extract services when needed",
    },
    {
        "topic": "Is TypeScript worth adopting for a small JavaScript project?",
        "pro": "Yes — type safety prevents bugs and improves developer experience",
        "con": "No — the added complexity and build step aren't worth it for small projects",
    },
    {
        "topic": "Should AI-generated code be reviewed differently than human-written code?",
        "pro": "Yes — AI code has distinct failure patterns that require specialized review",
        "con": "No — good code review practices apply regardless of the author",
    },
    {
        "topic": "Should teams use NoSQL databases as their primary store?",
        "pro": "Yes — NoSQL offers flexibility and horizontal scaling for modern workloads",
        "con": "No — relational databases handle most use cases better with stronger consistency guarantees",
    },
]


def mock_debate() -> DebateSession:
    session = DebateSession(
        topic=DEBATES[0]["topic"],
        pro_position=DEBATES[0]["pro"],
        con_position=DEBATES[0]["con"],
    )
    session.turns = [
        DebateTurn("pro", DEBATES[0]["pro"],
                   "Microservices from day one allow each component to scale independently — your payment service doesn't need to scale with your recommendation engine. Teams can deploy independently, reducing coordination overhead. Modern tooling (Docker, K8s) makes this feasible even for small teams, and starting correctly avoids painful refactoring later."),
        DebateTurn("con", DEBATES[0]["con"],
                   "The 'modern tooling' argument ignores the operational complexity cost: service discovery, distributed tracing, network failures, data consistency across services. A 2-person startup doesn't have a platform team. Monoliths deploy in one step; microservices require orchestration. Shopify, Stack Overflow, and Basecamp scaled to millions of users as monoliths. Extract services when you have a proven scaling bottleneck, not before."),
        DebateTurn("pro", DEBATES[0]["pro"],
                   "The operational complexity argument conflates microservices with Kubernetes — you can run microservices on simple PaaS without K8s overhead. And the 'refactor later' argument underestimates how painful it is to split a tightly-coupled monolith under load. If you know a component will need independent scaling (e.g., video processing), starting separate is cheaper than splitting later."),
    ]
    session.winner = "con"
    session.verdict = "Con makes the stronger case. The operational burden of microservices is real and well-documented, and the monolith-first approach is validated by successful large-scale companies. Pro's rebuttal about PaaS is valid but doesn't fully address the coordination overhead. The burden of proof is on microservices adoption, and Pro hasn't demonstrated a compelling reason to incur that cost from day one."
    return session
