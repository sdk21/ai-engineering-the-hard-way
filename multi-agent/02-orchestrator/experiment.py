"""
Orchestrator + Subagents
-------------------------
An orchestrator receives a complex goal, decomposes it into subtasks,
delegates each subtask to a specialized subagent, and assembles the results.

Unlike the router (exp 01) which routes to ONE agent, the orchestrator
coordinates MULTIPLE agents, each handling a different part of the task.

Pattern:
  1. Orchestrator receives goal
  2. Orchestrator decomposes into subtasks (via LLM call)
  3. Each subtask is dispatched to the appropriate subagent
  4. Orchestrator assembles results into a final response

Subagents in this experiment:
  - researcher: looks up facts and information
  - analyst:    interprets data, draws conclusions
  - writer:     formats results into clear prose
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import re


@dataclass
class Subtask:
    id: str
    agent: str                    # which subagent handles this
    instruction: str              # what to do
    result: str = ""
    status: str = "pending"       # pending | done | failed


@dataclass
class OrchestrationSession:
    goal: str
    subtasks: list[Subtask] = field(default_factory=list)
    final_response: str = ""

    def display(self) -> None:
        print(f"\n  Goal: {self.goal}")
        print(f"  Subtasks ({len(self.subtasks)}):")
        for st in self.subtasks:
            icon = "✓" if st.status == "done" else "✗" if st.status == "failed" else "○"
            print(f"    {icon} [{st.agent}] {st.instruction[:70]}")
            if st.result:
                print(f"        → {st.result[:80]}")
        if self.final_response:
            print(f"\n  Final Response:\n  {self.final_response}")


# ---------------------------------------------------------------------------
# Subagent definitions
# ---------------------------------------------------------------------------

SUBAGENTS = {
    "researcher": {
        "system": """You are a research specialist. Given a research question or topic,
provide accurate, factual information. Be specific and cite key facts.
Keep responses concise (2-4 sentences).""",
    },
    "analyst": {
        "system": """You are a data analyst and strategic thinker. Given information or data,
interpret it, identify patterns, and draw actionable conclusions.
Keep responses concise (2-4 sentences).""",
    },
    "writer": {
        "system": """You are a professional writer. Given structured information and findings,
craft a clear, well-organized response for a general audience.
Write in plain language. 2-4 sentences unless asked for more.""",
    },
}

# ---------------------------------------------------------------------------
# Orchestrator prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM = """You are an orchestrator. Given a goal, decompose it into 2-4 subtasks.
Each subtask should be assigned to one of these agents: researcher, analyst, writer.

- researcher: finding facts, looking up information
- analyst: interpreting information, drawing conclusions, comparing options
- writer: formatting final output, writing summaries, composing responses

Return JSON:
{
  "subtasks": [
    {"id": "s1", "agent": "researcher", "instruction": "..."},
    {"id": "s2", "agent": "analyst", "instruction": "..."},
    {"id": "s3", "agent": "writer", "instruction": "..."}
  ]
}

Each instruction should include context from prior subtasks where relevant.
Keep instructions specific and actionable."""

ASSEMBLER_SYSTEM = """You are a final assembler. Given a goal and the results from multiple specialist agents,
write a cohesive, complete final response. Integrate all findings naturally.
Do not list the subtasks — just write the answer."""


def orchestrator_prompt(goal: str) -> str:
    return f"Goal: {goal}"


def subagent_prompt(instruction: str, prior_results: list[tuple[str, str]]) -> str:
    context = ""
    if prior_results:
        context = "\n\nContext from prior steps:\n" + "\n".join(
            f"  [{agent}]: {result[:100]}" for agent, result in prior_results
        )
    return f"{instruction}{context}"


def assembler_prompt(goal: str, subtasks: list[Subtask]) -> str:
    lines = [f"Goal: {goal}", "", "Specialist findings:"]
    for st in subtasks:
        if st.result:
            lines.append(f"  [{st.agent}]: {st.result}")
    return "\n".join(lines)


def parse_subtasks(json_text: str) -> list[Subtask]:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    return [
        Subtask(id=s.get("id", f"s{i}"), agent=s["agent"], instruction=s["instruction"])
        for i, s in enumerate(data.get("subtasks", []), 1)
    ]


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

def mock_orchestration() -> OrchestrationSession:
    session = OrchestrationSession(goal="What is FastAPI and should I use it for my new REST API project?")
    session.subtasks = [
        Subtask("s1", "researcher", "Research what FastAPI is: origin, key features, ecosystem",
                result="FastAPI is a modern Python web framework created by Sebastián Ramírez in 2018. It's built on Starlette and Pydantic, supports async natively, and auto-generates OpenAPI/Swagger docs.",
                status="done"),
        Subtask("s2", "analyst", "Analyze FastAPI's strengths and weaknesses for REST API development",
                result="FastAPI's strengths: async performance, type safety via Pydantic, auto-documentation. Weakness: smaller ecosystem than Flask/Django; async can add complexity for simple CRUD apps.",
                status="done"),
        Subtask("s3", "writer", "Write a recommendation on whether to use FastAPI for a new REST API",
                result="FastAPI is an excellent choice for new REST API projects, especially if you value type safety, automatic documentation, and async performance. Choose Flask if you need maximum simplicity; choose Django if you need a full-stack framework with built-in ORM.",
                status="done"),
    ]
    session.final_response = "FastAPI is a modern Python web framework known for high performance, native async support, and automatic OpenAPI documentation. For a new REST API project, FastAPI is the recommended choice — it enforces type safety via Pydantic, generates interactive docs automatically, and performs excellently under load. If your team is new to async Python or needs maximum simplicity, Flask remains a solid alternative."
    return session


EXAMPLE_GOALS = [
    "What is FastAPI and should I use it for my new REST API project?",
    "Explain the CAP theorem and when to choose CP vs AP systems.",
    "What are the tradeoffs between PostgreSQL and MongoDB for a user data store?",
    "Summarize what Kubernetes is and when a startup should adopt it.",
]
