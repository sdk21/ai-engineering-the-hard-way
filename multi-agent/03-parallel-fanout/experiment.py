"""
Parallel Fan-out
-----------------
An orchestrator dispatches multiple independent subtasks simultaneously,
waits for all to complete, then aggregates the results.

Unlike the sequential orchestrator (exp 02) where subagent results feed
into subsequent agents, fan-out is for tasks with NO dependencies between them.

Pattern:
  1. Orchestrator identifies independent subtasks
  2. All subtasks launched concurrently (ThreadPoolExecutor)
  3. Aggregator waits for all, then combines results

When to use fan-out:
  - Subtasks are independent (no output of A feeds into B)
  - Network/API latency dominates (parallel saves wall-clock time)
  - You want multiple perspectives on the same question simultaneously

Examples:
  - Research the same topic from N different angles in parallel
  - Translate a document into N languages simultaneously
  - Run N different evaluation criteria on the same output
  - Query N different data sources for the same entity
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import concurrent.futures
import time


@dataclass
class FanoutTask:
    id: str
    agent_name: str
    instruction: str
    system_prompt: str
    result: str = ""
    duration_ms: float = 0.0
    status: str = "pending"       # pending | done | failed
    error: str = ""


@dataclass
class FanoutSession:
    goal: str
    tasks: list[FanoutTask] = field(default_factory=list)
    aggregated_result: str = ""
    total_wall_time_ms: float = 0.0
    sequential_estimate_ms: float = 0.0

    def speedup(self) -> float:
        if self.total_wall_time_ms == 0:
            return 0.0
        return self.sequential_estimate_ms / self.total_wall_time_ms

    def display(self) -> None:
        print(f"\n  Goal: {self.goal}")
        print(f"  Tasks ({len(self.tasks)}) — ran in parallel:")
        for t in self.tasks:
            icon = "✓" if t.status == "done" else "✗"
            print(f"    {icon} [{t.agent_name}] {t.instruction[:60]} ({t.duration_ms:.0f}ms)")
            if t.result:
                print(f"        → {t.result[:80]}")
        if self.total_wall_time_ms:
            print(f"\n  Wall time: {self.total_wall_time_ms:.0f}ms")
            print(f"  Sequential estimate: {self.sequential_estimate_ms:.0f}ms")
            print(f"  Speedup: {self.speedup():.1f}×")
        if self.aggregated_result:
            print(f"\n  Aggregated Result:\n  {self.aggregated_result}")


# ---------------------------------------------------------------------------
# Perspective agents for the same topic
# ---------------------------------------------------------------------------

PERSPECTIVE_AGENTS = [
    {
        "name": "technical",
        "system": "You are a technical expert. Analyze the given topic from a technical implementation perspective. Be specific about technology, architecture, and tradeoffs. 2-3 sentences.",
    },
    {
        "name": "business",
        "system": "You are a business strategist. Analyze the given topic from a business value, ROI, and organizational perspective. 2-3 sentences.",
    },
    {
        "name": "risk",
        "system": "You are a risk analyst. Identify the main risks, failure modes, and mitigation strategies for the given topic. 2-3 sentences.",
    },
    {
        "name": "user_experience",
        "system": "You are a UX expert. Analyze the given topic from an end-user experience and adoption perspective. 2-3 sentences.",
    },
]

AGGREGATOR_SYSTEM = """You are an aggregator. Synthesize perspectives from multiple specialist agents into a balanced, comprehensive analysis.
Weave the perspectives together — don't just list them.
Keep the response under 150 words."""


def aggregator_prompt(goal: str, tasks: list[FanoutTask]) -> str:
    lines = [f"Topic: {goal}", "", "Specialist perspectives:"]
    for t in tasks:
        if t.result:
            lines.append(f"  [{t.agent_name}]: {t.result}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

def mock_fanout_session() -> FanoutSession:
    session = FanoutSession(goal="Adopting Kubernetes for a 20-person startup")
    session.tasks = [
        FanoutTask("t1", "technical",
                   "Analyze Kubernetes adoption from a technical perspective",
                   "", status="done", duration_ms=312,
                   result="Kubernetes provides powerful container orchestration but introduces significant complexity: YAML configuration, networking, RBAC, and storage management. A 20-person startup likely lacks the platform engineering bandwidth to manage it well without managed offerings like EKS or GKE."),
        FanoutTask("t2", "business",
                   "Analyze Kubernetes adoption from a business perspective",
                   "", status="done", duration_ms=298,
                   result="Kubernetes adoption at this stage carries high opportunity cost — the engineering time spent on infrastructure could be spent on product. The ROI is negative unless you have clear scaling needs that simpler solutions (Heroku, Railway, Fly.io) can't meet."),
        FanoutTask("t3", "risk",
                   "Analyze Kubernetes adoption risks",
                   "", status="done", duration_ms=287,
                   result="Main risks: operational complexity leading to outages, steep learning curve for small teams, and over-engineering that slows velocity. Mitigation: use a managed Kubernetes service, invest in on-call training, and set a 6-month re-evaluation checkpoint."),
        FanoutTask("t4", "user_experience",
                   "Analyze Kubernetes adoption from a developer experience perspective",
                   "", status="done", duration_ms=305,
                   result="For developers, Kubernetes introduces friction: local development becomes complex, debugging is harder, and deployment workflows require new tooling (Helm, kubectl, Skaffold). This can hurt developer velocity and satisfaction unless proper internal tooling abstracts it away."),
    ]
    session.sequential_estimate_ms = sum(t.duration_ms for t in session.tasks)
    session.total_wall_time_ms = max(t.duration_ms for t in session.tasks)
    session.aggregated_result = "Kubernetes is premature for most 20-person startups. Technically powerful but operationally expensive — your engineers will spend more time on infrastructure than product. The business ROI is negative at this scale, and developer experience degrades without dedicated platform engineering. Recommendation: use managed PaaS (Fly.io, Railway, Heroku) until you have clear scaling requirements and a dedicated infrastructure engineer. Revisit Kubernetes when you're at 50+ engineers or have specific orchestration needs that simpler tools can't meet."
    return session


EXAMPLE_GOALS = [
    "Adopting Kubernetes for a 20-person startup",
    "Switching from a monolith to microservices",
    "Building an in-house LLM vs. using an API provider",
    "Implementing event-driven architecture for a SaaS platform",
]
