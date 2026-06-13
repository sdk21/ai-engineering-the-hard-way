"""
Task Auction
-------------
Tasks are auctioned to the most capable agent. Each agent bids on tasks
based on its self-assessed capability. The highest bidder wins the task.

Inspired by contract net protocol (FIPA CNP) — a classic multi-agent
coordination mechanism from the 1980s.

Pattern:
  1. Manager announces tasks with descriptions and requirements
  2. Each agent evaluates tasks and submits bids (capability score + rationale)
  3. Manager awards each task to the highest bidder
  4. Winning agents execute their tasks

Why auction?
  - Dynamic assignment: tasks go to whoever is best suited NOW
  - Specialization discovery: agents reveal their own strengths
  - Load balancing: busy agents bid lower; available agents bid higher
  - No hard-coded routing: unlike the router (exp 01), no one pre-assigns agents

Bid components:
  - Capability score: 0-10, how well-suited the agent is
  - Confidence: how certain the agent is of its capability
  - Rationale: why the agent thinks it's the right choice
  - Estimated effort: how long the task will take

Real-world applications:
  - Multi-robot task allocation
  - Distributed computing job scheduling
  - Service mesh load balancing
  - Agent marketplaces
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import re


@dataclass
class Task:
    id: str
    description: str
    requirements: list[str]          # skills/capabilities needed
    winner: Optional[str] = None
    result: str = ""
    status: str = "open"             # open | awarded | done


@dataclass
class Bid:
    agent_id: str
    task_id: str
    capability_score: float          # 0-10: how capable is this agent
    rationale: str
    estimated_effort: str = ""


@dataclass
class Agent:
    id: str
    name: str
    specialties: list[str]
    system_prompt: str


@dataclass
class AuctionSession:
    tasks: list[Task] = field(default_factory=list)
    agents: list[Agent] = field(default_factory=list)
    bids: list[Bid] = field(default_factory=list)
    awards: dict[str, str] = field(default_factory=dict)    # task_id → agent_id
    results: dict[str, str] = field(default_factory=dict)

    def display(self) -> None:
        print(f"\n  Tasks ({len(self.tasks)}), Agents ({len(self.agents)})")
        print("\n  Awards:")
        for task in self.tasks:
            agent_id = self.awards.get(task.id, "unawarded")
            agent = next((a for a in self.agents if a.id == agent_id), None)
            agent_name = agent.name if agent else agent_id
            winning_bid = next((b for b in self.bids if b.task_id == task.id and b.agent_id == agent_id), None)
            score_str = f" (bid: {winning_bid.capability_score:.1f}/10)" if winning_bid else ""
            result = self.results.get(task.id, "")
            print(f"    [{task.id}] → {agent_name}{score_str}: {task.description[:50]}")
            if result:
                print(f"        Result: {result[:80]}")

        print("\n  Bid matrix:")
        for task in self.tasks:
            task_bids = [b for b in self.bids if b.task_id == task.id]
            task_bids.sort(key=lambda b: -b.capability_score)
            print(f"    [{task.id}]:")
            for bid in task_bids:
                winner_mark = " ← WINNER" if self.awards.get(task.id) == bid.agent_id else ""
                print(f"      {bid.agent_id}: {bid.capability_score:.1f}/10 — {bid.rationale[:60]}{winner_mark}")


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

AGENTS = [
    Agent(
        id="data_agent",
        name="Data Analysis Agent",
        specialties=["data analysis", "statistics", "SQL", "visualization", "metrics"],
        system_prompt="You are a data analysis specialist. Excel at data manipulation, statistical analysis, SQL queries, and interpreting metrics.",
    ),
    Agent(
        id="code_agent",
        name="Code Generation Agent",
        specialties=["Python", "algorithms", "code review", "debugging", "architecture"],
        system_prompt="You are a code generation specialist. Excel at writing, reviewing, and debugging code in Python and other languages.",
    ),
    Agent(
        id="writing_agent",
        name="Technical Writing Agent",
        specialties=["documentation", "explanation", "blog posts", "summaries", "communication"],
        system_prompt="You are a technical writing specialist. Excel at documentation, explanations, and clear communication for technical topics.",
    ),
    Agent(
        id="research_agent",
        name="Research Agent",
        specialties=["research", "comparison", "evaluation", "literature review", "fact-checking"],
        system_prompt="You are a research specialist. Excel at gathering, comparing, and synthesizing information from multiple sources.",
    ),
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BIDDER_SYSTEM = """You are an agent evaluating whether to bid on a task.

You will be given:
1. Your specialties
2. A task description and its requirements
3. Other tasks in this auction (so you can decide where to focus)

Submit a bid with:
- capability_score: 0-10 (how well-suited you are)
- rationale: why you're the right agent (1 sentence)
- estimated_effort: e.g., "low", "medium", "high"

Return JSON:
{"capability_score": 8.5, "rationale": "...", "estimated_effort": "low"}

Be honest. Don't overbid on tasks outside your specialty."""

EXECUTOR_SYSTEM = """You are a specialist agent executing a task you won at auction.
Complete the task well. Be concise and specific. Return only the result."""


def bidder_prompt(agent: Agent, task: Task, all_tasks: list[Task]) -> str:
    other_tasks = [t for t in all_tasks if t.id != task.id]
    return (
        f"Your specialties: {', '.join(agent.specialties)}\n\n"
        f"Task to bid on:\n"
        f"  ID: {task.id}\n"
        f"  Description: {task.description}\n"
        f"  Requirements: {', '.join(task.requirements)}\n\n"
        f"Other tasks in this auction:\n" +
        "\n".join(f"  - [{t.id}] {t.description[:60]}" for t in other_tasks) +
        "\n\nSubmit your bid."
    )


def executor_prompt(agent: Agent, task: Task) -> str:
    return f"Task: {task.description}\nRequirements: {', '.join(task.requirements)}\n\nComplete this task."


def parse_bid(json_text: str, agent_id: str, task_id: str) -> Bid:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    return Bid(
        agent_id=agent_id,
        task_id=task_id,
        capability_score=float(data.get("capability_score", 5)),
        rationale=data.get("rationale", ""),
        estimated_effort=data.get("estimated_effort", "medium"),
    )


def award_tasks(tasks: list[Task], bids: list[Bid]) -> dict[str, str]:
    """Award each task to highest bidder. Simple greedy assignment."""
    awards = {}
    for task in tasks:
        task_bids = [b for b in bids if b.task_id == task.id]
        if task_bids:
            winner = max(task_bids, key=lambda b: b.capability_score)
            awards[task.id] = winner.agent_id
            task.winner = winner.agent_id
            task.status = "awarded"
    return awards


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

MOCK_TASKS = [
    Task("t1", "Analyze sales data and identify top 3 revenue drivers", ["data analysis", "statistics", "metrics"]),
    Task("t2", "Write a Python function to parse JSON logs and extract errors", ["Python", "code", "debugging"]),
    Task("t3", "Write documentation for our REST API endpoints", ["documentation", "technical writing"]),
    Task("t4", "Compare three database options for a high-write workload", ["research", "comparison", "databases"]),
]

MOCK_BIDS = [
    # data_agent bids
    Bid("data_agent", "t1", 9.5, "Data analysis and metrics identification is my core specialty"),
    Bid("data_agent", "t2", 3.0, "Can write Python but code generation isn't my strength"),
    Bid("data_agent", "t3", 4.0, "Can document data workflows but general API docs aren't my focus"),
    Bid("data_agent", "t4", 7.0, "Can evaluate database options from a data perspective"),
    # code_agent bids
    Bid("code_agent", "t1", 5.0, "Can analyze data programmatically but stats/viz not my specialty"),
    Bid("code_agent", "t2", 9.8, "Writing Python functions is my core capability"),
    Bid("code_agent", "t3", 5.5, "Can document code but technical writing isn't primary"),
    Bid("code_agent", "t4", 6.0, "Familiar with database tradeoffs from implementation experience"),
    # writing_agent bids
    Bid("writing_agent", "t1", 3.0, "Can present findings but data analysis itself isn't my strength"),
    Bid("writing_agent", "t2", 2.0, "Python code generation is outside my specialty"),
    Bid("writing_agent", "t3", 9.7, "API documentation is exactly my specialty"),
    Bid("writing_agent", "t4", 6.5, "Can write a comparison but deep technical evaluation isn't primary"),
    # research_agent bids
    Bid("research_agent", "t1", 6.0, "Can research sales data patterns but not run actual analysis"),
    Bid("research_agent", "t2", 3.5, "Can research Python patterns but not write production code"),
    Bid("research_agent", "t3", 6.0, "Can research API patterns but technical writing isn't primary"),
    Bid("research_agent", "t4", 9.6, "Comparing and evaluating options through research is my core strength"),
]

MOCK_RESULTS = {
    "t1": "Top 3 revenue drivers: (1) Enterprise tier accounts (62% of revenue, 23% of customers), (2) Annual plan upsells (18% revenue uplift vs monthly), (3) Add-on modules — specifically the Analytics add-on (highest attach rate in Q3).",
    "t2": 'def parse_errors(log_file):\n    errors = []\n    with open(log_file) as f:\n        for line in f:\n            try:\n                entry = json.loads(line)\n                if entry.get("level") == "error":\n                    errors.append(entry)\n            except json.JSONDecodeError:\n                pass\n    return errors',
    "t3": "# REST API Documentation\n\n## GET /users/{id}\nReturns user profile. Auth: Bearer token. Response: {id, name, email, created_at}.\n\n## POST /users\nCreate user. Body: {name, email, password}. Returns: 201 with user object.",
    "t4": "PostgreSQL: best for complex queries and ACID compliance; write throughput ~20K TPS with proper indexing. Cassandra: designed for high-write workloads (100K+ TPS), eventual consistency. ScyllaDB: Cassandra-compatible with 10× better performance per node. Recommendation for high-write: ScyllaDB if writes dominate; PostgreSQL with write-optimized settings if queries are complex.",
}


def mock_auction() -> AuctionSession:
    session = AuctionSession(tasks=MOCK_TASKS, agents=AGENTS, bids=MOCK_BIDS)
    session.awards = {"t1": "data_agent", "t2": "code_agent", "t3": "writing_agent", "t4": "research_agent"}
    session.results = MOCK_RESULTS
    for task in MOCK_TASKS:
        task.winner = session.awards[task.id]
        task.status = "done"
        task.result = MOCK_RESULTS[task.id]
    return session
