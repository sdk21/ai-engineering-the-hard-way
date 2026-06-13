"""
Replanning
-----------
When a step fails, discard the old plan and generate a new one from scratch
that accounts for the failure context.

Pattern:
  1. Generate initial plan
  2. Execute steps in order
  3. If step fails:
     a. Record failure reason
     b. Generate a new plan that works around the failure
     c. Resume execution with the new plan
  4. Limit replan attempts to avoid infinite loops

Key difference from backtracking (exp 10):
  Backtracking: undo the last assignment, try another value for the same variable
  Replanning: full plan regeneration after a catastrophic step failure

Key difference from adaptive planning (exp 11):
  Adaptive: step SUCCEEDS but reveals new info → tweak remaining steps
  Replanning: step FAILS → regenerate the whole remaining plan

Real-world examples:
  - CI pipeline fails at "deploy" step → replan to roll back and fix
  - Research step: "source unavailable" → replan to use alternative source
  - Booking step: "flight full" → replan the entire trip routing

Replan prompt design:
  The replanner sees: original goal + completed steps + failed step + failure reason
  It must generate a NEW path to the goal that avoids the same failure.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
import re


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ReplanStep:
    index: int
    description: str
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    failure_reason: str = ""
    plan_version: int = 1              # which plan version this step belongs to

    def display(self) -> None:
        icons = {
            StepStatus.PENDING: "○",
            StepStatus.RUNNING: "◐",
            StepStatus.DONE: "✓",
            StepStatus.FAILED: "✗",
            StepStatus.SKIPPED: "⊘",
        }
        icon = icons[self.status]
        version_str = f" [v{self.plan_version}]" if self.plan_version > 1 else ""
        result_str = f" → {self.result[:60]}" if self.result else ""
        failure_str = f" ✗ {self.failure_reason[:60]}" if self.failure_reason else ""
        print(f"  {icon} Step {self.index}: {self.description}{version_str}{result_str}{failure_str}")


@dataclass
class ReplanSession:
    goal: str
    history: list[ReplanStep] = field(default_factory=list)    # all steps ever (inc. failed)
    current_plan: list[ReplanStep] = field(default_factory=list)
    replan_count: int = 0
    final_answer: str = ""
    solved: bool = False

    def display(self) -> None:
        print(f"\n  Goal: {self.goal}")
        print(f"  Execution history ({len(self.history)} steps, {self.replan_count} replan(s)):")
        for step in self.history:
            step.display()
        if self.final_answer:
            print(f"\n  Final answer: {self.final_answer}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are a planning assistant. Given a goal, produce an ordered execution plan as JSON.

Return JSON:
{
  "steps": [
    {"description": "What to do"}
  ]
}

Keep plans focused: 3-5 steps."""

EXECUTOR_SYSTEM = """You are an execution assistant. Execute the given step and return its result.

IMPORTANT: You may simulate failures. If a step seems risky, you may return a failure response in the format:
  FAILED: <reason>

Otherwise return the result normally."""

REPLANNER_SYSTEM = """You are a replanning assistant. A plan step has FAILED. Your job is to create a new plan that achieves the original goal while working around the failure.

You will be given:
1. The original goal
2. Steps that completed successfully
3. The step that failed and why

Return a new plan as JSON:
{
  "steps": [
    {"description": "New step description"}
  ],
  "strategy": "Brief explanation of the new approach"
}

The new plan should:
- NOT retry the exact same approach that failed
- Build on what was already completed
- Find an alternative path to the goal"""

SYNTHESIZER_SYSTEM = """Given a goal and all completed step results, write a concise final answer."""


def planner_prompt(goal: str) -> str:
    return f"Goal: {goal}"


def executor_prompt(goal: str, step: ReplanStep, completed: list[ReplanStep]) -> str:
    context = ""
    if completed:
        context = "\n\nCompleted so far:\n" + "\n".join(
            f"  - {s.description}: {s.result[:60]}" for s in completed
        )
    return f"Goal: {goal}{context}\n\nExecute: {step.description}"


def replanner_prompt(session: ReplanSession, failed_step: ReplanStep) -> str:
    completed = [s for s in session.history if s.status == StepStatus.DONE]
    lines = [
        f"Original goal: {session.goal}",
        "",
        "Completed steps:",
        *[f"  ✓ {s.description}: {s.result[:60]}" for s in completed],
        "",
        f"FAILED step: {failed_step.description}",
        f"Failure reason: {failed_step.failure_reason}",
        "",
        "Generate a new plan to achieve the goal that works around this failure.",
    ]
    return "\n".join(lines)


def synthesizer_prompt(session: ReplanSession) -> str:
    completed = [s for s in session.history if s.status == StepStatus.DONE]
    lines = [f"Goal: {session.goal}", "", "Results:"]
    for s in completed:
        lines.append(f"  - {s.description}: {s.result}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_plan_steps(json_text: str) -> list[str]:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    return [s["description"] for s in data.get("steps", [])]


def is_failure(result: str) -> tuple[bool, str]:
    if result.strip().upper().startswith("FAILED:"):
        reason = result.strip()[7:].strip()
        return True, reason
    return False, ""


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

MOCK_GOAL = "Get the current stock price of Apple (AAPL) and calculate its P/E ratio."

def mock_replan_session() -> ReplanSession:
    session = ReplanSession(goal=MOCK_GOAL)

    # Version 1 attempt
    s1 = ReplanStep(1, "Query Yahoo Finance API for AAPL stock price", plan_version=1,
                    status=StepStatus.DONE, result="AAPL current price: $182.50")
    s2 = ReplanStep(2, "Query earnings data from SEC EDGAR API", plan_version=1,
                    status=StepStatus.FAILED,
                    failure_reason="SEC EDGAR API returned 503 Service Unavailable")
    # Version 2 (after replan)
    s3 = ReplanStep(3, "Search for AAPL trailing twelve months EPS from financial news", plan_version=2,
                    status=StepStatus.DONE, result="AAPL TTM EPS: $6.13 (from MarketWatch)")
    s4 = ReplanStep(4, "Calculate P/E ratio: price / EPS", plan_version=2,
                    status=StepStatus.DONE, result="P/E = 182.50 / 6.13 = 29.77")
    s5 = ReplanStep(5, "Summarize the findings", plan_version=2,
                    status=StepStatus.DONE,
                    result="AAPL is trading at $182.50 with a P/E ratio of approximately 29.8×")

    session.history = [s1, s2, s3, s4, s5]
    session.replan_count = 1
    session.solved = True
    session.final_answer = "Apple (AAPL) is currently trading at $182.50. With a trailing twelve-month EPS of $6.13, the P/E ratio is approximately 29.8×, which is slightly above the S&P 500 average of ~25×."
    return session


EXAMPLE_GOALS = [
    "Get the current stock price of Apple and calculate its P/E ratio.",
    "Book a flight from NYC to London for next Friday.",
    "Set up a Python development environment on a new machine.",
    "Deploy a Docker container to a cloud provider.",
]
