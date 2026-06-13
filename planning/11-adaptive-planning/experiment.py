"""
Adaptive Planning
------------------
A plan that updates itself in response to new information or changed conditions.

Standard planning (exp 04-07) produces a static plan upfront. Adaptive planning
adds a feedback loop:
  1. Execute the next step
  2. Observe the result
  3. Assess: does the result change what we should do next?
  4. If yes: update the remaining plan (insert, remove, or reorder steps)
  5. Continue

This is distinct from replanning (exp 12):
  Replanning: a step FAILS → the entire plan is regenerated from scratch
  Adaptive planning: a step SUCCEEDS but reveals NEW INFORMATION that
                     makes some future steps unnecessary or changes their order

Real-world analogy:
  You plan to visit 5 stores. At store 2 you find everything you need.
  Adaptive: remove stores 3-5 from the plan (new info changes future).
  Replanning: your car breaks down → regenerate the whole trip plan.

Key mechanism: after each step, the adapter model sees:
  - Original goal
  - Steps completed so far (with results)
  - Remaining steps
  → Decides whether to keep, modify, insert, or remove remaining steps
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
import re


class StepStatus(Enum):
    PENDING = "pending"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class AdaptiveStep:
    index: int
    description: str
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    is_inserted: bool = False          # True if added mid-execution

    def display(self) -> None:
        icons = {
            StepStatus.PENDING: "○",
            StepStatus.DONE: "✓",
            StepStatus.SKIPPED: "⊘",
            StepStatus.FAILED: "✗",
        }
        icon = icons[self.status]
        inserted = " [inserted]" if self.is_inserted else ""
        result_str = f" → {self.result[:60]}" if self.result else ""
        print(f"  {icon} Step {self.index}: {self.description}{inserted}{result_str}")


@dataclass
class AdaptivePlan:
    goal: str
    steps: list[AdaptiveStep] = field(default_factory=list)
    adaptations: int = 0               # how many times plan was modified
    final_answer: str = ""

    def display(self, title: str = "Plan") -> None:
        print(f"\n  {title}: {self.goal}")
        for step in self.steps:
            step.display()
        if self.adaptations:
            print(f"  [Plan adapted {self.adaptations} time(s)]")
        if self.final_answer:
            print(f"  Answer: {self.final_answer}")

    def pending_steps(self) -> list[AdaptiveStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    def completed_steps(self) -> list[AdaptiveStep]:
        return [s for s in self.steps if s.status == StepStatus.DONE]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are a planning assistant. Given a goal, produce an ordered plan as JSON.

Return JSON:
{
  "steps": [
    {"description": "What to do"},
    ...
  ]
}

Keep plans focused: 3-6 steps."""

EXECUTOR_SYSTEM = """You are an execution assistant. Execute the given step and return its result concisely."""

ADAPTER_SYSTEM = """You are a plan adaptation assistant. You are given:
1. The original goal
2. Steps completed so far (with results)
3. The remaining steps

Your job: decide if the remaining plan should be modified given what we've learned.

Return JSON:
{
  "adapt": true/false,
  "reason": "why adaptation is needed (or 'no changes needed')",
  "new_remaining_steps": [
    {"description": "step description"},
    ...
  ]
}

If adapt=false, new_remaining_steps will be ignored.
Only adapt when the completed steps reveal information that genuinely changes what's needed next.
Common adaptations: skip now-unnecessary steps, insert new steps, reorder."""

SYNTHESIZER_SYSTEM = """Given a goal and the results of all completed steps, write a concise final answer."""


def planner_prompt(goal: str) -> str:
    return f"Goal: {goal}"


def executor_prompt(goal: str, step: AdaptiveStep, completed: list[AdaptiveStep]) -> str:
    context = ""
    if completed:
        context = "\n\nPrevious results:\n" + "\n".join(
            f"  Step {s.index}: {s.description} → {s.result[:60]}" for s in completed
        )
    return f"Goal: {goal}{context}\n\nExecute: {step.description}"


def adapter_prompt(plan: AdaptivePlan) -> str:
    completed = plan.completed_steps()
    pending = plan.pending_steps()
    lines = [
        f"Goal: {plan.goal}",
        "",
        "Completed steps:",
        *[f"  Step {s.index}: {s.description} → {s.result[:80]}" for s in completed],
        "",
        "Remaining steps:",
        *[f"  Step {s.index}: {s.description}" for s in pending],
    ]
    return "\n".join(lines)


def synthesizer_prompt(plan: AdaptivePlan) -> str:
    completed = [s for s in plan.steps if s.status == StepStatus.DONE]
    lines = [f"Goal: {plan.goal}", "", "Results:"]
    for s in completed:
        lines.append(f"  Step {s.index}: {s.description}")
        if s.result:
            lines.append(f"    {s.result}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_plan(goal: str, json_text: str) -> AdaptivePlan:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    plan = AdaptivePlan(goal=goal)
    for i, s in enumerate(data.get("steps", []), 1):
        plan.steps.append(AdaptiveStep(index=i, description=s["description"]))
    return plan


def parse_adaptation(json_text: str) -> tuple[bool, str, list[str]]:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    adapt = bool(data.get("adapt", False))
    reason = data.get("reason", "")
    new_steps = [s["description"] for s in data.get("new_remaining_steps", [])]
    return adapt, reason, new_steps


# ---------------------------------------------------------------------------
# Mock adaptive plan
# ---------------------------------------------------------------------------

MOCK_GOAL = "Research the best Python web framework for a small API project and write a recommendation."

def mock_adaptive_plan() -> AdaptivePlan:
    plan = AdaptivePlan(goal=MOCK_GOAL)

    steps_data = [
        (1, "Research top Python web frameworks (Flask, FastAPI, Django)", StepStatus.DONE,
         "FastAPI and Flask are the top choices for small APIs. FastAPI has built-in async support, auto-generated docs, and type validation. Flask is simpler but lacks these features."),
        (2, "Compare performance benchmarks", StepStatus.SKIPPED,
         "Skipped: Step 1 results already contain sufficient differentiation for a small API recommendation."),
        (3, "Check community adoption and maintenance status", StepStatus.DONE,
         "FastAPI: rapidly growing, actively maintained. Flask: mature, stable. Both have large communities."),
        (4, "Write recommendation", StepStatus.DONE,
         "Recommend FastAPI for new small API projects: async-first, type-safe, auto-docs, minimal boilerplate."),
    ]

    for idx, desc, status, result in steps_data:
        s = AdaptiveStep(index=idx, description=desc, status=status, result=result)
        if idx == 2:
            # This was skipped due to adaptation
            pass
        plan.steps.append(s)

    plan.adaptations = 1
    plan.final_answer = "Recommendation: Use FastAPI for your small API project. It offers async support, automatic OpenAPI documentation, and type validation out of the box — all with minimal boilerplate. Flask is a solid alternative if you need maximum simplicity, but FastAPI's features make it the better default for new projects."
    return plan


EXAMPLE_GOALS = [
    "Research the best Python web framework for a small API project and write a recommendation.",
    "Plan a weekend trip to a nearby city: find activities, check weather, estimate costs.",
    "Debug a slow database query: profile it, identify the bottleneck, propose a fix.",
    "Prepare a 5-minute presentation on a technical topic of your choice.",
]
