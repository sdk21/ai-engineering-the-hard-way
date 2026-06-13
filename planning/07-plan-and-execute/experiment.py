"""
Plan-and-Execute
-----------------
A two-phase agent pattern:
  1. PLAN  — produce a complete, ordered list of steps before doing anything
  2. EXECUTE — work through each step, optionally replanning if a step fails

Key differences from simple task decomposition (exp 04):
  - Explicit plan → execute handoff (the model sees its full plan before acting)
  - Each execution step gets tool access (search, compute, read)
  - The executor can update step results and trigger a replan on failure

Why plan first?
  - Holistic view: seeing all steps together lets the model spot missing
    dependencies, duplicate work, or ordering errors
  - Separation of concerns: planning and acting use different prompts /
    temperature settings
  - Inspectability: users can review / edit the plan before any action is taken

Compared to ReAct (exp 03):
  ReAct interleaves planning and acting step-by-step (no global plan).
  Plan-and-Execute produces the full plan upfront, then acts.
  ReAct is more adaptive; Plan-and-Execute is more predictable.
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


@dataclass
class ExecutionStep:
    index: int
    description: str
    tool: str = ""                  # which tool to use ("search", "calculate", "none")
    tool_input: str = ""            # input to pass to the tool
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    error: str = ""

    def display(self, verbose: bool = False) -> None:
        icons = {
            StepStatus.PENDING: "○",
            StepStatus.RUNNING: "◐",
            StepStatus.DONE: "✓",
            StepStatus.FAILED: "✗",
        }
        icon = icons[self.status]
        tool_str = f" [{self.tool}]" if self.tool and self.tool != "none" else ""
        print(f"  {icon} Step {self.index}: {self.description}{tool_str}")
        if verbose and self.result:
            print(f"       → {self.result[:80]}")
        if verbose and self.error:
            print(f"       ✗ {self.error[:80]}")


@dataclass
class ExecutionPlan:
    goal: str
    steps: list[ExecutionStep] = field(default_factory=list)
    final_answer: str = ""

    def display(self, verbose: bool = False) -> None:
        print(f"\n  Goal: {self.goal}")
        print(f"  Plan ({len(self.steps)} steps):")
        for step in self.steps:
            step.display(verbose=verbose)
        if self.final_answer:
            print(f"\n  Answer: {self.final_answer}")

    def pending_steps(self) -> list[ExecutionStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.FAILED) for s in self.steps)

    def has_failures(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are a planning assistant. Given a goal, produce a detailed execution plan as a JSON list of steps.

Each step should specify:
- what to do
- which tool to use (if any): "search", "calculate", or "none"
- what input to pass to the tool

Return JSON:
{
  "steps": [
    {
      "description": "What this step does",
      "tool": "search",
      "tool_input": "query or expression to pass to the tool"
    }
  ]
}

Available tools:
- search: look up facts, find information
- calculate: evaluate a math expression (e.g. "45 * 12 / 100")
- none: reasoning/synthesis step, no external tool needed

Keep plans focused: 3-6 steps is ideal."""

EXECUTOR_SYSTEM = """You are an execution assistant. You are given:
1. The original goal
2. The full plan (all steps)
3. The current step to execute
4. Results from previous steps

Your job: execute the current step and return the result.
Be concise. Return only the result of this step, not a narrative."""

SYNTHESIZER_SYSTEM = """You are a synthesis assistant. Given a goal, a plan, and the results of each step, write a clear, direct final answer to the goal.

Be concise and factual. Draw from the step results."""


def planner_prompt(goal: str) -> str:
    return f"Goal: {goal}"


def executor_prompt(plan: ExecutionPlan, step: ExecutionStep) -> str:
    lines = [f"Goal: {plan.goal}", "", "Full plan:"]
    for s in plan.steps:
        marker = "→ CURRENT" if s.index == step.index else "  "
        done = f" [DONE: {s.result[:60]}]" if s.status == StepStatus.DONE else ""
        lines.append(f"  {marker} Step {s.index}: {s.description}{done}")
    lines += ["", f"Execute step {step.index}: {step.description}"]
    if step.tool and step.tool != "none":
        lines.append(f"Tool: {step.tool}({step.tool_input})")
    return "\n".join(lines)


def synthesizer_prompt(plan: ExecutionPlan) -> str:
    lines = [f"Goal: {plan.goal}", "", "Steps and results:"]
    for s in plan.steps:
        lines.append(f"  Step {s.index}: {s.description}")
        if s.result:
            lines.append(f"    Result: {s.result}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_plan(goal: str, json_text: str) -> ExecutionPlan:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    plan = ExecutionPlan(goal=goal)
    for i, s in enumerate(data.get("steps", []), 1):
        plan.steps.append(ExecutionStep(
            index=i,
            description=s["description"],
            tool=s.get("tool", "none"),
            tool_input=s.get("tool_input", ""),
        ))
    return plan


# ---------------------------------------------------------------------------
# Mock tools (same as ReAct experiment)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "eiffel tower height": "The Eiffel Tower is 330 meters tall (including antenna).",
    "eiffel tower": "The Eiffel Tower is 330 meters tall (including antenna). Built 1887-1889.",
    "burj khalifa height": "The Burj Khalifa is 828 meters tall.",
    "burj khalifa": "The Burj Khalifa in Dubai is 828 meters tall, completed in 2010.",
    "population of france": "France has a population of approximately 68 million people (2023).",
    "france population": "France has approximately 68 million people.",
    "population of paris": "Paris has a metropolitan population of approximately 12 million people.",
    "python created": "Python was created by Guido van Rossum and first released in 1991.",
    "speed of light": "The speed of light is approximately 299,792,458 meters per second.",
}


def mock_search(query: str) -> str:
    q = query.lower().strip()
    for key, val in KNOWLEDGE_BASE.items():
        if key in q or q in key:
            return val
    return f"No information found for: {query}"


def mock_calculate(expression: str) -> str:
    try:
        # Only allow safe math expressions
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return f"Invalid expression: {expression}"
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def dispatch_tool(tool: str, tool_input: str) -> str:
    if tool == "search":
        return mock_search(tool_input)
    elif tool == "calculate":
        return mock_calculate(tool_input)
    return "(no tool output)"


# ---------------------------------------------------------------------------
# Mock plan-and-execute
# ---------------------------------------------------------------------------

MOCK_GOAL = "How many times taller is the Burj Khalifa than the Eiffel Tower?"

MOCK_STEPS = [
    ExecutionStep(1, "Find the height of the Burj Khalifa", "search", "Burj Khalifa height"),
    ExecutionStep(2, "Find the height of the Eiffel Tower", "search", "Eiffel Tower height"),
    ExecutionStep(3, "Calculate the ratio of heights", "calculate", "828 / 330"),
    ExecutionStep(4, "Summarize the answer", "none", ""),
]

MOCK_RESULTS = [
    "The Burj Khalifa is 828 meters tall.",
    "The Eiffel Tower is 330 meters tall (including antenna).",
    "2.509090909090909",
    "The Burj Khalifa (828m) is approximately 2.5× taller than the Eiffel Tower (330m).",
]


def mock_plan_and_execute() -> ExecutionPlan:
    plan = ExecutionPlan(goal=MOCK_GOAL, steps=list(MOCK_STEPS))
    for step, result in zip(plan.steps, MOCK_RESULTS):
        step.result = result
        step.status = StepStatus.DONE
    plan.final_answer = MOCK_RESULTS[-1]
    return plan


EXAMPLE_GOALS = [
    "How many times taller is the Burj Khalifa than the Eiffel Tower?",
    "What year was Python created and how old is it now?",
    "What is the population density of Paris (people per sq km, area = 105 sq km)?",
    "If the speed of light is ~3×10^8 m/s, how many km does light travel in 1 minute?",
]
