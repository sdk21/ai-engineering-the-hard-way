"""
Task Decomposition
------------------
Breaking a complex goal into a flat ordered list of sub-tasks that can
be executed sequentially. The model generates the plan; a runner executes it.

Key concepts:
- Plan generation: produce a numbered list of concrete steps
- Step validation: ensure steps are actionable and non-overlapping
- Dependency awareness: steps that depend on earlier steps reference them
- Plan review: human or model reviews the plan before execution
- Execution trace: track which steps completed, failed, or were skipped

This is the simplest planning pattern:
  Goal → [step 1, step 2, step 3, ...] → execute in order

Limitations (addressed by later experiments):
  - No parallelism (task graph, exp 05)
  - No hierarchy (hierarchical planning, exp 06)
  - No replanning (replanning, exp 12)
"""

from dataclasses import dataclass, field
from enum import Enum


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    index: int
    description: str
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    depends_on: list[int] = field(default_factory=list)  # step indices


@dataclass
class Plan:
    goal: str
    steps: list[Step] = field(default_factory=list)

    def display(self) -> None:
        print(f"\n  Goal: {self.goal}")
        print(f"  Steps ({len(self.steps)}):")
        icons = {StepStatus.PENDING: "○", StepStatus.RUNNING: "◐",
                 StepStatus.DONE: "✓", StepStatus.FAILED: "✗", StepStatus.SKIPPED: "–"}
        for s in self.steps:
            icon = icons[s.status]
            result_str = f" → {s.result[:50]}" if s.result else ""
            print(f"    {icon} {s.index}. {s.description}{result_str}")

    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps)

    def next_pending(self) -> Step | None:
        for s in self.steps:
            if s.status == StepStatus.PENDING:
                # Check all dependencies are done
                deps_done = all(
                    self.steps[d - 1].status == StepStatus.DONE
                    for d in s.depends_on
                )
                if deps_done:
                    return s
        return None


# ---------------------------------------------------------------------------
# Plan generation prompt
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """You are a planning assistant. Given a goal, produce a numbered list of concrete,
actionable steps to achieve it. Each step should be specific enough that an executor can carry it out.

Format your response as:
1. [step description]
2. [step description]
...

Rules:
- Steps must be in execution order
- Each step should be a single concrete action
- Do not include steps that are vague or unmeasurable
- 3-8 steps is ideal; use more only if genuinely needed
- If a step depends on the result of a previous step, say so explicitly
"""

def decompose_prompt(goal: str, context: str = "") -> str:
    base = f"Goal: {goal}"
    if context:
        base += f"\n\nContext: {context}"
    base += "\n\nProvide a numbered list of steps:"
    return base


def parse_plan(goal: str, text: str) -> Plan:
    """Parse numbered list into a Plan."""
    import re
    plan = Plan(goal=goal)
    for match in re.finditer(r"^\s*(\d+)[.)]\s*(.+?)$", text, re.MULTILINE):
        idx = int(match.group(1))
        desc = match.group(2).strip()
        plan.steps.append(Step(index=idx, description=desc))
    return plan


# ---------------------------------------------------------------------------
# Execution simulation
# ---------------------------------------------------------------------------

def simulate_execute_step(step: Step, plan: Plan) -> str:
    """Simulate executing a step. Returns a mock result."""
    desc_lower = step.description.lower()

    if "research" in desc_lower or "gather" in desc_lower or "find" in desc_lower:
        return "Gathered relevant information and sources."
    elif "write" in desc_lower or "draft" in desc_lower:
        return "Draft written (2 pages)."
    elif "review" in desc_lower or "check" in desc_lower:
        return "Reviewed and approved with minor edits."
    elif "send" in desc_lower or "publish" in desc_lower or "submit" in desc_lower:
        return "Sent/published successfully."
    elif "test" in desc_lower or "verify" in desc_lower:
        return "Tests passed."
    elif "install" in desc_lower or "set up" in desc_lower or "configure" in desc_lower:
        return "Setup complete."
    else:
        return f"Step completed."


# ---------------------------------------------------------------------------
# Example goals
# ---------------------------------------------------------------------------

EXAMPLE_GOALS = [
    "Write and publish a blog post about the benefits of daily exercise.",
    "Set up a Python development environment on a new machine.",
    "Plan and execute a team offsite for 20 people.",
    "Debug and fix a memory leak in a production web service.",
    "Launch a new feature on a SaaS product.",
]
