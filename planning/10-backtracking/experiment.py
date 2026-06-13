"""
Backtracking
-------------
A constraint-satisfaction approach: attempt a solution step-by-step, detect
violations, and undo (backtrack) to the last decision point to try a different
choice.

Classical backtracking (algorithm):
  - Assign variables one at a time
  - After each assignment, check constraints
  - If any constraint is violated → undo last assignment, try next value
  - If all values exhausted at a node → backtrack further

LLM-augmented backtracking (this experiment):
  - The model proposes the next step
  - A constraint checker (model or rules) validates the step
  - On failure, the step is removed and the model is told to try differently
  - History of failed attempts is passed back so the model doesn't repeat them

Example — scheduling:
  Slot slots 1-5, tasks A-E, constraints:
    A before B, C not at slot 3, D after C, ...
  The model assigns tasks, we validate constraints, backtrack on violation.

Example — N-Queens:
  Place N queens on an N×N board, no two attacking each other.
  Classic backtracking problem with clear constraint checking.

Key insight:
  Unlike self-reflection (exp 08) which improves quality iteratively,
  backtracking handles *hard constraints* — things that are simply wrong
  and must be undone, not just improved.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re
import json


@dataclass
class BacktrackState:
    """Represents the current partial solution and history."""
    assignments: list[tuple[str, str]] = field(default_factory=list)   # (variable, value)
    failed_attempts: list[tuple[str, str, str]] = field(default_factory=list)  # (variable, value, reason)
    backtracks: int = 0

    def assign(self, variable: str, value: str) -> None:
        self.assignments.append((variable, value))

    def undo_last(self, reason: str) -> Optional[tuple[str, str]]:
        if not self.assignments:
            return None
        var, val = self.assignments.pop()
        self.failed_attempts.append((var, val, reason))
        self.backtracks += 1
        return var, val

    def display(self) -> None:
        print(f"  Assignments ({len(self.assignments)}):")
        for var, val in self.assignments:
            print(f"    {var} = {val}")
        if self.failed_attempts:
            print(f"  Failed attempts ({len(self.failed_attempts)}):")
            for var, val, reason in self.failed_attempts[-3:]:  # show last 3
                print(f"    {var} ≠ {val} ({reason})")
        print(f"  Backtracks: {self.backtracks}")


@dataclass
class BacktrackSession:
    problem: str
    variables: list[str]             # ordered list of variables to assign
    constraints: list[str]           # human-readable constraint descriptions
    state: BacktrackState = field(default_factory=BacktrackState)
    solved: bool = False
    solution: dict[str, str] = field(default_factory=dict)

    def display(self) -> None:
        print(f"\n  Problem: {self.problem}")
        print(f"  Variables: {', '.join(self.variables)}")
        print(f"  Constraints: {len(self.constraints)}")
        for c in self.constraints:
            print(f"    - {c}")
        self.state.display()
        if self.solved:
            print(f"\n  Solution: {self.solution}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROPOSER_SYSTEM = """You are a constraint satisfaction solver. You will be given:
1. A problem with variables and constraints
2. The current partial assignment
3. Failed attempts for the next variable (do NOT repeat these)

Your task: propose a value for the next unassigned variable.

Return JSON: {{"variable": "name", "value": "proposed value", "reasoning": "why this satisfies constraints"}}"""

CHECKER_SYSTEM = """You are a constraint validator. Check if the proposed assignment violates any constraints.

Return JSON: {{"valid": true/false, "violation": "which constraint is violated and why (empty if valid)"}}"""


def proposer_prompt(session: BacktrackSession) -> str:
    assigned = {var: val for var, val in session.state.assignments}
    remaining = [v for v in session.variables if v not in assigned]
    next_var = remaining[0] if remaining else None

    failed_for_next = [(val, reason) for var, val, reason in session.state.failed_attempts
                       if var == next_var]

    lines = [
        f"Problem: {session.problem}",
        "",
        "Constraints:",
        *[f"  - {c}" for c in session.constraints],
        "",
        f"Current assignments: {json.dumps(assigned)}",
        f"Next variable to assign: {next_var}",
    ]
    if failed_for_next:
        lines.append("Failed values for this variable (do NOT use):")
        for val, reason in failed_for_next:
            lines.append(f"  - {val!r}: {reason}")
    return "\n".join(lines)


def checker_prompt(session: BacktrackSession, variable: str, value: str) -> str:
    assigned = {var: val for var, val in session.state.assignments}
    assigned[variable] = value
    return (
        f"Problem: {session.problem}\n\n"
        f"Constraints:\n" + "\n".join(f"  - {c}" for c in session.constraints) +
        f"\n\nPartial assignment: {json.dumps(assigned)}\n\n"
        f"Check if assigning {variable}={value!r} violates any constraint."
    )


# ---------------------------------------------------------------------------
# Mock: scheduling problem
# ---------------------------------------------------------------------------

MOCK_PROBLEM = "Schedule tasks A, B, C, D into slots 1-4 (one task per slot)."
MOCK_VARIABLES = ["slot_1", "slot_2", "slot_3", "slot_4"]
MOCK_CONSTRAINTS = [
    "Task A must come before task B (A's slot < B's slot)",
    "Task C must not be in slot 1",
    "Task D must be in the last slot (slot 4)",
]

# Mock execution: show a backtrack
def mock_backtrack_session() -> BacktrackSession:
    session = BacktrackSession(
        problem=MOCK_PROBLEM,
        variables=MOCK_VARIABLES,
        constraints=MOCK_CONSTRAINTS,
    )
    # First attempt: violates "C not in slot 1"
    session.state.assign("slot_1", "C")
    session.state.undo_last("Violates: Task C must not be in slot 1")

    # Second attempt: try A in slot 1
    session.state.assign("slot_1", "A")
    session.state.assign("slot_2", "B")
    session.state.assign("slot_3", "C")
    session.state.assign("slot_4", "D")

    session.solved = True
    session.solution = {"slot_1": "A", "slot_2": "B", "slot_3": "C", "slot_4": "D"}
    return session


EXAMPLE_PROBLEMS = [
    {
        "problem": "Schedule tasks A, B, C, D into slots 1-4 (one task per slot).",
        "variables": ["slot_1", "slot_2", "slot_3", "slot_4"],
        "domain": ["A", "B", "C", "D"],
        "constraints": [
            "Task A must come before task B (A's slot < B's slot)",
            "Task C must not be in slot 1",
            "Task D must be in slot 4",
        ],
    },
    {
        "problem": "Assign colors Red, Green, Blue to nodes 1, 2, 3, 4 (map coloring).",
        "variables": ["node_1", "node_2", "node_3", "node_4"],
        "domain": ["Red", "Green", "Blue"],
        "constraints": [
            "node_1 and node_2 are adjacent (must differ)",
            "node_2 and node_3 are adjacent (must differ)",
            "node_3 and node_4 are adjacent (must differ)",
            "node_1 and node_3 are adjacent (must differ)",
        ],
    },
]
