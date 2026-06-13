"""
Tree of Thoughts (ToT)
-----------------------
Explores multiple reasoning paths in a tree structure, evaluating each branch
before deciding which to pursue. Mimics deliberate problem-solving.

Search strategies implemented:
  BFS (Breadth-First Search):
    Expand all nodes at depth d before going deeper.
    Good when: the best answer is likely shallow; you want to compare many options.

  DFS (Depth-First Search):
    Follow one path as deep as possible, backtrack on failure.
    Good when: you want any valid answer quickly; exploring deeply is cheap.

Node lifecycle:
  OPEN → being explored
  PROMISING → scored high, worth expanding
  DEAD_END → scored low or failed evaluation, prune this branch
  SOLUTION → satisfies the goal condition

Scoring:
  Each thought is evaluated by the model on a 1-10 scale.
  Nodes scoring < threshold are pruned (DEAD_END).
  Nodes scoring >= threshold are expanded.

Key insight:
  Standard CoT is a single path through the thought tree.
  ToT explicitly branches at decision points and uses evaluation to guide search.
  This is especially valuable for tasks where the first plausible-sounding path
  may be wrong (puzzles, proofs, creative writing with constraints).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re


class NodeStatus(Enum):
    OPEN = "open"
    PROMISING = "promising"
    DEAD_END = "dead_end"
    SOLUTION = "solution"


@dataclass
class ThoughtNode:
    id: str
    thought: str
    depth: int
    score: float = 0.0                  # 1-10 from evaluator
    status: NodeStatus = NodeStatus.OPEN
    children: list[ThoughtNode] = field(default_factory=list)
    parent_id: Optional[str] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def display(self, indent: int = 0) -> None:
        status_icons = {
            NodeStatus.OPEN: "○",
            NodeStatus.PROMISING: "◉",
            NodeStatus.DEAD_END: "✗",
            NodeStatus.SOLUTION: "★",
        }
        icon = status_icons[self.status]
        score_str = f" [{self.score:.1f}/10]" if self.score > 0 else ""
        prefix = "  " * indent
        connector = "└─ " if indent > 0 else ""
        thought_short = self.thought[:70] + ("..." if len(self.thought) > 70 else "")
        print(f"{prefix}{connector}{icon} [{self.id}]{score_str} {thought_short}")
        for child in self.children:
            child.display(indent + 1)


@dataclass
class ThoughtTree:
    problem: str
    root: Optional[ThoughtNode] = None
    all_nodes: dict[str, ThoughtNode] = field(default_factory=dict)
    solution: Optional[ThoughtNode] = None

    def add_node(self, node: ThoughtNode) -> None:
        self.all_nodes[node.id] = node

    def display(self) -> None:
        print(f"\n  Problem: {self.problem}")
        if self.root:
            self.root.display()
        if self.solution:
            print(f"\n  Solution: {self.solution.thought}")

    def total_nodes(self) -> int:
        return len(self.all_nodes)

    def pruned_nodes(self) -> int:
        return sum(1 for n in self.all_nodes.values() if n.status == NodeStatus.DEAD_END)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

THOUGHT_GEN_SYSTEM = """You are a systematic problem solver. Generate {k} distinct next steps or partial solutions for solving the given problem.

Each thought should:
- Make meaningful progress toward the solution
- Be different from the others (explore diverse approaches)
- Be 1-3 sentences

Return JSON:
{{
  "thoughts": [
    "First distinct approach or step...",
    "Second distinct approach or step..."
  ]
}}"""

EVALUATOR_SYSTEM = """You are a critical evaluator. Rate the quality of this partial solution on a scale of 1-10.

Consider:
- Is the reasoning sound?
- Does it make genuine progress toward the answer?
- Is it free from logical errors?
- Could this lead to a correct final answer?

Return JSON: {{"score": 7, "reason": "brief explanation"}}

Be strict: 1-3 = likely wrong path, 4-6 = uncertain, 7-10 = promising."""

SOLUTION_CHECK_SYSTEM = """You are a solution verifier. Determine if the given thought/reasoning fully solves the problem.

Return JSON: {{"is_solution": true/false, "answer": "the final answer if solved, else empty string"}}"""


def thought_gen_prompt(problem: str, path_so_far: list[str], k: int = 3) -> tuple[str, str]:
    system = THOUGHT_GEN_SYSTEM.format(k=k)
    path_str = ""
    if path_so_far:
        path_str = "\n\nReasoning so far:\n" + "\n".join(f"- {t}" for t in path_so_far)
    user = f"Problem: {problem}{path_str}\n\nGenerate {k} next thoughts:"
    return system, user


def evaluator_prompt(problem: str, path_so_far: list[str], thought: str) -> str:
    path_str = ""
    if path_so_far:
        path_str = "\nPrior reasoning:\n" + "\n".join(f"- {t}" for t in path_so_far)
    return f"Problem: {problem}{path_str}\n\nThought to evaluate: {thought}"


def solution_check_prompt(problem: str, path_so_far: list[str]) -> str:
    path_str = "\n".join(f"- {t}" for t in path_so_far)
    return f"Problem: {problem}\n\nFull reasoning path:\n{path_str}\n\nDoes this fully solve the problem?"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_thoughts(json_text: str) -> list[str]:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    import json
    data = json.loads(json_text)
    return data.get("thoughts", [])


def parse_score(json_text: str) -> tuple[float, str]:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    import json
    data = json.loads(json_text)
    return float(data.get("score", 5)), data.get("reason", "")


def parse_solution_check(json_text: str) -> tuple[bool, str]:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    import json
    data = json.loads(json_text)
    return bool(data.get("is_solution", False)), data.get("answer", "")


# ---------------------------------------------------------------------------
# Mock Tree of Thoughts
# ---------------------------------------------------------------------------

def mock_tot_tree() -> ThoughtTree:
    problem = "A farmer has 17 sheep. All but 9 die. How many sheep are left?"
    tree = ThoughtTree(problem=problem)

    root = ThoughtNode("root", "Starting problem: 17 sheep, all but 9 die.", depth=0,
                       score=0, status=NodeStatus.OPEN)

    # Branch A: wrong interpretation
    a = ThoughtNode("A", "All but 9 die means all sheep die except 9. 17 - 9 = 8 sheep died, so 9 remain.", depth=1,
                    score=8.5, status=NodeStatus.SOLUTION, parent_id="root")
    # Branch B: wrong interpretation
    b = ThoughtNode("B", "17 minus all = 0, then add 9 back = 9. But this logic is unclear.", depth=1,
                    score=4.0, status=NodeStatus.PROMISING, parent_id="root")
    b1 = ThoughtNode("B1", "This approach is confused. '0 + 9' doesn't follow from the problem statement.", depth=2,
                     score=2.0, status=NodeStatus.DEAD_END, parent_id="B")
    # Branch C: wrong interpretation
    c = ThoughtNode("C", "All 17 die, but 9 is the answer because the problem says 'all but 9'.", depth=1,
                    score=3.0, status=NodeStatus.DEAD_END, parent_id="root")

    b.children.append(b1)
    root.children.extend([a, b, c])
    tree.root = root
    tree.solution = a
    for n in [root, a, b, b1, c]:
        tree.add_node(n)
    return tree


EXAMPLE_PROBLEMS = [
    "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
    "You have 3 boxes: one with apples, one with oranges, one with both. All labels are wrong. You can take one fruit from one box. How do you label them correctly?",
    "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons?",
]
