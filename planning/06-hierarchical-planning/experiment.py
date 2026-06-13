"""
Hierarchical Planning
----------------------
Decomposes a goal at multiple levels of abstraction:
  Level 0: Goal
  Level 1: Sub-goals (high-level phases)
  Level 2: Tasks (concrete activities within each phase)
  Level 3: Actions (specific steps within each task)  ← optional depth

Key concepts:
- Abstract-to-concrete refinement: plan at high level first, refine each node
- Focus: level-2 planning can focus on one sub-goal at a time
- Reuse: common sub-goals (testing, deployment) appear in many plans
- Scope control: stop refinement when a task is atomic enough to execute
- HTN (Hierarchical Task Network): the classical AI planning approach this mirrors

Compared to task graph (exp 05):
  Task graph: flat nodes with dependency edges
  Hierarchical: tree structure where nodes contain sub-plans

Example — product launch:
  Goal: Launch new product
  ├── Sub-goal 1: Product Development
  │   ├── Task 1.1: Define requirements
  │   ├── Task 1.2: Build MVP
  │   └── Task 1.3: QA testing
  ├── Sub-goal 2: Marketing
  │   ├── Task 2.1: Create landing page
  │   └── Task 2.2: Email campaign
  └── Sub-goal 3: Release
      ├── Task 3.1: Staged rollout
      └── Task 3.2: Monitor metrics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import re


@dataclass
class PlanNode:
    id: str
    description: str
    level: int                                    # 0=goal, 1=subgoal, 2=task, 3=action
    children: list[PlanNode] = field(default_factory=list)
    parent_id: Optional[str] = None
    is_atomic: bool = False                       # True = no further decomposition needed
    estimated_effort: str = ""                    # e.g. "2 hours", "1 week"

    def add_child(self, child: PlanNode) -> None:
        child.parent_id = self.id
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def all_leaves(self) -> list[PlanNode]:
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.all_leaves())
        return leaves

    def display(self, indent: int = 0) -> None:
        level_labels = {0: "GOAL", 1: "SUB-GOAL", 2: "TASK", 3: "ACTION"}
        label = level_labels.get(self.level, f"L{self.level}")
        effort = f" [{self.estimated_effort}]" if self.estimated_effort else ""
        atomic = " (atomic)" if self.is_atomic else ""
        prefix = "  " * indent
        print(f"{prefix}{'└─ ' if indent > 0 else ''}{label}: {self.description}{effort}{atomic}")
        for child in self.children:
            child.display(indent + 1)


@dataclass
class HierarchicalPlan:
    root: PlanNode

    def display(self) -> None:
        print()
        self.root.display()

    def leaf_tasks(self) -> list[PlanNode]:
        return self.root.all_leaves()

    def total_nodes(self) -> int:
        def count(node: PlanNode) -> int:
            return 1 + sum(count(c) for c in node.children)
        return count(self.root)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUBGOAL_SYSTEM = """You are a strategic planning assistant. Given a high-level goal, decompose it into 3-5 major sub-goals (phases or workstreams). Each sub-goal should represent a distinct area of work.

Return JSON:
{
  "sub_goals": [
    {"id": "sg1", "description": "Sub-goal description", "estimated_effort": "2 weeks"}
  ]
}"""

TASK_SYSTEM = """You are a planning assistant. Given a sub-goal, decompose it into 3-6 concrete tasks. Each task should be specific and actionable.

Return JSON:
{
  "tasks": [
    {"id": "t1", "description": "Task description", "estimated_effort": "2 days", "is_atomic": true}
  ]
}"""


def parse_subgoals(goal_node: PlanNode, json_text: str) -> list[PlanNode]:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    nodes = []
    for i, sg in enumerate(data.get("sub_goals", []), 1):
        node = PlanNode(
            id=sg.get("id", f"sg{i}"),
            description=sg["description"],
            level=1,
            estimated_effort=sg.get("estimated_effort", ""),
        )
        nodes.append(node)
    return nodes


def parse_tasks(subgoal_node: PlanNode, json_text: str) -> list[PlanNode]:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    nodes = []
    for i, t in enumerate(data.get("tasks", []), 1):
        node = PlanNode(
            id=t.get("id", f"t{i}"),
            description=t["description"],
            level=2,
            estimated_effort=t.get("estimated_effort", ""),
            is_atomic=t.get("is_atomic", True),
        )
        nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# Mock hierarchical plan
# ---------------------------------------------------------------------------

def mock_product_launch_plan() -> HierarchicalPlan:
    root = PlanNode("goal", "Launch new SaaS product", level=0)

    dev = PlanNode("sg1", "Product Development", level=1, estimated_effort="6 weeks")
    dev.add_child(PlanNode("t1_1", "Define product requirements and user stories", level=2, estimated_effort="3 days", is_atomic=True))
    dev.add_child(PlanNode("t1_2", "Build and test core features (MVP)", level=2, estimated_effort="4 weeks", is_atomic=False))
    dev.add_child(PlanNode("t1_3", "Security audit and performance testing", level=2, estimated_effort="1 week", is_atomic=True))

    mkt = PlanNode("sg2", "Marketing & Go-to-Market", level=1, estimated_effort="4 weeks")
    mkt.add_child(PlanNode("t2_1", "Create landing page and sign-up flow", level=2, estimated_effort="1 week", is_atomic=True))
    mkt.add_child(PlanNode("t2_2", "Write launch blog post and press release", level=2, estimated_effort="3 days", is_atomic=True))
    mkt.add_child(PlanNode("t2_3", "Set up email campaign for waitlist", level=2, estimated_effort="2 days", is_atomic=True))

    rel = PlanNode("sg3", "Release & Operations", level=1, estimated_effort="1 week")
    rel.add_child(PlanNode("t3_1", "Staged rollout to beta users", level=2, estimated_effort="2 days", is_atomic=True))
    rel.add_child(PlanNode("t3_2", "Monitor metrics and on-call rotation", level=2, estimated_effort="ongoing", is_atomic=True))
    rel.add_child(PlanNode("t3_3", "Gather and triage user feedback", level=2, estimated_effort="ongoing", is_atomic=True))

    root.add_child(dev)
    root.add_child(mkt)
    root.add_child(rel)

    return HierarchicalPlan(root=root)


EXAMPLE_GOALS = [
    "Launch a new SaaS product",
    "Migrate a company's infrastructure to the cloud",
    "Build and open-source a Python library",
    "Set up a data science team from scratch",
]
