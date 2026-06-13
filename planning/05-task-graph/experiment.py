"""
Task Graph (DAG Planning)
--------------------------
A task graph represents a plan as a directed acyclic graph (DAG) where:
  - Nodes are tasks
  - Edges represent dependencies (A → B means B requires A to be done first)
  - Independent tasks (no dependency between them) can run in parallel

Key concepts:
- DAG construction: model generates tasks + dependencies
- Topological ordering: which tasks can run now vs. must wait
- Critical path: the longest dependency chain (determines minimum total time)
- Parallel execution: run independent tasks concurrently
- Visualisation: render the graph as an adjacency structure

Flat decomposition (exp 04): [1, 2, 3, 4, 5] — strictly sequential
Task graph (exp 05):         nodes + edges — parallel where possible

Example — build and deploy a web application:
  ┌── design_db_schema
  ├── design_api ──────→ implement_api ──┐
  │                                      ├── integration_test → deploy
  └── implement_frontend ───────────────┘

  design_db_schema, design_api, implement_frontend can all start in parallel.
  implement_api must wait for design_api (and ideally design_db_schema).
  integration_test must wait for both implement_api and implement_frontend.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator
import json


class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"      # dependencies satisfied, can run now
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    estimated_minutes: int = 0


@dataclass
class TaskGraph:
    goal: str
    tasks: dict[str, Task] = field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        self.tasks[task.id] = task

    def ready_tasks(self) -> list[Task]:
        """Tasks whose dependencies are all DONE and which are PENDING."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if all(self.tasks[dep].status == TaskStatus.DONE for dep in task.depends_on if dep in self.tasks):
                ready.append(task)
        return ready

    def is_complete(self) -> bool:
        return all(t.status == TaskStatus.DONE for t in self.tasks.values())

    def critical_path(self) -> list[str]:
        """Return task IDs on the critical path (longest dependency chain)."""
        # Longest path in DAG via dynamic programming
        memo: dict[str, int] = {}

        def longest(task_id: str) -> int:
            if task_id in memo:
                return memo[task_id]
            task = self.tasks.get(task_id)
            if not task or not task.depends_on:
                memo[task_id] = task.estimated_minutes if task else 0
                return memo[task_id]
            best = max(longest(d) for d in task.depends_on if d in self.tasks)
            memo[task_id] = best + (task.estimated_minutes or 1)
            return memo[task_id]

        if not self.tasks:
            return []

        end_task_id = max(self.tasks, key=lambda t: longest(t))
        # Trace back
        path = [end_task_id]
        current = end_task_id
        while True:
            task = self.tasks[current]
            if not task.depends_on:
                break
            current = max(task.depends_on, key=lambda d: memo.get(d, 0))
            path.append(current)
        return list(reversed(path))

    def topological_layers(self) -> list[list[str]]:
        """Group tasks into layers where all tasks in a layer can run in parallel."""
        in_degree = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.depends_on:
                if dep in in_degree:
                    in_degree[task.id] = in_degree.get(task.id, 0) + 1

        layers = []
        remaining = dict(in_degree)
        while remaining:
            layer = [tid for tid, deg in remaining.items() if deg == 0]
            if not layer:
                break  # cycle detected
            layers.append(layer)
            for tid in layer:
                del remaining[tid]
                for other_id, task in self.tasks.items():
                    if other_id in remaining and tid in task.depends_on:
                        remaining[other_id] -= 1
        return layers

    def display(self) -> None:
        icons = {TaskStatus.PENDING: "○", TaskStatus.READY: "◎",
                 TaskStatus.RUNNING: "◐", TaskStatus.DONE: "✓", TaskStatus.FAILED: "✗"}
        print(f"\n  Goal: {self.goal}")
        layers = self.topological_layers()
        for i, layer in enumerate(layers):
            parallel_note = f" [can run in parallel]" if len(layer) > 1 else ""
            print(f"\n  Layer {i+1}{parallel_note}:")
            for tid in layer:
                t = self.tasks[tid]
                icon = icons[t.status]
                dep_str = f" (after: {', '.join(t.depends_on)})" if t.depends_on else ""
                result_str = f" → {t.result[:40]}" if t.result else ""
                print(f"    {icon} [{tid}] {t.description}{dep_str}{result_str}")


# ---------------------------------------------------------------------------
# Plan generation prompt
# ---------------------------------------------------------------------------

GRAPH_SYSTEM = """You are a planning assistant. Given a goal, produce a task graph as JSON.

Return ONLY valid JSON in this format:
{
  "tasks": [
    {
      "id": "short_snake_case_id",
      "description": "What this task does",
      "depends_on": ["id_of_prerequisite_task"],
      "estimated_minutes": 30
    }
  ]
}

Rules:
- Use short snake_case IDs (e.g. "design_api", "write_tests")
- depends_on is a list of task IDs that must complete before this task
- Tasks with no dependencies can start immediately (run in parallel)
- Only add dependencies that are genuinely required
- 4-10 tasks is ideal
- estimated_minutes should be realistic
"""


def parse_task_graph(goal: str, json_text: str) -> TaskGraph:
    """Parse model JSON output into a TaskGraph."""
    import re
    # Extract JSON from response (model may wrap it in markdown)
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    graph = TaskGraph(goal=goal)
    for t in data.get("tasks", []):
        graph.add_task(Task(
            id=t["id"],
            description=t["description"],
            depends_on=t.get("depends_on", []),
            estimated_minutes=t.get("estimated_minutes", 0),
        ))
    return graph


# ---------------------------------------------------------------------------
# Mock graph for demo
# ---------------------------------------------------------------------------

def mock_webapp_graph() -> TaskGraph:
    goal = "Build and deploy a new web application feature"
    g = TaskGraph(goal=goal)
    tasks = [
        Task("design_db", "Design database schema changes", [], estimated_minutes=60),
        Task("design_api", "Design REST API endpoints", [], estimated_minutes=45),
        Task("design_ui", "Design UI wireframes", [], estimated_minutes=90),
        Task("impl_db", "Implement database migrations", ["design_db"], estimated_minutes=120),
        Task("impl_api", "Implement API endpoints", ["design_api", "design_db"], estimated_minutes=180),
        Task("impl_ui", "Implement frontend components", ["design_ui"], estimated_minutes=240),
        Task("write_tests", "Write unit and integration tests", ["impl_api", "impl_db"], estimated_minutes=90),
        Task("integration_test", "Run full integration test suite", ["impl_api", "impl_ui", "write_tests"], estimated_minutes=30),
        Task("deploy", "Deploy to production", ["integration_test"], estimated_minutes=15),
    ]
    for t in tasks:
        g.add_task(t)
    return g


EXAMPLE_GOALS = [
    "Build and deploy a new web application feature",
    "Plan and execute a product launch",
    "Migrate a monolith to microservices",
    "Set up a CI/CD pipeline from scratch",
]
