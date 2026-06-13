"""
Capstone: Autonomous Task Solver
----------------------------------
Combines the best planning techniques from this vertical into a single agent:

  1. HIERARCHICAL PLANNING (exp 06)
     Decompose the goal into sub-goals, then into concrete tasks.

  2. TASK GRAPH (exp 05)
     Identify dependencies between tasks; run independent tasks in parallel.

  3. PLAN-AND-EXECUTE (exp 07)
     Full plan upfront before any execution; tools available during execution.

  4. SELF-REFLECTION (exp 08)
     After producing a draft answer, critique and revise it once.

  5. REPLANNING (exp 12)
     If a task fails, regenerate the affected sub-plan.

Architecture:
  ┌────────────────────────────────────────────┐
  │              Autonomous Task Solver         │
  │                                            │
  │  [Hierarchical Planner]                    │
  │    Goal → Sub-goals → Tasks                │
  │         ↓                                  │
  │  [Dependency Resolver]                     │
  │    Tasks → DAG (topological layers)        │
  │         ↓                                  │
  │  [Parallel Executor]                       │
  │    Execute layer by layer (with tools)     │
  │         ↓ (on failure)                     │
  │  [Replanner]                               │
  │    Failed task → new sub-plan              │
  │         ↓                                  │
  │  [Self-Reflector]                          │
  │    Draft answer → critique → final answer  │
  └────────────────────────────────────────────┘

Tools available to executor:
  - search(query): look up facts
  - calculate(expr): evaluate math expressions
  - summarize(text): condense long text
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
import re
import concurrent.futures


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class SolverTask:
    id: str
    description: str
    subgoal_id: str
    depends_on: list[str] = field(default_factory=list)
    tool: str = "none"
    tool_input: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    failure_reason: str = ""
    plan_version: int = 1

    def display(self) -> None:
        icons = {TaskStatus.PENDING: "○", TaskStatus.RUNNING: "◐",
                 TaskStatus.DONE: "✓", TaskStatus.FAILED: "✗"}
        icon = icons[self.status]
        tool_str = f" [{self.tool}]" if self.tool != "none" else ""
        result_str = f" → {self.result[:60]}" if self.result else ""
        fail_str = f" ✗ {self.failure_reason[:60]}" if self.failure_reason else ""
        print(f"    {icon} [{self.id}]{tool_str} {self.description}{result_str}{fail_str}")


@dataclass
class SubGoal:
    id: str
    description: str
    tasks: list[SolverTask] = field(default_factory=list)

    def display(self) -> None:
        print(f"  Sub-goal [{self.id}]: {self.description}")
        for t in self.tasks:
            t.display()


@dataclass
class SolverSession:
    goal: str
    subgoals: list[SubGoal] = field(default_factory=list)
    all_tasks: dict[str, SolverTask] = field(default_factory=dict)
    draft_answer: str = ""
    critique: str = ""
    final_answer: str = ""
    replans: int = 0

    def display(self) -> None:
        print(f"\n  Goal: {self.goal}")
        for sg in self.subgoals:
            sg.display()
        if self.final_answer:
            print(f"\n  Final Answer:\n  {self.final_answer}")
        if self.replans:
            print(f"\n  [Total replans: {self.replans}]")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUBGOAL_SYSTEM = """Decompose this goal into 2-4 major sub-goals (phases). Return JSON:
{"sub_goals": [{"id": "sg1", "description": "..."}]}"""

TASK_SYSTEM = """Decompose this sub-goal into 2-4 concrete tasks. Each task should specify a tool if needed.
Available tools: search, calculate, none.
Return JSON:
{"tasks": [{"id": "t1", "description": "...", "depends_on": [], "tool": "search", "tool_input": "query"}]}"""

EXECUTOR_SYSTEM = """Execute the given task. Return ONLY the result.
If the task cannot be completed, return: FAILED: <reason>"""

REPLANNER_SYSTEM = """A task failed. Generate a replacement task (or tasks) to achieve the same sub-goal differently.
Return JSON: {"tasks": [{"id": "r1", "description": "...", "depends_on": [], "tool": "search", "tool_input": "..."}]}
Do NOT use the same approach that failed."""

DRAFTER_SYSTEM = """Given the goal and all task results, write a comprehensive answer."""

CRITIC_SYSTEM = """Critique this answer. If it's good enough, reply: NO_ISSUES
Otherwise list specific problems (2-4 bullet points)."""

REVISER_SYSTEM = """Revise the answer to address the critique. Return only the revised answer."""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "python": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. Known for readability and simplicity.",
    "fastapi": "FastAPI is a modern, high-performance Python web framework for building APIs, built on Starlette and Pydantic. Supports async, auto-generates OpenAPI docs.",
    "flask": "Flask is a lightweight Python web framework. Simple, flexible, widely used. Lacks built-in async support.",
    "django": "Django is a high-level Python web framework with 'batteries included' — ORM, admin, auth built in.",
    "machine learning": "Machine learning is a subset of AI where systems learn from data. Key libraries: scikit-learn, TensorFlow, PyTorch.",
    "docker": "Docker is a containerization platform that packages applications with their dependencies for consistent deployment.",
    "kubernetes": "Kubernetes (K8s) is a container orchestration system for automating deployment, scaling, and management of containerized applications.",
    "postgresql": "PostgreSQL is a powerful, open-source relational database system known for extensibility and standards compliance.",
    "redis": "Redis is an in-memory data structure store used as a database, cache, and message broker.",
    "rest api": "REST (Representational State Transfer) is an architectural style for distributed hypermedia systems using HTTP methods.",
}


def tool_search(query: str) -> str:
    q = query.lower()
    for key, val in KNOWLEDGE_BASE.items():
        if key in q or q in key:
            return val
    return f"No specific information found for: {query}. General knowledge applies."


def tool_calculate(expr: str) -> str:
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expr):
            return f"Invalid expression: {expr}"
        result = eval(expr)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def tool_summarize(text: str) -> str:
    if len(text) <= 200:
        return text
    return text[:197] + "..."


def dispatch_tool(tool: str, tool_input: str) -> str:
    if tool == "search":
        return tool_search(tool_input)
    elif tool == "calculate":
        return tool_calculate(tool_input)
    elif tool == "summarize":
        return tool_summarize(tool_input)
    return "(no tool)"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


def topological_layers(tasks: list[SolverTask]) -> list[list[SolverTask]]:
    task_map = {t.id: t for t in tasks}
    in_degree = {t.id: 0 for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            if dep in in_degree:
                in_degree[t.id] = in_degree.get(t.id, 0) + 1

    layers = []
    remaining = dict(in_degree)
    while remaining:
        layer_ids = [tid for tid, deg in remaining.items() if deg == 0]
        if not layer_ids:
            break
        layers.append([task_map[tid] for tid in layer_ids])
        for tid in layer_ids:
            del remaining[tid]
            for other in tasks:
                if other.id in remaining and tid in other.depends_on:
                    remaining[other.id] -= 1
    return layers


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def run_solver(goal: str, client, verbose: bool = False) -> SolverSession:
    session = SolverSession(goal=goal)

    # ── Step 1: Hierarchical planning ──
    if verbose:
        print("  [Step 1: Hierarchical planning]")

    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=SUBGOAL_SYSTEM,
        messages=[{"role": "user", "content": f"Goal: {goal}"}],
    )
    data = _extract_json(r.content[0].text)
    subgoal_counter = [0]
    task_counter = [0]

    for sg_data in data.get("sub_goals", []):
        sg = SubGoal(id=sg_data["id"], description=sg_data["description"])

        r2 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=TASK_SYSTEM,
            messages=[{"role": "user", "content": f"Sub-goal: {sg.description}\n(Part of goal: {goal})"}],
        )
        t_data = _extract_json(r2.content[0].text)
        for td in t_data.get("tasks", []):
            task = SolverTask(
                id=td.get("id", f"t{task_counter[0]}"),
                description=td["description"],
                subgoal_id=sg.id,
                depends_on=td.get("depends_on", []),
                tool=td.get("tool", "none"),
                tool_input=td.get("tool_input", ""),
            )
            task_counter[0] += 1
            sg.tasks.append(task)
            session.all_tasks[task.id] = task
        session.subgoals.append(sg)

    if verbose:
        session.display()

    # ── Step 2: Execute sub-goals in order, tasks in topological layers ──
    if verbose:
        print("\n  [Step 2: Executing tasks]")

    for sg in session.subgoals:
        if verbose:
            print(f"\n  Sub-goal: {sg.description}")

        layers = topological_layers(sg.tasks)
        replan_attempts = 0

        for layer in layers:
            if verbose:
                parallel_note = " [parallel]" if len(layer) > 1 else ""
                print(f"    Layer{parallel_note}: {[t.id for t in layer]}")

            def execute_task(task: SolverTask) -> SolverTask:
                task.status = TaskStatus.RUNNING
                if task.tool != "none" and task.tool_input:
                    result = dispatch_tool(task.tool, task.tool_input)
                    task.result = result
                    task.status = TaskStatus.DONE
                else:
                    # Gather context from completed tasks in this subgoal
                    completed_context = "\n".join(
                        f"  - {t.description}: {t.result[:60]}"
                        for t in sg.tasks if t.status == TaskStatus.DONE and t.id != task.id
                    )
                    prompt = f"Goal: {goal}\nSub-goal: {sg.description}\n"
                    if completed_context:
                        prompt += f"Completed so far:\n{completed_context}\n"
                    prompt += f"\nExecute: {task.description}"
                    r = client.messages.create(
                        model="claude-haiku-4-5-20251001", max_tokens=256,
                        system=EXECUTOR_SYSTEM,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    result_text = r.content[0].text.strip()
                    if result_text.upper().startswith("FAILED:"):
                        task.status = TaskStatus.FAILED
                        task.failure_reason = result_text[7:].strip()
                    else:
                        task.result = result_text
                        task.status = TaskStatus.DONE
                return task

            # Run layer tasks (parallel if multiple)
            if len(layer) > 1:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    list(executor.map(execute_task, layer))
            else:
                execute_task(layer[0])

            # Handle failures with replanning
            failed = [t for t in layer if t.status == TaskStatus.FAILED]
            for failed_task in failed:
                if replan_attempts >= 2:
                    if verbose:
                        print(f"    [Max replans reached for sub-goal {sg.id}]")
                    continue
                if verbose:
                    print(f"    [Replanning for {failed_task.id}: {failed_task.failure_reason}]")

                completed_results = "\n".join(
                    f"  {t.id}: {t.result[:60]}" for t in sg.tasks if t.status == TaskStatus.DONE
                )
                r_replan = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=256,
                    system=REPLANNER_SYSTEM,
                    messages=[{"role": "user", "content":
                        f"Sub-goal: {sg.description}\n"
                        f"Failed task: {failed_task.description}\n"
                        f"Failure: {failed_task.failure_reason}\n"
                        f"Completed tasks:\n{completed_results}"}],
                )
                try:
                    r_data = _extract_json(r_replan.content[0].text)
                    for td in r_data.get("tasks", []):
                        task_counter[0] += 1
                        new_task = SolverTask(
                            id=td.get("id", f"r{task_counter[0]}"),
                            description=td["description"],
                            subgoal_id=sg.id,
                            depends_on=td.get("depends_on", []),
                            tool=td.get("tool", "none"),
                            tool_input=td.get("tool_input", ""),
                            plan_version=2,
                        )
                        sg.tasks.append(new_task)
                        session.all_tasks[new_task.id] = new_task
                        execute_task(new_task)
                    session.replans += 1
                    replan_attempts += 1
                except Exception:
                    pass

    # ── Step 3: Draft answer ──
    if verbose:
        print("\n  [Step 3: Drafting answer]")

    all_results = "\n".join(
        f"  [{sg.description}] {t.id}: {t.result[:80]}"
        for sg in session.subgoals
        for t in sg.tasks
        if t.status == TaskStatus.DONE and t.result
    )
    r_draft = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=DRAFTER_SYSTEM,
        messages=[{"role": "user", "content": f"Goal: {goal}\n\nTask results:\n{all_results}"}],
    )
    session.draft_answer = r_draft.content[0].text.strip()

    # ── Step 4: Self-reflection ──
    if verbose:
        print("  [Step 4: Self-reflection]")

    r_critique = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=256,
        system=CRITIC_SYSTEM,
        messages=[{"role": "user", "content": f"Task: {goal}\n\nDraft answer:\n{session.draft_answer}"}],
    )
    session.critique = r_critique.content[0].text.strip()

    if "NO_ISSUES" in session.critique.upper():
        session.final_answer = session.draft_answer
    else:
        r_revise = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=REVISER_SYSTEM,
            messages=[{"role": "user", "content":
                f"Task: {goal}\n\nDraft:\n{session.draft_answer}\n\nCritique:\n{session.critique}"}],
        )
        session.final_answer = r_revise.content[0].text.strip()

    return session
