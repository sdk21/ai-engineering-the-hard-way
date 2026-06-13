"""
Swarm
------
Many simple agents with local rules produce complex collective behavior.
No central orchestrator — emergent coordination from individual interactions.

Inspired by biological swarms (ants, bees, starlings) where individuals
follow simple rules and the collective solves hard problems.

LLM swarm pattern:
  - N agents, each with the same (or similar) simple prompt
  - Each agent processes one item / one task
  - Agents may read a shared pool and contribute back
  - No single agent has a global view
  - The "intelligence" emerges from the collective

Classic swarm algorithms (for context):
  - Ant Colony Optimization: ants deposit pheromones; paths reinforce
  - Particle Swarm Optimization: particles adjust velocity toward best known position
  - Boid flocking: separation, alignment, cohesion rules produce flocking

LLM swarm example here: document annotation
  - 10 document chunks, 3 worker agents running in parallel
  - Each worker picks an un-annotated chunk, processes it, marks done
  - No coordinator — workers just pick from the pool
  - Final aggregator combines all annotations

This demonstrates: work queue pattern, parallel processing, no-orchestrator design
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import threading
import concurrent.futures
import time


@dataclass
class WorkItem:
    id: str
    content: str
    annotation: str = ""
    worker_id: Optional[str] = None
    status: str = "pending"     # pending | processing | done


@dataclass
class SwarmSession:
    task_description: str
    work_items: list[WorkItem] = field(default_factory=list)
    num_workers: int = 3
    aggregated_result: str = ""
    wall_time_ms: float = 0.0

    def display(self) -> None:
        done = [w for w in self.work_items if w.status == "done"]
        print(f"\n  Task: {self.task_description}")
        print(f"  Items: {len(self.work_items)} | Workers: {self.num_workers} | Completed: {len(done)}")
        for item in self.work_items:
            icon = "✓" if item.status == "done" else "○"
            worker = f" [worker-{item.worker_id}]" if item.worker_id else ""
            print(f"    {icon} [{item.id}]{worker}: {item.annotation[:80]}")
        if self.aggregated_result:
            print(f"\n  Aggregated Result:\n  {self.aggregated_result}")
        if self.wall_time_ms:
            print(f"\n  Wall time: {self.wall_time_ms:.0f}ms ({self.num_workers} workers)")


# ---------------------------------------------------------------------------
# Swarm worker prompt
# ---------------------------------------------------------------------------

WORKER_SYSTEM = """You are a document annotation agent. Your task: read the given text chunk and produce a brief annotation.

For each chunk, provide:
- Category: what type of content is this? (e.g., technical, narrative, data, opinion)
- Key point: the single most important thing in this chunk (1 sentence)
- Tags: 2-3 keyword tags

Return JSON:
{"category": "...", "key_point": "...", "tags": ["...", "..."]}"""

AGGREGATOR_SYSTEM = """You are an aggregator. Given annotations from multiple agents processing document chunks,
write a coherent summary of the overall document. Include: main themes, content types, and key points.
Keep it under 100 words."""


def worker_prompt(item: WorkItem) -> str:
    return f"Chunk [{item.id}]:\n{item.content}"


def aggregator_prompt(task: str, items: list[WorkItem]) -> str:
    annotations = "\n".join(
        f"  [{item.id}]: {item.annotation[:100]}" for item in items if item.annotation
    )
    return f"Task: {task}\n\nChunk annotations:\n{annotations}\n\nWrite a summary of the overall document."


# ---------------------------------------------------------------------------
# Mock swarm
# ---------------------------------------------------------------------------

MOCK_CHUNKS = [
    ("chunk_1", "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant whitespace."),
    ("chunk_2", "Python 3 introduced breaking changes from Python 2, including print as a function and unicode strings by default. Migration tools like 2to3 helped the ecosystem transition."),
    ("chunk_3", "The Python Package Index (PyPI) hosts over 400,000 packages. pip is the standard package manager. Virtual environments (venv) isolate project dependencies."),
    ("chunk_4", "Python dominates data science with libraries like NumPy, pandas, and matplotlib. The Jupyter notebook ecosystem made interactive computing accessible."),
    ("chunk_5", "Django and Flask are the most popular Python web frameworks. FastAPI is the modern choice for high-performance APIs. All three follow different philosophy: batteries-included vs. micro vs. speed-first."),
]

MOCK_ANNOTATIONS = [
    '{"category": "historical", "key_point": "Python was created in 1991 by Guido van Rossum with a focus on readability.", "tags": ["history", "python", "guido"]}',
    '{"category": "technical", "key_point": "Python 3 broke compatibility with Python 2 but introduced important improvements like unicode support.", "tags": ["python3", "migration", "breaking-changes"]}',
    '{"category": "ecosystem", "key_point": "PyPI hosts 400,000+ packages; pip and venv are the standard dependency management tools.", "tags": ["pypi", "pip", "packages"]}',
    '{"category": "data-science", "key_point": "Python dominates data science via NumPy, pandas, matplotlib, and the Jupyter ecosystem.", "tags": ["data-science", "jupyter", "numpy"]}',
    '{"category": "web", "key_point": "Django, Flask, and FastAPI serve different use cases: full-stack, micro, and high-performance APIs.", "tags": ["django", "flask", "fastapi"]}',
]


def mock_swarm_session() -> SwarmSession:
    session = SwarmSession(
        task_description="Annotate chunks from a Python programming language overview document",
        num_workers=3,
        wall_time_ms=412,
    )
    worker_assignments = ["1", "2", "3", "1", "2"]
    for i, (chunk_id, content) in enumerate(MOCK_CHUNKS):
        item = WorkItem(id=chunk_id, content=content,
                        annotation=MOCK_ANNOTATIONS[i],
                        worker_id=worker_assignments[i],
                        status="done")
        session.work_items.append(item)
    session.aggregated_result = "This document provides an overview of Python: its 1991 origins and readability philosophy, the Python 2→3 transition, the PyPI/pip ecosystem (400K+ packages), Python's dominance in data science (NumPy, pandas, Jupyter), and its web framework landscape (Django for full-stack, Flask for micro, FastAPI for performance). The content is historical, technical, and ecosystem-focused."
    return session


EXAMPLE_DOCUMENT_CHUNKS = [
    [
        ("chunk_1", "Python was created by Guido van Rossum and first released in 1991."),
        ("chunk_2", "Python 3 introduced breaking changes from Python 2, including print as a function."),
        ("chunk_3", "PyPI hosts over 400,000 packages. pip is the standard package manager."),
        ("chunk_4", "Python dominates data science with NumPy, pandas, and matplotlib."),
        ("chunk_5", "Django, Flask, and FastAPI are the most popular Python web frameworks."),
    ],
    [
        ("chunk_1", "Docker containers package applications with their dependencies for consistent deployment."),
        ("chunk_2", "Docker images are built from Dockerfiles using a layered filesystem."),
        ("chunk_3", "Docker Compose orchestrates multi-container applications on a single host."),
        ("chunk_4", "Kubernetes extends container orchestration to clusters of machines."),
        ("chunk_5", "Container registries like Docker Hub and ECR store and distribute images."),
    ],
]
