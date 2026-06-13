"""
Critic Agent
-------------
A dedicated agent whose sole job is to evaluate another agent's output.

Unlike self-reflection (planning exp 08) where the same model critiques itself,
a critic agent is a SEPARATE agent with a different system prompt and potentially
a different model. This gives a more independent perspective.

Pattern:
  1. Generator agent produces output
  2. Critic agent evaluates it against criteria
  3. If critic approves → done
  4. If critic rejects → generator revises (with critique as context)
  5. Repeat until approved or max_rounds

Critic agent design principles:
  - Specific criteria: give the critic explicit dimensions to evaluate
  - Structured output: score + verdict + specific issues (not vague)
  - Independence: different system prompt from the generator
  - Actionable feedback: critique must tell the generator exactly what to fix

Use cases:
  - Code review agent checking generated code
  - Fact-checker agent verifying claims
  - Style agent enforcing brand guidelines
  - Safety agent screening for harmful content
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import re


@dataclass
class CritiqueResult:
    approved: bool
    score: int                         # 1-10
    issues: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    verdict: str = ""


@dataclass
class CriticSession:
    task: str
    rounds: list[tuple[str, CritiqueResult]] = field(default_factory=list)  # (draft, critique)
    final_output: str = ""

    def display(self) -> None:
        print(f"\n  Task: {self.task}")
        for i, (draft, critique) in enumerate(self.rounds, 1):
            print(f"\n  Round {i}:")
            print(f"    Draft: {draft[:100]}{'...' if len(draft) > 100 else ''}")
            print(f"    Score: {critique.score}/10 | Approved: {critique.approved}")
            if critique.issues:
                print(f"    Issues: {'; '.join(critique.issues[:2])}")
            if critique.approved:
                print("    [Critic: APPROVED]")
        if self.final_output:
            print(f"\n  Final Output ({len(self.rounds)} round(s)):\n  {self.final_output}")


# ---------------------------------------------------------------------------
# Generator and Critic prompts
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM = """You are a technical writer. Write clear, accurate technical content.
Be concise, specific, and avoid jargon where possible."""

CRITIC_SYSTEM = """You are a rigorous technical content critic. Evaluate the given content against these criteria:

1. Accuracy: Is it factually correct?
2. Clarity: Is it easy to understand for the target audience?
3. Completeness: Does it cover the key points without being exhaustive?
4. Conciseness: Is it appropriately brief (no padding)?
5. Actionability: Does it give the reader something useful?

Return JSON:
{
  "score": 7,
  "approved": false,
  "issues": ["specific issue 1", "specific issue 2"],
  "strengths": ["strength 1"],
  "verdict": "one sentence summary"
}

approved=true if score >= 7 AND no critical issues. Be strict."""

REVISER_SYSTEM = """You are a technical writer doing a revision. You will receive:
1. The original task
2. Your previous draft
3. A critique with specific issues

Rewrite the content to address ALL listed issues. Return only the revised content."""


def generator_prompt(task: str) -> str:
    return task


def critic_prompt(task: str, draft: str) -> str:
    return f"Task: {task}\n\nContent to evaluate:\n{draft}"


def reviser_prompt(task: str, draft: str, critique: CritiqueResult) -> str:
    issues_str = "\n".join(f"  - {issue}" for issue in critique.issues)
    return f"Task: {task}\n\nPrevious draft:\n{draft}\n\nIssues to fix:\n{issues_str}"


def parse_critique(json_text: str) -> CritiqueResult:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    return CritiqueResult(
        approved=bool(data.get("approved", False)),
        score=int(data.get("score", 5)),
        issues=data.get("issues", []),
        strengths=data.get("strengths", []),
        verdict=data.get("verdict", ""),
    )


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

MOCK_TASK = "Explain what a database index is to a junior developer in 3 sentences."

MOCK_ROUNDS = [
    (
        "A database index is a data structure that improves the speed of data retrieval operations. It works by creating a separate lookup table that the database uses to find data quickly, similar to a book index. Without indexes, the database has to scan every row to find matching records.",
        CritiqueResult(
            approved=False, score=6,
            issues=["Vague about what 'lookup table' means to a junior developer", "Doesn't mention the tradeoff (indexes slow down writes)"],
            strengths=["Good book analogy", "Correct explanation of full table scan"],
            verdict="Accurate but incomplete — misses the key tradeoff a junior dev needs to know.",
        ),
    ),
    (
        "A database index is like a book's index: instead of reading every page to find 'authentication', you jump directly to page 47. Under the hood, it's a sorted data structure (usually a B-tree) that lets the database jump to matching rows instead of scanning every row. The tradeoff: indexes speed up reads but slow down writes and use extra storage, so index thoughtfully.",
        CritiqueResult(
            approved=True, score=9,
            issues=[],
            strengths=["Excellent book analogy", "Mentions B-tree specifically", "Covers the tradeoff clearly"],
            verdict="Clear, accurate, complete, and actionable for a junior developer.",
        ),
    ),
]


def mock_critic_session() -> CriticSession:
    session = CriticSession(task=MOCK_TASK, rounds=MOCK_ROUNDS)
    session.final_output = MOCK_ROUNDS[-1][0]
    return session


EXAMPLE_TASKS = [
    "Explain what a database index is to a junior developer in 3 sentences.",
    "Describe the difference between authentication and authorization in 2 sentences.",
    "Explain what a REST API is to a non-technical product manager.",
    "Write a one-sentence definition of 'technical debt' for a job posting.",
]
