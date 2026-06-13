"""
Self-Reflection
----------------
A model reviews its own output and iteratively improves it.

Pattern:
  1. DRAFT  — generate initial response
  2. REFLECT — critique the draft (what's wrong, missing, unclear?)
  3. REVISE  — produce improved response addressing the critique
  4. Repeat until: critique says "no issues" OR max_rounds reached

Key concepts:
- Self-critique: model acts as its own reviewer
- Iterative refinement: each round improves on the last
- Termination: explicit "LGTM" / "no issues" signal, or max_rounds
- Separate prompts for generation vs. critique → different perspectives

Why it works:
  Models are better at *identifying* errors than avoiding them on first pass.
  A second pass with explicit focus on critique catches mistakes that slipped
  through in generation mode.

Variants:
  - Single-model: same model generates and critiques (shown here)
  - Multi-model: separate models for generation and critique
  - Constitutional AI: critique against a fixed set of principles
  - Self-consistency: generate N drafts, pick the best one (exp 02)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class ReflectionRound:
    round_num: int
    draft: str
    critique: str
    revised: str = ""
    is_final: bool = False          # True if critique said no issues


@dataclass
class ReflectionSession:
    task: str
    rounds: list[ReflectionRound] = field(default_factory=list)
    final_answer: str = ""

    def display(self, verbose: bool = False) -> None:
        print(f"\n  Task: {self.task}")
        for r in self.rounds:
            print(f"\n  --- Round {r.round_num} ---")
            if verbose:
                print(f"  Draft:\n    {r.draft[:200]}")
            else:
                print(f"  Draft: {r.draft[:100]}{'...' if len(r.draft) > 100 else ''}")
            print(f"  Critique: {r.critique[:120]}{'...' if len(r.critique) > 120 else ''}")
            if r.is_final:
                print("  [Critique: no issues found — accepted]")
            elif r.revised:
                print(f"  Revised: {r.revised[:100]}{'...' if len(r.revised) > 100 else ''}")
        if self.final_answer:
            print(f"\n  Final answer ({len(self.rounds)} round(s)):\n  {self.final_answer}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DRAFTER_SYSTEM = """You are a helpful assistant. Answer the given task directly and completely. Be specific and accurate."""

CRITIC_SYSTEM = """You are a critical reviewer. Your job is to find problems in a draft response.

Look for:
- Factual errors or unsupported claims
- Missing important information
- Logical inconsistencies
- Unclear or ambiguous statements
- Unnecessary repetition or padding

If the response is good enough to deliver as-is, reply with exactly:
  NO_ISSUES

Otherwise, list the specific problems concisely (2-5 bullet points)."""

REVISER_SYSTEM = """You are a revision assistant. You will be given:
1. The original task
2. A draft response
3. A critique of the draft

Rewrite the response to address ALL critique points. Produce only the revised response — no meta-commentary."""


def drafter_prompt(task: str) -> str:
    return task


def critic_prompt(task: str, draft: str) -> str:
    return f"Task: {task}\n\nDraft response:\n{draft}"


def reviser_prompt(task: str, draft: str, critique: str) -> str:
    return f"Task: {task}\n\nDraft:\n{draft}\n\nCritique:\n{critique}\n\nRevised response:"


def is_lgtm(critique: str) -> bool:
    return "NO_ISSUES" in critique.upper() or critique.strip().upper() == "NO_ISSUES"


# ---------------------------------------------------------------------------
# Mock session
# ---------------------------------------------------------------------------

MOCK_TASK = "Explain why the sky is blue in 2-3 sentences suitable for a 10-year-old."

MOCK_ROUNDS = [
    ReflectionRound(
        round_num=1,
        draft="The sky is blue because of Rayleigh scattering. When sunlight enters the atmosphere, shorter blue wavelengths scatter more than longer red wavelengths. This scattered blue light reaches your eyes from all directions.",
        critique="• Uses jargon ('Rayleigh scattering', 'wavelengths') without explanation — a 10-year-old won't understand these terms.\n• 'From all directions' is accurate but could be more vivid.",
    ),
    ReflectionRound(
        round_num=2,
        draft="The sky looks blue because of the way sunlight plays with the air. Sunlight is actually made of all the colors of the rainbow mixed together. When it hits tiny air particles, the blue part bounces around much more than the other colors — so when you look up, you see blue coming at you from everywhere!",
        critique="NO_ISSUES",
        is_final=True,
    ),
]


def mock_reflection_session() -> ReflectionSession:
    session = ReflectionSession(task=MOCK_TASK, rounds=MOCK_ROUNDS)
    session.final_answer = MOCK_ROUNDS[-1].draft
    return session


EXAMPLE_TASKS = [
    "Explain why the sky is blue in 2-3 sentences suitable for a 10-year-old.",
    "Write a one-paragraph bio for Albert Einstein suitable for a museum placard.",
    "Summarize the pros and cons of remote work in 4 bullet points.",
    "Explain what a database index is to a junior developer in 3 sentences.",
]
