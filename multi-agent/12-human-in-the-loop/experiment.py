"""
Human-in-the-Loop (HITL)
-------------------------
The agent pauses at designated checkpoints and requests human approval,
clarification, or correction before continuing.

Why HITL?
  - High-stakes actions (sending emails, deploying code, making purchases)
  - Ambiguous situations where the agent isn't confident
  - Compliance requirements (regulated industries)
  - Trust building: users verify before autonomous action takes effect

Checkpoint types:
  1. Approval gate: agent proposes action, human approves/rejects
  2. Clarification request: agent is unsure, asks human for input
  3. Review gate: agent shows completed work, human reviews before delivery
  4. Escalation: agent detects it can't handle this, routes to human

HITL is NOT a fallback for bad agents. It's a design choice:
  - Know WHICH decisions require human judgment
  - Know WHEN the agent's confidence is too low to proceed
  - Know WHAT level of detail humans need to make a good decision

This experiment demonstrates a content moderation + email drafting workflow:
  Agent drafts a response to customer complaint
  → HITL review before sending
  → Human can approve, edit, or reject
  → Agent sends (or revises and loops)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable


class CheckpointType(Enum):
    APPROVAL = "approval"
    CLARIFICATION = "clarification"
    REVIEW = "review"
    ESCALATION = "escalation"


class HumanDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    ESCALATE = "escalate"


@dataclass
class Checkpoint:
    type: CheckpointType
    question: str                        # what the agent is asking the human
    context: str                         # relevant context for the human
    proposed_action: str = ""            # what the agent wants to do
    human_decision: Optional[HumanDecision] = None
    human_comment: str = ""              # human's feedback or edit


@dataclass
class HITLSession:
    task: str
    checkpoints: list[Checkpoint] = field(default_factory=list)
    agent_draft: str = ""
    final_output: str = ""
    approved: bool = False
    human_interventions: int = 0

    def display(self) -> None:
        print(f"\n  Task: {self.task}")
        print(f"\n  Agent Draft:\n  {self.agent_draft[:200]}")
        for i, cp in enumerate(self.checkpoints, 1):
            print(f"\n  Checkpoint {i} [{cp.type.value.upper()}]:")
            print(f"    Question: {cp.question}")
            print(f"    Decision: {cp.human_decision.value if cp.human_decision else 'pending'}")
            if cp.human_comment:
                print(f"    Comment: {cp.human_comment}")
        if self.final_output:
            print(f"\n  Final Output:\n  {self.final_output}")
        print(f"\n  Approved: {self.approved} | Human interventions: {self.human_interventions}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DRAFTER_SYSTEM = """You are a customer success agent drafting responses to customer complaints.
Write professional, empathetic, and solution-oriented responses.
Keep responses concise (3-5 sentences)."""

CONFIDENCE_CHECKER_SYSTEM = """You are evaluating whether an AI-drafted customer response should be
sent automatically or reviewed by a human.

Flag for human review if ANY of the following:
- The complaint involves a refund over $100
- Legal or compliance language is present
- The customer explicitly asked to speak to a human
- The situation involves potential data breach or security issue
- Sentiment is extremely negative or the customer seems escalated

Return JSON: {"needs_review": true/false, "reason": "why", "risk_level": "low|medium|high"}"""

REVISER_SYSTEM = """You are revising a customer response based on human feedback.
Incorporate the human's edits and preferences. Return only the revised response."""


def drafter_prompt(complaint: str) -> str:
    return f"Customer complaint:\n{complaint}\n\nDraft a response."


def confidence_prompt(complaint: str, draft: str) -> str:
    return f"Customer complaint:\n{complaint}\n\nDraft response:\n{draft}\n\nShould this be reviewed by a human?"


def reviser_prompt(complaint: str, draft: str, human_comment: str) -> str:
    return f"Complaint:\n{complaint}\n\nOriginal draft:\n{draft}\n\nHuman feedback:\n{human_comment}\n\nRevise the response."


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

MOCK_COMPLAINT = "I've been charged $250 for a service I cancelled 3 months ago. I've contacted support 4 times and nobody has fixed this. I'm going to dispute this with my credit card company if this isn't resolved TODAY."

MOCK_DRAFT = "Thank you for reaching out and I sincerely apologize for this frustrating experience. Charging you for a cancelled service is unacceptable, and I can see why you're upset after contacting us multiple times. I'm escalating this to our billing team right now to process your $250 refund immediately — you'll receive a confirmation email within the hour. Please reply to this message if you don't receive it, and I'll personally ensure it's resolved."

MOCK_CHECKPOINT = Checkpoint(
    type=CheckpointType.REVIEW,
    question="This response involves a $250 refund. Please review before sending.",
    context="Customer complaint involves: disputed charge ($250), repeated contact failures (4x), threat of chargeback.",
    proposed_action="Send the drafted response and initiate a $250 refund.",
    human_decision=HumanDecision.EDIT,
    human_comment="Good tone. Add our refund processing timeframe (3-5 business days for bank credit). Don't promise 'within the hour' — that's not realistic.",
)

MOCK_FINAL = "Thank you for reaching out and I sincerely apologize for this frustrating experience. Charging you for a cancelled service is unacceptable, and I can see why you're upset after contacting us multiple times without resolution. I'm escalating this to our billing team now to process your $250 refund — you'll receive a confirmation email today, with the credit appearing in your account within 3-5 business days. Please reply if you have any questions."


def mock_hitl_session() -> HITLSession:
    session = HITLSession(task="Draft and send a response to an escalated billing complaint")
    session.agent_draft = MOCK_DRAFT
    session.checkpoints = [MOCK_CHECKPOINT]
    session.final_output = MOCK_FINAL
    session.approved = True
    session.human_interventions = 1
    return session


EXAMPLE_COMPLAINTS = [
    "I've been charged $250 for a service I cancelled 3 months ago. I've contacted support 4 times with no resolution. I'm disputing with my credit card company if not fixed TODAY.",
    "Your app deleted all my project data after the update. Two years of work gone. I need this restored immediately.",
    "When will the export feature be ready? I've been waiting 6 months since you announced it.",
    "I love the new dashboard! Just wanted to say the team did a great job.",
]
