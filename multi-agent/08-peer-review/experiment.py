"""
Peer Review
------------
Agent A produces work. Agent B reviews it and provides feedback.
Agent A revises based on the feedback. Optionally, Agent B signs off.

This is the academic peer review process applied to LLM outputs.

Compared to Critic Agent (exp 05):
  Critic Agent: one agent critiques, one revises (generator/critic are distinct)
  Peer Review: reviewer gives structured feedback WITH suggested improvements,
               then the original author revises, then reviewer signs off

The key distinction: the peer reviewer doesn't just say what's wrong —
they suggest HOW to fix it. This makes revision more targeted.

Use cases:
  - Code review (reviewer spots bugs AND suggests fixes)
  - Technical writing (reviewer fixes unclear sections specifically)
  - Data analysis (reviewer questions methodology AND suggests alternatives)
  - Design review (reviewer flags usability issues AND proposes changes)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import re


@dataclass
class ReviewComment:
    line_ref: str              # e.g. "paragraph 2" or "line 5"
    issue: str
    suggestion: str
    severity: str              # "minor", "major", "critical"


@dataclass
class ReviewResult:
    approved: bool
    overall_assessment: str
    comments: list[ReviewComment] = field(default_factory=list)
    score: int = 0             # 1-10


@dataclass
class PeerReviewSession:
    task: str
    author_draft: str = ""
    review: Optional[ReviewResult] = None
    revised_draft: str = ""
    final_approved: bool = False

    def display(self) -> None:
        print(f"\n  Task: {self.task}")
        if self.author_draft:
            print(f"\n  Author's Draft:\n  {self.author_draft[:200]}")
        if self.review:
            print(f"\n  Review (score: {self.review.score}/10, approved: {self.review.approved}):")
            print(f"  Assessment: {self.review.overall_assessment[:120]}")
            for c in self.review.comments:
                print(f"    [{c.severity.upper()}] {c.line_ref}: {c.issue}")
                print(f"      Suggestion: {c.suggestion[:80]}")
        if self.revised_draft:
            print(f"\n  Revised Draft:\n  {self.revised_draft[:200]}")
        if self.final_approved:
            print("\n  [APPROVED by reviewer]")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

AUTHOR_SYSTEM = """You are a technical author. Write clear, accurate technical content.
Focus on correctness and completeness."""

REVIEWER_SYSTEM = """You are a technical peer reviewer. Review the given content carefully.

Provide:
1. Overall assessment (1-2 sentences)
2. Specific comments with suggestions for improvement
3. A score (1-10) and approval decision (approve if score >= 7)

Return JSON:
{
  "score": 7,
  "approved": false,
  "overall_assessment": "...",
  "comments": [
    {
      "line_ref": "paragraph 1",
      "issue": "what's wrong",
      "suggestion": "specific fix",
      "severity": "minor|major|critical"
    }
  ]
}

Be constructive: every issue must include a concrete suggestion."""

REVISER_SYSTEM = """You are the original author revising your work based on peer review feedback.
Address ALL comments from the reviewer. For each comment, apply the suggested fix or explain why you're taking a different approach.
Return only the revised content."""

FINAL_REVIEWER_SYSTEM = """You are a peer reviewer doing a final check after revisions.
The author has addressed your previous comments. Confirm whether the revision is satisfactory.
Return JSON: {"approved": true/false, "note": "brief note"}"""


def author_prompt(task: str) -> str:
    return task


def reviewer_prompt(task: str, draft: str) -> str:
    return f"Task: {task}\n\nContent to review:\n{draft}"


def reviser_prompt(task: str, draft: str, review: ReviewResult) -> str:
    comments_str = "\n".join(
        f"  [{c.severity.upper()}] {c.line_ref}: {c.issue}\n  Suggestion: {c.suggestion}"
        for c in review.comments
    )
    return f"Task: {task}\n\nOriginal draft:\n{draft}\n\nReviewer comments:\n{comments_str}"


def final_review_prompt(task: str, original: str, revised: str, original_comments: list[ReviewComment]) -> str:
    comments_str = "\n".join(f"  - {c.issue}" for c in original_comments)
    return (f"Task: {task}\n\n"
            f"Original issues:\n{comments_str}\n\n"
            f"Revised content:\n{revised}\n\n"
            f"Are the issues addressed?")


def parse_review(json_text: str) -> ReviewResult:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    comments = [
        ReviewComment(
            line_ref=c.get("line_ref", ""),
            issue=c.get("issue", ""),
            suggestion=c.get("suggestion", ""),
            severity=c.get("severity", "minor"),
        )
        for c in data.get("comments", [])
    ]
    return ReviewResult(
        approved=bool(data.get("approved", False)),
        overall_assessment=data.get("overall_assessment", ""),
        comments=comments,
        score=int(data.get("score", 5)),
    )


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

MOCK_TASK = "Write a technical explanation of how HTTPS works for a junior developer blog post (3-4 sentences)."


def mock_peer_review_session() -> PeerReviewSession:
    session = PeerReviewSession(task=MOCK_TASK)
    session.author_draft = "HTTPS is the secure version of HTTP. It uses SSL/TLS to encrypt data between the browser and server. This prevents hackers from reading your data. Websites with HTTPS show a padlock icon."
    session.review = ReviewResult(
        approved=False, score=6,
        overall_assessment="Correct but too surface-level. Missing the certificate/PKI explanation that's essential for a developer audience.",
        comments=[
            ReviewComment("sentence 2", "Mentions SSL/TLS but doesn't explain what they do (handshake, asymmetric → symmetric key exchange)", "Add: 'During the TLS handshake, the server shares its public key certificate; the browser uses it to establish a shared symmetric key for the session.'", "major"),
            ReviewComment("sentence 3", "'Hackers' is vague — be specific about the threat (man-in-the-middle, eavesdropping)", "Replace with: 'This prevents eavesdropping and man-in-the-middle attacks.'", "minor"),
        ]
    )
    session.revised_draft = "HTTPS is the secure version of HTTP. It uses TLS (Transport Layer Security) to encrypt communication: during the TLS handshake, the server presents a certificate containing its public key; the browser uses this to establish a shared symmetric session key. This prevents eavesdropping and man-in-the-middle attacks — an interceptor sees only encrypted bytes. Websites using HTTPS show a padlock icon in the browser address bar."
    session.final_approved = True
    return session


EXAMPLE_TASKS = [
    "Write a technical explanation of how HTTPS works for a junior developer blog post (3-4 sentences).",
    "Write a 3-sentence explanation of what a load balancer does for a product manager.",
    "Explain the difference between a process and a thread in 3 sentences.",
    "Write a brief description of what CI/CD is for a new engineering hire.",
]
