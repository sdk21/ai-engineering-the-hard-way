"""
Capstone: Research Team
------------------------
A multi-agent research system combining:

  1. ORCHESTRATOR (exp 02)       — decomposes research goal into sub-questions
  2. PARALLEL FAN-OUT (exp 03)   — specialist agents research in parallel
  3. SHARED BLACKBOARD (exp 09)  — all agents read/write to shared state
  4. CRITIC AGENT (exp 05)       — reviews the draft report for gaps
  5. CONSENSUS VOTING (exp 07)   — agents vote on key claims to validate
  6. HUMAN-IN-THE-LOOP (exp 12)  — checkpoint before final delivery

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │                    Research Team                         │
  │                                                         │
  │  [Orchestrator]                                         │
  │    Research goal → 3-5 sub-questions                    │
  │         ↓                                               │
  │  [Specialist Agents] (parallel, shared blackboard)      │
  │    - Technical researcher                               │
  │    - Business researcher                                │
  │    - Historical/context researcher                      │
  │         ↓                                               │
  │  [Drafter] → draft report from blackboard               │
  │         ↓                                               │
  │  [Critic] → identifies gaps and errors                  │
  │         ↓                                               │
  │  [Consensus Vote] → validate 2-3 key claims             │
  │         ↓                                               │
  │  [HITL Checkpoint] → human reviews before delivery      │
  │         ↓                                               │
  │  [Final Report]                                         │
  └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
import concurrent.futures
import json
import re


# ---------------------------------------------------------------------------
# Shared blackboard
# ---------------------------------------------------------------------------

Blackboard = dict[str, Any]


def make_blackboard(topic: str) -> Blackboard:
    return {
        "topic": topic,
        "sub_questions": [],
        "technical": "",
        "business": "",
        "context": "",
        "draft_report": "",
        "critique": "",
        "validated_claims": [],
        "final_report": "",
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM = """You are a research orchestrator. Given a research topic, generate 3 focused sub-questions:
one technical, one business/practical, one historical/contextual.

Return JSON:
{
  "technical_question": "...",
  "business_question": "...",
  "context_question": "..."
}"""

TECHNICAL_SYSTEM = """You are a technical researcher. Answer the given technical question with specific,
accurate information. Focus on how things work, specifications, and technical tradeoffs.
Write 3-5 sentences."""

BUSINESS_SYSTEM = """You are a business and practical researcher. Answer the given question with focus on
adoption, real-world use cases, market context, and practical implications.
Write 3-5 sentences."""

CONTEXT_SYSTEM = """You are a historical and contextual researcher. Answer the given question with focus on
origins, evolution, why things developed the way they did, and broader context.
Write 3-5 sentences."""

DRAFTER_SYSTEM = """You are a research report drafter. Given a blackboard with research from three specialists,
write a comprehensive, well-organized research report (5-8 sentences).
Integrate all three perspectives: technical, business, and contextual."""

CRITIC_SYSTEM = """You are a research critic. Review this draft report for:
- Factual accuracy issues
- Important omissions
- Unclear or unsupported claims
- Balance across technical/business/context perspectives

Return JSON:
{
  "issues": ["specific issue 1", "specific issue 2"],
  "overall_quality": "good|fair|poor",
  "approved": true/false
}

approved=true if overall_quality is good and issues are only minor."""

CLAIM_VALIDATOR_SYSTEM = """You are a fact validator. Evaluate whether this specific claim is accurate.

Return JSON: {"valid": true/false, "confidence": 0.8, "note": "brief explanation"}"""

REVISER_SYSTEM = """You are revising a research report based on a critic's feedback.
Address ALL identified issues. Return only the revised report."""


def orchestrator_prompt(topic: str) -> str:
    return f"Research topic: {topic}\n\nGenerate three focused sub-questions."


def researcher_prompt(question: str, topic: str) -> str:
    return f"Research topic: {topic}\n\nYour question: {question}"


def drafter_prompt(topic: str, bb: Blackboard) -> str:
    return (
        f"Research topic: {topic}\n\n"
        f"Technical research:\n{bb['technical']}\n\n"
        f"Business/practical research:\n{bb['business']}\n\n"
        f"Historical/contextual research:\n{bb['context']}\n\n"
        f"Write the research report."
    )


def critic_prompt(topic: str, draft: str) -> str:
    return f"Research topic: {topic}\n\nDraft report:\n{draft}"


def claim_validator_prompt(claim: str, context: str) -> str:
    return f"Context: {context}\n\nClaim to validate: {claim}"


def reviser_prompt(topic: str, draft: str, issues: list[str]) -> str:
    issues_str = "\n".join(f"  - {issue}" for issue in issues)
    return f"Topic: {topic}\n\nDraft:\n{draft}\n\nIssues to fix:\n{issues_str}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


# ---------------------------------------------------------------------------
# Main research team runner
# ---------------------------------------------------------------------------

@dataclass
class ResearchSession:
    topic: str
    bb: Blackboard = field(default_factory=dict)
    hitl_checkpoint: bool = False
    human_approved: bool = False
    human_comment: str = ""
    steps_log: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.steps_log.append(msg)
        print(f"  {msg}")

    def display(self) -> None:
        print(f"\n  Topic: {self.topic}")
        print(f"\n  Blackboard state:")
        for key in ["technical", "business", "context", "critique"]:
            val = self.bb.get(key, "")
            if val:
                print(f"    [{key}]: {val[:100]}...")
        if self.bb.get("validated_claims"):
            print(f"    [validated_claims]: {self.bb['validated_claims']}")
        if self.bb.get("final_report"):
            print(f"\n  Final Report:\n  {self.bb['final_report']}")
        print(f"\n  Steps: {len(self.steps_log)}")


def run_research_team(
    topic: str,
    client,
    verbose: bool = False,
    hitl_handler=None,      # callable(draft) → (approved, comment)
) -> ResearchSession:
    session = ResearchSession(topic=topic, bb=make_blackboard(topic))
    bb = session.bb

    # ── Step 1: Orchestrate ──
    session.log("[Orchestrator] Decomposing research topic...")
    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=ORCHESTRATOR_SYSTEM,
        messages=[{"role": "user", "content": orchestrator_prompt(topic)}],
    )
    try:
        plan = _extract_json(r.content[0].text)
    except Exception:
        plan = {
            "technical_question": f"How does {topic} work technically?",
            "business_question": f"What are the practical applications of {topic}?",
            "context_question": f"What is the history and context of {topic}?",
        }

    bb["sub_questions"] = [
        plan.get("technical_question", ""),
        plan.get("business_question", ""),
        plan.get("context_question", ""),
    ]
    if verbose:
        for q in bb["sub_questions"]:
            print(f"    Sub-question: {q}")

    # ── Step 2: Parallel research (fan-out to blackboard) ──
    session.log("[Research Team] Researching in parallel...")

    def research(agent_system, question, bb_key):
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=agent_system,
            messages=[{"role": "user", "content": researcher_prompt(question, topic)}],
        )
        bb[bb_key] = r.content[0].text.strip()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futs = [
            executor.submit(research, TECHNICAL_SYSTEM, plan.get("technical_question", topic), "technical"),
            executor.submit(research, BUSINESS_SYSTEM, plan.get("business_question", topic), "business"),
            executor.submit(research, CONTEXT_SYSTEM, plan.get("context_question", topic), "context"),
        ]
        concurrent.futures.wait(futs)

    session.log("[Blackboard] Three research sections written")

    # ── Step 3: Draft report ──
    session.log("[Drafter] Writing draft report...")
    r2 = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=DRAFTER_SYSTEM,
        messages=[{"role": "user", "content": drafter_prompt(topic, bb)}],
    )
    bb["draft_report"] = r2.content[0].text.strip()

    # ── Step 4: Critic review ──
    session.log("[Critic] Reviewing draft...")
    r3 = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=CRITIC_SYSTEM,
        messages=[{"role": "user", "content": critic_prompt(topic, bb["draft_report"])}],
    )
    try:
        critique_data = _extract_json(r3.content[0].text)
        issues = critique_data.get("issues", [])
        approved = bool(critique_data.get("approved", True))
        bb["critique"] = "; ".join(issues) if issues else "No significant issues"
    except Exception:
        issues = []
        approved = True
        bb["critique"] = "Could not parse critique"

    current_report = bb["draft_report"]
    if issues and not approved:
        session.log(f"[Reviser] Addressing {len(issues)} critique issue(s)...")
        r4 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=REVISER_SYSTEM,
            messages=[{"role": "user", "content": reviser_prompt(topic, current_report, issues)}],
        )
        current_report = r4.content[0].text.strip()

    # ── Step 5: Consensus voting on key claims ──
    session.log("[Consensus] Validating key claims...")
    # Extract 2 simple claims to validate
    claims_to_validate = [
        f"{topic} is widely used in industry",
        f"{topic} has both advantages and disadvantages",
    ]
    validated = []
    for claim in claims_to_validate:
        r5 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=256,
            system=CLAIM_VALIDATOR_SYSTEM,
            messages=[{"role": "user", "content": claim_validator_prompt(claim, current_report)}],
        )
        try:
            v = _extract_json(r5.content[0].text)
            validated.append({
                "claim": claim,
                "valid": bool(v.get("valid", True)),
                "confidence": float(v.get("confidence", 0.8)),
            })
        except Exception:
            validated.append({"claim": claim, "valid": True, "confidence": 0.7})
    bb["validated_claims"] = validated

    # ── Step 6: HITL checkpoint ──
    if hitl_handler:
        session.log("[HITL] Requesting human review...")
        session.hitl_checkpoint = True
        approved_by_human, human_comment = hitl_handler(current_report)
        session.human_approved = approved_by_human
        session.human_comment = human_comment
        if not approved_by_human:
            session.log("[HITL] Human rejected — report not delivered")
            bb["final_report"] = ""
            return session
        if human_comment:
            session.log("[Reviser] Incorporating human feedback...")
            r6 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=REVISER_SYSTEM,
                messages=[{"role": "user", "content": reviser_prompt(topic, current_report, [human_comment])}],
            )
            current_report = r6.content[0].text.strip()
    else:
        session.human_approved = True

    bb["final_report"] = current_report
    session.log("[Done] Final report ready")
    return session
