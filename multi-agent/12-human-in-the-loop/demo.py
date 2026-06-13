"""
Demo: Human-in-the-Loop
Usage:
    python demo.py --mock
    python demo.py --real [--auto-approve]
"""

import argparse
import os
import sys
import json
import re

from experiment import (
    EXAMPLE_COMPLAINTS, HITLSession, Checkpoint, CheckpointType, HumanDecision,
    DRAFTER_SYSTEM, CONFIDENCE_CHECKER_SYSTEM, REVISER_SYSTEM,
    drafter_prompt, confidence_prompt, reviser_prompt, mock_hitl_session,
)


def mock_demo() -> None:
    print("\n=== Human-in-the-Loop Demo [MOCK] ===")
    session = mock_hitl_session()
    session.display()


def request_human_decision(checkpoint: Checkpoint) -> tuple[HumanDecision, str]:
    """Prompt the human for a decision at a checkpoint."""
    print(f"\n  ┌─ CHECKPOINT [{checkpoint.type.value.upper()}] ─────────────────")
    print(f"  │ {checkpoint.question}")
    print(f"  │")
    print(f"  │ Context: {checkpoint.context[:120]}")
    print(f"  │")
    print(f"  │ Proposed action: {checkpoint.proposed_action[:100]}")
    print(f"  │")
    print(f"  │ Options: [a]pprove / [r]eject / [e]dit / [s]escalate")
    print(f"  └─────────────────────────────────────────────────────")

    while True:
        try:
            choice = input("  Your decision: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return HumanDecision.REJECT, "interrupted"

        if choice in ("a", "approve"):
            return HumanDecision.APPROVE, ""
        elif choice in ("r", "reject"):
            reason = input("  Reason for rejection: ").strip()
            return HumanDecision.REJECT, reason
        elif choice in ("e", "edit"):
            feedback = input("  Your feedback/edits: ").strip()
            return HumanDecision.EDIT, feedback
        elif choice in ("s", "escalate"):
            return HumanDecision.ESCALATE, "Escalated to human agent"
        else:
            print("  Enter 'a', 'r', 'e', or 's'")


def real_demo(auto_approve: bool) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("\n=== Human-in-the-Loop Demo [REAL] ===")
    if auto_approve:
        print("  [Auto-approve mode: no human prompts]")
    print("Example complaints:")
    for i, c in enumerate(EXAMPLE_COMPLAINTS, 1):
        print(f"  {i}. {c[:70]}")
    print("\nEnter a complaint (or number 1-4) or 'quit':\n")

    while True:
        try:
            inp = input("Complaint: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit":
            break
        complaint = EXAMPLE_COMPLAINTS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        session = HITLSession(task="Draft and send a response to a customer complaint")

        # Step 1: Agent drafts response
        print("\n  [Agent drafting response...]")
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=DRAFTER_SYSTEM,
            messages=[{"role": "user", "content": drafter_prompt(complaint)}],
        )
        session.agent_draft = r.content[0].text.strip()
        print(f"\n  Agent Draft:\n  {session.agent_draft}")

        # Step 2: Confidence check — should this go to human review?
        print("\n  [Checking if human review is needed...]")
        r2 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=256,
            system=CONFIDENCE_CHECKER_SYSTEM,
            messages=[{"role": "user", "content": confidence_prompt(complaint, session.agent_draft)}],
        )
        try:
            match = re.search(r'\{.*\}', r2.content[0].text, re.DOTALL)
            check = json.loads(match.group(0)) if match else {}
            needs_review = bool(check.get("needs_review", False))
            reason = check.get("reason", "")
            risk_level = check.get("risk_level", "low")
        except Exception:
            needs_review = True
            reason = "Could not assess risk"
            risk_level = "medium"

        print(f"  Risk level: {risk_level} | Needs review: {needs_review}")
        if reason:
            print(f"  Reason: {reason}")

        current_draft = session.agent_draft
        final_decision = HumanDecision.APPROVE

        if needs_review and not auto_approve:
            checkpoint = Checkpoint(
                type=CheckpointType.REVIEW,
                question=f"Risk level: {risk_level}. {reason}. Please review before sending.",
                context=f"Complaint: {complaint[:120]}",
                proposed_action=f"Send the drafted response to the customer.",
            )
            decision, comment = request_human_decision(checkpoint)
            checkpoint.human_decision = decision
            checkpoint.human_comment = comment
            session.checkpoints.append(checkpoint)
            session.human_interventions += 1
            final_decision = decision

            if decision == HumanDecision.EDIT and comment:
                print("\n  [Agent revising based on feedback...]")
                r3 = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=512,
                    system=REVISER_SYSTEM,
                    messages=[{"role": "user", "content": reviser_prompt(complaint, current_draft, comment)}],
                )
                current_draft = r3.content[0].text.strip()
                final_decision = HumanDecision.APPROVE  # Assume approved after revision
        elif auto_approve:
            print("  [Auto-approved]")

        if final_decision in (HumanDecision.APPROVE, HumanDecision.EDIT):
            session.final_output = current_draft
            session.approved = True
            print(f"\n  [SENDING]\n  {session.final_output}")
        elif final_decision == HumanDecision.REJECT:
            print("\n  [REJECTED — response not sent]")
        elif final_decision == HumanDecision.ESCALATE:
            print("\n  [ESCALATED to human agent]")

        session.display()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--auto-approve", action="store_true", help="Skip human prompts")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(auto_approve=args.auto_approve)
