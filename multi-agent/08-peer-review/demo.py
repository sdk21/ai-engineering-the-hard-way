"""
Demo: Peer Review
Usage:
    python demo.py --mock
    python demo.py --real
"""

import argparse
import os
import sys
import json, re

from experiment import (
    EXAMPLE_TASKS, PeerReviewSession, ReviewResult,
    AUTHOR_SYSTEM, REVIEWER_SYSTEM, REVISER_SYSTEM, FINAL_REVIEWER_SYSTEM,
    author_prompt, reviewer_prompt, reviser_prompt, final_review_prompt,
    parse_review, mock_peer_review_session,
)


def mock_demo() -> None:
    print("\n=== Peer Review Demo [MOCK] ===")
    session = mock_peer_review_session()
    session.display()


def real_demo() -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("\n=== Peer Review Demo [REAL] ===")
    print("Example tasks:")
    for i, t in enumerate(EXAMPLE_TASKS, 1):
        print(f"  {i}. {t}")
    print("\nEnter a task (or number 1-4) or 'quit':\n")

    while True:
        try:
            inp = input("Task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit":
            break
        task = EXAMPLE_TASKS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        session = PeerReviewSession(task=task)

        # Author writes
        print("\n  [Author writing...]")
        r1 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=AUTHOR_SYSTEM,
            messages=[{"role": "user", "content": author_prompt(task)}],
        )
        session.author_draft = r1.content[0].text.strip()
        print(f"  Draft: {session.author_draft[:150]}")

        # Reviewer reviews
        print("\n  [Reviewer reviewing...]")
        r2 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=REVIEWER_SYSTEM,
            messages=[{"role": "user", "content": reviewer_prompt(task, session.author_draft)}],
        )
        try:
            session.review = parse_review(r2.content[0].text)
        except Exception as e:
            print(f"  Failed to parse review: {e}"); continue

        print(f"  Score: {session.review.score}/10 | Assessment: {session.review.overall_assessment[:100]}")
        for c in session.review.comments:
            print(f"    [{c.severity.upper()}] {c.line_ref}: {c.issue[:60]}")
            print(f"      → {c.suggestion[:80]}")

        if session.review.approved:
            session.revised_draft = session.author_draft
            session.final_approved = True
            print("\n  [Approved on first review]")
        else:
            # Author revises
            print("\n  [Author revising...]")
            r3 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=REVISER_SYSTEM,
                messages=[{"role": "user", "content": reviser_prompt(task, session.author_draft, session.review)}],
            )
            session.revised_draft = r3.content[0].text.strip()
            print(f"  Revised: {session.revised_draft[:150]}")

            # Final review
            print("\n  [Final review...]")
            r4 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=FINAL_REVIEWER_SYSTEM,
                messages=[{"role": "user", "content": final_review_prompt(
                    task, session.author_draft, session.revised_draft, session.review.comments)}],
            )
            try:
                match = re.search(r'\{.*\}', r4.content[0].text, re.DOTALL)
                data = json.loads(match.group(0)) if match else {}
                session.final_approved = bool(data.get("approved", True))
                note = data.get("note", "")
                print(f"  Final: {'APPROVED' if session.final_approved else 'REJECTED'} — {note}")
            except Exception:
                session.final_approved = True

        session.display()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo()
