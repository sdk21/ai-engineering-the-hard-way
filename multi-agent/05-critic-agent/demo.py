"""
Demo: Critic Agent
Usage:
    python demo.py --mock
    python demo.py --real [--max-rounds 3]
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_TASKS, CriticSession,
    GENERATOR_SYSTEM, CRITIC_SYSTEM, REVISER_SYSTEM,
    generator_prompt, critic_prompt, reviser_prompt,
    parse_critique, mock_critic_session,
)


def mock_demo() -> None:
    print("\n=== Critic Agent Demo [MOCK] ===")
    session = mock_critic_session()
    session.display()


def real_demo(max_rounds: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Critic Agent Demo [REAL, max_rounds={max_rounds}] ===")
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

        session = CriticSession(task=task)
        current_draft = None

        for round_num in range(1, max_rounds + 1):
            print(f"\n  --- Round {round_num} ---")

            if current_draft is None:
                print("  [Generator...]")
                r = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=512,
                    system=GENERATOR_SYSTEM,
                    messages=[{"role": "user", "content": generator_prompt(task)}],
                )
                current_draft = r.content[0].text.strip()

            print(f"  Draft: {current_draft[:120]}")

            print("  [Critic...]")
            r2 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=CRITIC_SYSTEM,
                messages=[{"role": "user", "content": critic_prompt(task, current_draft)}],
            )
            try:
                critique = parse_critique(r2.content[0].text)
            except Exception:
                print("  [Failed to parse critique]"); break

            session.rounds.append((current_draft, critique))
            print(f"  Score: {critique.score}/10 | Verdict: {critique.verdict}")

            if critique.approved:
                print("  [APPROVED]")
                break

            if critique.issues:
                print(f"  Issues: {'; '.join(critique.issues[:2])}")

            if round_num == max_rounds:
                print(f"  [Max rounds reached]")
                break

            print("  [Revising...]")
            r3 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=REVISER_SYSTEM,
                messages=[{"role": "user", "content": reviser_prompt(task, current_draft, critique)}],
            )
            current_draft = r3.content[0].text.strip()

        session.final_output = current_draft or ""
        print(f"\n  Final Output ({len(session.rounds)} round(s)):\n  {session.final_output}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(max_rounds=args.max_rounds)
