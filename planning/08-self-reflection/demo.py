"""
Demo: Self-Reflection
Usage:
    python demo.py --mock
    python demo.py --real [--rounds 3] [--verbose]
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_TASKS, ReflectionRound, ReflectionSession,
    DRAFTER_SYSTEM, CRITIC_SYSTEM, REVISER_SYSTEM,
    drafter_prompt, critic_prompt, reviser_prompt,
    is_lgtm, mock_reflection_session,
)


def mock_demo() -> None:
    print("\n=== Self-Reflection Demo [MOCK] ===")
    session = mock_reflection_session()
    session.display(verbose=True)


def real_demo(max_rounds: int, verbose: bool) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Self-Reflection Demo [REAL, max_rounds={max_rounds}] ===")
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

        session = ReflectionSession(task=task)
        current_draft = None

        for round_num in range(1, max_rounds + 1):
            print(f"\n  --- Round {round_num} ---")

            # Draft (or use previous revision as new draft)
            if current_draft is None:
                print("  [Drafting...]")
                r = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=512,
                    system=DRAFTER_SYSTEM,
                    messages=[{"role": "user", "content": drafter_prompt(task)}],
                )
                current_draft = r.content[0].text.strip()

            if verbose:
                print(f"  Draft:\n    {current_draft}")
            else:
                print(f"  Draft: {current_draft[:100]}{'...' if len(current_draft) > 100 else ''}")

            # Critique
            print("  [Critiquing...]")
            r2 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=CRITIC_SYSTEM,
                messages=[{"role": "user", "content": critic_prompt(task, current_draft)}],
            )
            critique = r2.content[0].text.strip()
            print(f"  Critique: {critique[:120]}{'...' if len(critique) > 120 else ''}")

            rnd = ReflectionRound(round_num=round_num, draft=current_draft, critique=critique)

            if is_lgtm(critique):
                rnd.is_final = True
                session.rounds.append(rnd)
                print("  [Accepted — no issues found]")
                break

            # Revise
            print("  [Revising...]")
            r3 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=REVISER_SYSTEM,
                messages=[{"role": "user", "content": reviser_prompt(task, current_draft, critique)}],
            )
            revised = r3.content[0].text.strip()
            rnd.revised = revised
            session.rounds.append(rnd)
            current_draft = revised

            if round_num == max_rounds:
                print(f"  [Max rounds ({max_rounds}) reached]")

        session.final_answer = current_draft or ""
        print(f"\n  Final answer ({len(session.rounds)} round(s)):")
        print(f"  {session.final_answer}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(max_rounds=args.rounds, verbose=args.verbose)
