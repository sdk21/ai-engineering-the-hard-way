"""
Demo: Replanning
Usage:
    python demo.py --mock
    python demo.py --real [--max-replans 3]
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_GOALS, ReplanStep, ReplanSession, StepStatus,
    PLANNER_SYSTEM, EXECUTOR_SYSTEM, REPLANNER_SYSTEM, SYNTHESIZER_SYSTEM,
    planner_prompt, executor_prompt, replanner_prompt, synthesizer_prompt,
    parse_plan_steps, is_failure, mock_replan_session,
)


def mock_demo() -> None:
    print("\n=== Replanning Demo [MOCK] ===")
    session = mock_replan_session()
    session.display()


def real_demo(max_replans: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Replanning Demo [REAL, max_replans={max_replans}] ===")
    print("Example goals:")
    for i, g in enumerate(EXAMPLE_GOALS, 1):
        print(f"  {i}. {g}")
    print("\nEnter a goal (or number 1-4) or 'quit':\n")

    while True:
        try:
            inp = input("Goal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit":
            break
        goal = EXAMPLE_GOALS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        session = ReplanSession(goal=goal)
        step_counter = [0]
        plan_version = [1]

        def make_steps(descriptions: list[str]) -> list[ReplanStep]:
            steps = []
            for desc in descriptions:
                step_counter[0] += 1
                steps.append(ReplanStep(
                    index=step_counter[0],
                    description=desc,
                    plan_version=plan_version[0],
                ))
            return steps

        # Initial plan
        print("\n  [Planning...]")
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": planner_prompt(goal)}],
        )
        try:
            step_descs = parse_plan_steps(r.content[0].text)
        except Exception as e:
            print(f"  Failed to parse plan: {e}"); continue

        session.current_plan = make_steps(step_descs)
        print(f"  Initial plan: {len(session.current_plan)} steps")

        while session.current_plan:
            step = session.current_plan.pop(0)
            step.status = StepStatus.RUNNING
            session.history.append(step)

            completed_done = [s for s in session.history if s.status == StepStatus.DONE]
            print(f"\n  [Step {step.index} v{step.plan_version}: {step.description[:60]}]")

            r2 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=EXECUTOR_SYSTEM,
                messages=[{"role": "user", "content": executor_prompt(goal, step, completed_done)}],
            )
            result_text = r2.content[0].text.strip()
            failed, reason = is_failure(result_text)

            if failed:
                step.status = StepStatus.FAILED
                step.failure_reason = reason
                print(f"    ✗ FAILED: {reason}")

                if session.replan_count >= max_replans:
                    print(f"  [Max replans ({max_replans}) reached — stopping]")
                    session.current_plan = []
                    break

                # Replan
                plan_version[0] += 1
                session.replan_count += 1
                print(f"  [Replanning (attempt {session.replan_count})...]")

                r3 = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=512,
                    system=REPLANNER_SYSTEM,
                    messages=[{"role": "user", "content": replanner_prompt(session, step)}],
                )
                try:
                    new_descs = parse_plan_steps(r3.content[0].text)
                    session.current_plan = make_steps(new_descs)
                    print(f"  New plan: {len(session.current_plan)} steps")
                except Exception as e:
                    print(f"  Failed to parse replan: {e}")
                    session.current_plan = []
                    break
            else:
                step.status = StepStatus.DONE
                step.result = result_text
                print(f"    ✓ {result_text[:80]}")

        # Check if we have enough completed steps
        completed = [s for s in session.history if s.status == StepStatus.DONE]
        if completed:
            print("\n  [Synthesizing...]")
            r4 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=SYNTHESIZER_SYSTEM,
                messages=[{"role": "user", "content": synthesizer_prompt(session)}],
            )
            session.final_answer = r4.content[0].text.strip()
            session.solved = True

        session.display()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--max-replans", type=int, default=3)
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(max_replans=args.max_replans)
