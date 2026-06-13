"""
Demo: Adaptive Planning
Usage:
    python demo.py --mock
    python demo.py --real
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_GOALS, AdaptiveStep, AdaptivePlan, StepStatus,
    PLANNER_SYSTEM, EXECUTOR_SYSTEM, ADAPTER_SYSTEM, SYNTHESIZER_SYSTEM,
    planner_prompt, executor_prompt, adapter_prompt, synthesizer_prompt,
    parse_plan, parse_adaptation, mock_adaptive_plan,
)


def mock_demo() -> None:
    print("\n=== Adaptive Planning Demo [MOCK] ===")
    plan = mock_adaptive_plan()
    plan.display(title="Adaptive Plan")


def real_demo() -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("\n=== Adaptive Planning Demo [REAL] ===")
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

        # Initial plan
        print("\n  [Planning...]")
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": planner_prompt(goal)}],
        )
        try:
            plan = parse_plan(goal, r.content[0].text)
        except Exception as e:
            print(f"  Failed to parse plan: {e}"); continue

        plan.display(title="Initial Plan")
        print()

        # Execute with adaptation
        step_counter = [len(plan.steps)]

        while True:
            pending = plan.pending_steps()
            if not pending:
                break

            step = pending[0]
            step.status = StepStatus.DONE
            print(f"  [Executing step {step.index}: {step.description[:50]}...]")

            r2 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=EXECUTOR_SYSTEM,
                messages=[{"role": "user", "content": executor_prompt(goal, step, plan.completed_steps())}],
            )
            step.result = r2.content[0].text.strip()
            print(f"    → {step.result[:80]}")

            # Check if plan needs adaptation (only if there are remaining steps)
            remaining = plan.pending_steps()
            if not remaining:
                break

            print(f"  [Checking if plan needs adaptation...]")
            r3 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=ADAPTER_SYSTEM,
                messages=[{"role": "user", "content": adapter_prompt(plan)}],
            )
            try:
                adapt, reason, new_steps = parse_adaptation(r3.content[0].text)
            except Exception:
                adapt = False
                reason = ""
                new_steps = []

            if adapt and new_steps:
                plan.adaptations += 1
                # Replace remaining pending steps with new ones
                plan.steps = [s for s in plan.steps if s.status != StepStatus.PENDING]
                for desc in new_steps:
                    step_counter[0] += 1
                    plan.steps.append(AdaptiveStep(
                        index=step_counter[0],
                        description=desc,
                        is_inserted=True,
                    ))
                print(f"    [Plan adapted: {reason[:80]}]")
                print(f"    New remaining steps: {len(new_steps)}")

        # Synthesize
        print("\n  [Synthesizing final answer...]")
        r4 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=SYNTHESIZER_SYSTEM,
            messages=[{"role": "user", "content": synthesizer_prompt(plan)}],
        )
        plan.final_answer = r4.content[0].text.strip()

        plan.display(title="Final Plan")
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
