"""
Demo: Task Decomposition
Usage:
    python demo.py --mock
    python demo.py --real [--execute]
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_GOALS, Plan, Step, StepStatus,
    DECOMPOSE_SYSTEM, decompose_prompt, parse_plan, simulate_execute_step,
)

MOCK_PLAN = """1. Research the latest scientific studies on exercise benefits
2. Outline the blog post structure (intro, 3 main points, conclusion)
3. Write the first draft of the blog post
4. Review and edit the draft for clarity and flow
5. Add relevant statistics and citations
6. Proofread for grammar and spelling errors
7. Publish to the blog platform and share on social media"""


def mock_demo() -> None:
    print("\n=== Task Decomposition Demo [MOCK] ===\n")
    goal = EXAMPLE_GOALS[0]
    print(f"  Goal: {goal}\n")
    print("  Generated plan:")
    plan = parse_plan(goal, MOCK_PLAN)
    plan.display()
    print("\n  Simulating execution...")
    for step in plan.steps:
        step.status = StepStatus.RUNNING
        result = simulate_execute_step(step, plan)
        step.result = result
        step.status = StepStatus.DONE
        print(f"    ✓ Step {step.index}: {result}")
    print(f"\n  Plan complete: {plan.is_complete()}")


def real_demo(execute: bool) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Task Decomposition Demo [REAL] ===")
    print("Example goals:")
    for i, g in enumerate(EXAMPLE_GOALS, 1):
        print(f"  {i}. {g}")
    print("\nEnter a goal (or number 1-5) or 'quit':\n")

    while True:
        try:
            inp = input("Goal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit": break
        goal = EXAMPLE_GOALS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 5 else inp

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=DECOMPOSE_SYSTEM,
            messages=[{"role": "user", "content": decompose_prompt(goal)}],
        )
        plan = parse_plan(goal, response.content[0].text)
        plan.display()

        if execute:
            print("\n  Simulating execution...")
            while not plan.is_complete():
                step = plan.next_pending()
                if not step: break
                step.status = StepStatus.RUNNING
                result = simulate_execute_step(step, plan)
                step.result = result
                step.status = StepStatus.DONE
                print(f"    ✓ {step.index}. {result}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Simulate step execution")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock: mock_demo()
    else: real_demo(execute=args.execute)
