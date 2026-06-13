"""
Demo: Plan-and-Execute
Usage:
    python demo.py --mock
    python demo.py --real
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_GOALS, ExecutionPlan, StepStatus,
    PLANNER_SYSTEM, EXECUTOR_SYSTEM, SYNTHESIZER_SYSTEM,
    planner_prompt, executor_prompt, synthesizer_prompt,
    parse_plan, dispatch_tool, mock_plan_and_execute,
)


def mock_demo() -> None:
    print("\n=== Plan-and-Execute Demo [MOCK] ===")
    plan = mock_plan_and_execute()
    plan.display(verbose=True)


def real_demo() -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("\n=== Plan-and-Execute Demo [REAL] ===")
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

        # Phase 1: Plan
        print("\n  [Planning...]")
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=1024,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": planner_prompt(goal)}],
        )
        try:
            plan = parse_plan(goal, r.content[0].text)
        except Exception as e:
            print(f"  Failed to parse plan: {e}\n  Raw: {r.content[0].text[:200]}")
            continue

        plan.display()

        # Phase 2: Execute
        print("\n  [Executing...]")
        for step in plan.steps:
            step.status = StepStatus.RUNNING
            step.display()

            # Use mock tools directly (no API call needed for tool steps)
            if step.tool and step.tool != "none":
                tool_result = dispatch_tool(step.tool, step.tool_input)
                step.result = tool_result
                step.status = StepStatus.DONE
                print(f"       → {tool_result[:80]}")
            else:
                # Synthesis step: call the model
                r2 = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=512,
                    system=EXECUTOR_SYSTEM,
                    messages=[{"role": "user", "content": executor_prompt(plan, step)}],
                )
                step.result = r2.content[0].text.strip()
                step.status = StepStatus.DONE
                print(f"       → {step.result[:80]}")

        # Phase 3: Synthesize final answer
        print("\n  [Synthesizing...]")
        r3 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=SYNTHESIZER_SYSTEM,
            messages=[{"role": "user", "content": synthesizer_prompt(plan)}],
        )
        plan.final_answer = r3.content[0].text.strip()
        print(f"\n  Answer: {plan.final_answer}")
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
