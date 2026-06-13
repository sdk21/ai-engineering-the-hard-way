"""
Demo: Hierarchical Planning
Usage:
    python demo.py --mock
    python demo.py --real [--depth 1|2]
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_GOALS, PlanNode, HierarchicalPlan, mock_product_launch_plan,
    SUBGOAL_SYSTEM, TASK_SYSTEM, parse_subgoals, parse_tasks,
)


def mock_demo() -> None:
    print("\n=== Hierarchical Planning Demo [MOCK] ===")
    plan = mock_product_launch_plan()
    plan.display()
    leaves = plan.leaf_tasks()
    print(f"\n  Total nodes: {plan.total_nodes()}")
    print(f"  Executable leaf tasks: {len(leaves)}")
    print(f"  Leaves: {', '.join(l.id for l in leaves)}")


def real_demo(depth: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Hierarchical Planning Demo [REAL, depth={depth}] ===")
    print("Example goals:")
    for i, g in enumerate(EXAMPLE_GOALS, 1):
        print(f"  {i}. {g}")
    print("\nEnter a goal (or number) or 'quit':\n")

    while True:
        try:
            inp = input("Goal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit": break
        goal = EXAMPLE_GOALS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        root = PlanNode("goal", goal, level=0)

        # Level 1: sub-goals
        r1 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=1024,
            system=SUBGOAL_SYSTEM,
            messages=[{"role": "user", "content": f"Goal: {goal}"}],
        )
        subgoals = parse_subgoals(root, r1.content[0].text)
        for sg in subgoals:
            root.add_child(sg)
        print(f"\n  Generated {len(subgoals)} sub-goals.")

        if depth >= 2:
            # Level 2: tasks for each sub-goal
            for sg in subgoals:
                r2 = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=1024,
                    system=TASK_SYSTEM,
                    messages=[{"role": "user", "content": f"Sub-goal: {sg.description}\n(Part of goal: {goal})"}],
                )
                tasks = parse_tasks(sg, r2.content[0].text)
                for t in tasks:
                    sg.add_child(t)

        plan = HierarchicalPlan(root=root)
        plan.display()
        print(f"\n  Total nodes: {plan.total_nodes()}")
        print(f"  Leaf tasks: {len(plan.leaf_tasks())}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--depth", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock: mock_demo()
    else: real_demo(depth=args.depth)
