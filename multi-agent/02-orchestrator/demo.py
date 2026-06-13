"""
Demo: Orchestrator + Subagents
Usage:
    python demo.py --mock
    python demo.py --real
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_GOALS, OrchestrationSession, Subtask, SUBAGENTS,
    ORCHESTRATOR_SYSTEM, ASSEMBLER_SYSTEM,
    orchestrator_prompt, subagent_prompt, assembler_prompt,
    parse_subtasks, mock_orchestration,
)


def mock_demo() -> None:
    print("\n=== Orchestrator + Subagents Demo [MOCK] ===")
    session = mock_orchestration()
    session.display()


def real_demo() -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("\n=== Orchestrator + Subagents Demo [REAL] ===")
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

        session = OrchestrationSession(goal=goal)

        # Orchestrate
        print("\n  [Orchestrating...]")
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=ORCHESTRATOR_SYSTEM,
            messages=[{"role": "user", "content": orchestrator_prompt(goal)}],
        )
        try:
            session.subtasks = parse_subtasks(r.content[0].text)
        except Exception as e:
            print(f"  Failed to parse subtasks: {e}"); continue

        print(f"  Decomposed into {len(session.subtasks)} subtasks")

        # Execute subtasks sequentially (passing prior results as context)
        prior_results = []
        for st in session.subtasks:
            agent_config = SUBAGENTS.get(st.agent, SUBAGENTS["researcher"])
            print(f"\n  [{st.agent}] {st.instruction[:60]}...")
            r2 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=agent_config["system"],
                messages=[{"role": "user", "content": subagent_prompt(st.instruction, prior_results)}],
            )
            st.result = r2.content[0].text.strip()
            st.status = "done"
            prior_results.append((st.agent, st.result))
            print(f"    → {st.result[:80]}")

        # Assemble final response
        print("\n  [Assembling final response...]")
        r3 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=ASSEMBLER_SYSTEM,
            messages=[{"role": "user", "content": assembler_prompt(goal, session.subtasks)}],
        )
        session.final_response = r3.content[0].text.strip()
        print(f"\n  Final Response:\n  {session.final_response}")
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
