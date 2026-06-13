"""
Demo: Shared Blackboard
Usage:
    python demo.py --mock
    python demo.py --real [--verbose]
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_TOPICS, Blackboard, display_blackboard,
    RESEARCHER_SYSTEM, ANALYST_SYSTEM, CRITIC_SYSTEM, SYNTHESIZER_SYSTEM,
    researcher_prompt, analyst_prompt, critic_prompt, synthesizer_prompt,
    mock_blackboard_session,
)


def mock_demo() -> None:
    print("\n=== Shared Blackboard Demo [MOCK] ===")
    topic, bb = mock_blackboard_session()
    display_blackboard(bb)
    print(f"\n  Final Report:\n  {bb.get('report', '')}")


def real_demo(verbose: bool) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("\n=== Shared Blackboard Demo [REAL] ===")
    print("Example topics:")
    for i, t in enumerate(EXAMPLE_TOPICS, 1):
        print(f"  {i}. {t}")
    print("\nEnter a topic (or number 1-4) or 'quit':\n")

    while True:
        try:
            inp = input("Topic: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit":
            break
        topic = EXAMPLE_TOPICS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        bb: Blackboard = {"topic": topic, "facts": "", "analysis": "", "critique": "", "report": ""}

        agents = [
            ("researcher", RESEARCHER_SYSTEM, researcher_prompt(topic), "facts"),
            ("analyst", ANALYST_SYSTEM, analyst_prompt(topic, bb), "analysis"),
            ("critic", CRITIC_SYSTEM, critic_prompt(topic, bb), "critique"),
            ("synthesizer", SYNTHESIZER_SYSTEM, synthesizer_prompt(topic, bb), "report"),
        ]

        for agent_name, system, _, write_key in agents:
            # Re-generate prompt with current blackboard state
            if agent_name == "researcher":
                prompt = researcher_prompt(topic)
            elif agent_name == "analyst":
                prompt = analyst_prompt(topic, bb)
            elif agent_name == "critic":
                prompt = critic_prompt(topic, bb)
            else:
                prompt = synthesizer_prompt(topic, bb)

            print(f"\n  [{agent_name}] writing to blackboard['{write_key}']...")
            r = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            bb[write_key] = r.content[0].text.strip()
            if verbose:
                print(f"    → {bb[write_key][:120]}")
            else:
                print(f"    ✓ written ({len(bb[write_key])} chars)")

        if verbose:
            display_blackboard(bb)
        print(f"\n  Final Report:\n  {bb['report']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(verbose=args.verbose)
