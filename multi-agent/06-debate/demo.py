"""
Demo: Debate
Usage:
    python demo.py --mock
    python demo.py --real [--debate 1|2|3|4] [--rounds 1|2]
"""

import argparse
import os
import sys

from experiment import (
    DEBATES, DebateSession, DebateTurn,
    PRO_SYSTEM, CON_SYSTEM, JUDGE_SYSTEM,
    pro_opening_prompt, con_rebuttal_prompt, pro_response_prompt,
    judge_prompt, parse_verdict, mock_debate,
)


def mock_demo() -> None:
    print("\n=== Debate Demo [MOCK] ===")
    session = mock_debate()
    session.display()


def real_demo(debate_idx: int, rounds: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    debate = DEBATES[debate_idx - 1]
    session = DebateSession(
        topic=debate["topic"],
        pro_position=debate["pro"],
        con_position=debate["con"],
    )

    print(f"\n=== Debate Demo [REAL, rounds={rounds}] ===")
    print(f"\n  Topic: {session.topic}")
    print(f"  PRO: {session.pro_position}")
    print(f"  CON: {session.con_position}")

    # Opening argument (PRO)
    print("\n  [PRO opening argument...]")
    r1 = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=PRO_SYSTEM,
        messages=[{"role": "user", "content": pro_opening_prompt(session.topic, session.pro_position)}],
    )
    pro_arg = r1.content[0].text.strip()
    session.turns.append(DebateTurn("pro", session.pro_position, pro_arg))
    print(f"    {pro_arg[:200]}")

    # CON rebuttal
    print("\n  [CON rebuttal...]")
    r2 = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=CON_SYSTEM,
        messages=[{"role": "user", "content": con_rebuttal_prompt(session.topic, session.con_position, pro_arg)}],
    )
    con_arg = r2.content[0].text.strip()
    session.turns.append(DebateTurn("con", session.con_position, con_arg))
    print(f"    {con_arg[:200]}")

    if rounds >= 2:
        # PRO response
        print("\n  [PRO response...]")
        r3 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=PRO_SYSTEM,
            messages=[{"role": "user", "content": pro_response_prompt(session.topic, session.pro_position, con_arg)}],
        )
        pro_resp = r3.content[0].text.strip()
        session.turns.append(DebateTurn("pro", session.pro_position, pro_resp))
        print(f"    {pro_resp[:200]}")

    # Judge
    print("\n  [Judge deliberating...]")
    r4 = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_prompt(session)}],
    )
    try:
        session.winner, session.verdict = parse_verdict(r4.content[0].text)
    except Exception:
        session.winner = "neither"
        session.verdict = r4.content[0].text.strip()

    print(f"\n  Judge's Verdict [{session.winner.upper()}]:")
    print(f"  {session.verdict}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--debate", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--rounds", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(debate_idx=args.debate, rounds=args.rounds)
