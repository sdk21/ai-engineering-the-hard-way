"""
Demo: Zero-Shot CoT
Usage:
    python demo.py --mock
    python demo.py --real [--trigger original|careful|verify|expert|decompose|plan|none]
    python demo.py --real --self-consistency --samples 5
"""

import argparse
import os
import sys

from experiment import (
    PROBLEMS, TRIGGERS, MOCK_RESPONSES,
    zero_shot_cot_stage1, zero_shot_cot_stage2,
    zero_shot_cot_single_stage, ReasoningPath, majority_vote,
)


def mock_demo() -> None:
    print("\n=== Zero-Shot CoT Demo [MOCK] ===")
    print("Showing the effect of the trigger phrase on classic trick questions.\n")
    for pid in ["p1", "p4", "p3"]:
        p = next(x for x in PROBLEMS if x.id == pid)
        print(f"  Question: {p.question}")
        print(f"  Expected: {p.answer}")
        print()
        for trigger in ["none", "original"]:
            resp = MOCK_RESPONSES.get(pid, {}).get(trigger, f"[{trigger}] {p.answer}")
            label = "No trigger " if trigger == "none" else "With trigger"
            correct = p.answer.split()[0].lower() in resp.lower()
            print(f"  [{label}] {resp[:120]}")
            print(f"  → {'✓ correct' if correct else '✗ wrong'}")
        print("  " + "─" * 60 + "\n")


def real_demo(trigger: str, self_consistency: bool, samples: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Zero-Shot CoT Demo [REAL, trigger={trigger}] ===\n")

    for p in PROBLEMS:
        print(f"  Question: {p.question}")

        if self_consistency:
            # Sample multiple reasoning paths
            paths = []
            for i in range(samples):
                prompt1 = zero_shot_cot_stage1(p.question, trigger)
                r1 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=512,
                                            messages=[{"role": "user", "content": prompt1}])
                reasoning = r1.content[0].text
                prompt2 = zero_shot_cot_stage2(p.question, reasoning)
                r2 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=64,
                                            messages=[{"role": "user", "content": prompt2}])
                extracted = r2.content[0].text.strip()
                paths.append(ReasoningPath(reasoning=reasoning, extracted_answer=extracted))
                print(f"    [path {i+1}] answer: {extracted}")

            final = majority_vote(paths)
            correct = p.answer.lower() in final.lower()
            print(f"  Majority vote: {final} → {'✓' if correct else '✗'} (expected: {p.answer})")
        else:
            # Two-stage
            prompt1 = zero_shot_cot_stage1(p.question, trigger)
            r1 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=512,
                                        messages=[{"role": "user", "content": prompt1}])
            reasoning = r1.content[0].text
            print(f"  Reasoning: {reasoning[:200]}{'...' if len(reasoning)>200 else ''}")

            prompt2 = zero_shot_cot_stage2(p.question, reasoning)
            r2 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=64,
                                        messages=[{"role": "user", "content": prompt2}])
            answer = r2.content[0].text.strip()
            correct = p.answer.lower() in answer.lower()
            print(f"  Final answer: {answer} → {'✓' if correct else '✗'} (expected: {p.answer})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    parser.add_argument("--trigger", choices=list(TRIGGERS.keys()), default="original")
    parser.add_argument("--self-consistency", action="store_true")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)

    if args.mock:
        mock_demo()
    else:
        real_demo(args.trigger, args.self_consistency, args.samples)
