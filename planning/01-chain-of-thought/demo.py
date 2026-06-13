"""
Demo: Chain-of-Thought Prompting
Usage:
    python demo.py --mock
    python demo.py --real [--mode direct|zero-shot|few-shot|structured]
    python demo.py --real --compare   (runs all modes on same problems, shows accuracy)
"""

import argparse
import os
import sys

from experiment import (
    PROBLEMS, Problem,
    direct_prompt, zero_shot_cot_prompt, few_shot_cot_prompt, structured_cot_prompt,
    mock_response, check_answer, EvalResult,
)


# ---------------------------------------------------------------------------
# Mock demo — pre-scripted to show the direct vs CoT difference
# ---------------------------------------------------------------------------

def mock_demo() -> None:
    print("\n=== Chain-of-Thought Demo [MOCK] ===")
    print("Showing pre-scripted direct vs. CoT responses on trick questions.\n")

    showcase = ["math_02", "common_02", "common_03"]
    for pid in showcase:
        p = next(x for x in PROBLEMS if x.id == pid)
        direct = mock_response(p, "direct")
        cot = mock_response(p, "cot")

        print(f"  [{p.category}] {p.question}")
        print(f"  Ground truth: {p.answer}")
        print()
        print(f"  Direct → {direct}")
        correct_d = check_answer(direct, p.answer)
        print(f"  Correct: {'✓' if correct_d else '✗'}")
        print()
        print(f"  CoT    → {cot}")
        correct_c = check_answer(cot, p.answer)
        print(f"  Correct: {'✓' if correct_c else '✗'}")
        print("  " + "─" * 60)
        print()


# ---------------------------------------------------------------------------
# Real demo — single problem with chosen prompt mode
# ---------------------------------------------------------------------------

def build_prompt(problem: Problem, mode: str) -> str:
    return {
        "direct": direct_prompt,
        "zero-shot": zero_shot_cot_prompt,
        "few-shot": few_shot_cot_prompt,
        "structured": structured_cot_prompt,
    }[mode](problem)


def run_single(problem: Problem, mode: str, client) -> EvalResult:
    prompt = build_prompt(problem, mode)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    correct = check_answer(text, problem.answer)
    return EvalResult(
        problem_id=problem.id,
        category=problem.category,
        difficulty=problem.difficulty,
        mode=mode,
        response=text,
        correct=correct,
        ground_truth=problem.answer,
    )


def real_single(mode: str) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Chain-of-Thought Demo [REAL, mode={mode}] ===\n")

    for p in PROBLEMS:
        print(f"  [{p.difficulty}] {p.question}")
        result = run_single(p, mode, client)
        print(f"  Response: {result.response[:200]}{'...' if len(result.response)>200 else ''}")
        print(f"  Expected: {p.answer} → {'✓' if result.correct else '✗'}")
        print()


def real_compare() -> None:
    """Run all four modes on all problems and show accuracy table."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    modes = ["direct", "zero-shot", "few-shot", "structured"]
    results: dict[str, list[EvalResult]] = {m: [] for m in modes}

    print(f"\n=== Chain-of-Thought Comparison [REAL] ===")
    print(f"Running {len(PROBLEMS)} problems × {len(modes)} modes...\n")

    for p in PROBLEMS:
        for mode in modes:
            r = run_single(p, mode, client)
            results[mode].append(r)
            print(f"  [{p.id}] [{mode}] {'✓' if r.correct else '✗'}")

    print("\n── Accuracy by mode ──────────────────────────────────────")
    for mode in modes:
        rs = results[mode]
        acc = sum(r.correct for r in rs) / len(rs)
        print(f"  {mode:12s}  {acc:.0%}  ({sum(r.correct for r in rs)}/{len(rs)})")

    print("\n── Accuracy by category (CoT zero-shot) ──────────────────")
    zs = results["zero-shot"]
    for cat in ["arithmetic", "logic", "commonsense"]:
        cat_rs = [r for r in zs if r.category == cat]
        if cat_rs:
            acc = sum(r.correct for r in cat_rs) / len(cat_rs)
            print(f"  {cat:14s}  {acc:.0%}  ({sum(r.correct for r in cat_rs)}/{len(cat_rs)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(use_mock: bool, mode: str = "zero-shot", compare: bool = False) -> None:
    if use_mock:
        mock_demo()
        return
    if compare:
        real_compare()
    else:
        real_single(mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    parser.add_argument("--mode", choices=["direct", "zero-shot", "few-shot", "structured"],
                        default="zero-shot")
    parser.add_argument("--compare", action="store_true",
                        help="Run all modes and compare accuracy")
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock, mode=args.mode, compare=args.compare)
