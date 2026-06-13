"""
Demo: Backtracking
Usage:
    python demo.py --mock
    python demo.py --real [--problem 1|2] [--max-backtracks 10]
"""

import argparse
import os
import sys
import json

from experiment import (
    BacktrackSession, BacktrackState,
    PROPOSER_SYSTEM, CHECKER_SYSTEM,
    proposer_prompt, checker_prompt,
    mock_backtrack_session, EXAMPLE_PROBLEMS,
)


def mock_demo() -> None:
    print("\n=== Backtracking Demo [MOCK] ===")
    session = mock_backtrack_session()
    session.display()


def real_demo(problem_idx: int, max_backtracks: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prob = EXAMPLE_PROBLEMS[problem_idx - 1]
    session = BacktrackSession(
        problem=prob["problem"],
        variables=prob["variables"],
        constraints=prob["constraints"],
    )
    domain = prob["domain"]

    print(f"\n=== Backtracking Demo [REAL] ===")
    print(f"  Problem: {session.problem}")
    print(f"  Constraints:")
    for c in session.constraints:
        print(f"    - {c}")
    print(f"  Domain: {domain}")
    print()

    assigned = {}
    MAX_ITER = len(session.variables) * (max_backtracks + 1)

    for iteration in range(MAX_ITER):
        assigned = {var: val for var, val in session.state.assignments}
        remaining = [v for v in session.variables if v not in assigned]

        if not remaining:
            session.solved = True
            session.solution = assigned
            break

        if session.state.backtracks >= max_backtracks:
            print(f"  [Max backtracks ({max_backtracks}) reached]")
            break

        next_var = remaining[0]
        print(f"  Assigning {next_var}...")

        # Propose a value
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=256,
            system=PROPOSER_SYSTEM,
            messages=[{"role": "user", "content": proposer_prompt(session)}],
        )
        try:
            import re
            match = re.search(r'\{.*\}', r.content[0].text, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
            proposed_value = data.get("value", domain[0])
        except Exception:
            proposed_value = domain[0]

        print(f"    Proposed: {next_var} = {proposed_value!r}")

        # Check the proposed assignment
        r2 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=256,
            system=CHECKER_SYSTEM,
            messages=[{"role": "user", "content": checker_prompt(session, next_var, proposed_value)}],
        )
        try:
            match = re.search(r'\{.*\}', r2.content[0].text, re.DOTALL)
            check = json.loads(match.group(0)) if match else {}
            valid = bool(check.get("valid", True))
            violation = check.get("violation", "")
        except Exception:
            valid = True
            violation = ""

        if valid:
            session.state.assign(next_var, proposed_value)
            print(f"    ✓ Valid: {next_var} = {proposed_value!r}")
        else:
            print(f"    ✗ Violation: {violation}")
            session.state.failed_attempts.append((next_var, proposed_value, violation))
            # If we've tried too many values for this variable, backtrack
            failed_here = [v for var, v, _ in session.state.failed_attempts if var == next_var]
            if len(failed_here) >= len(domain):
                print(f"    All values exhausted for {next_var}, backtracking...")
                session.state.undo_last("all values exhausted")
                # Clear failed attempts for the variable we backtracked to
                if session.state.assignments:
                    prev_var = session.state.assignments[-1][0]
                    # Don't clear — keep failures so we don't retry them

    session.display()
    if session.solved:
        print(f"\n  Solved in {session.state.backtracks} backtrack(s)!")
    else:
        print("\n  Could not find a complete solution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--problem", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max-backtracks", type=int, default=10)
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(problem_idx=args.problem, max_backtracks=args.max_backtracks)
