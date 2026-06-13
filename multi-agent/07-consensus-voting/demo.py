"""
Demo: Consensus Voting
Usage:
    python demo.py --mock
    python demo.py --real [--voters 3|4|5]
"""

import argparse
import os
import sys
import concurrent.futures

from experiment import (
    EXAMPLE_QUESTIONS, AgentVote, VotingResult,
    VOTER_PERSONAS, voter_prompt, parse_vote,
    majority_vote, weighted_vote, mock_voting_result,
)


def mock_demo() -> None:
    print("\n=== Consensus Voting Demo [MOCK] ===")
    result = mock_voting_result()
    result.display()


def real_demo(num_voters: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    personas = VOTER_PERSONAS[:num_voters]
    print(f"\n=== Consensus Voting Demo [REAL, voters={num_voters}] ===")
    print("Example questions:")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"  {i}. {q[:70]}")
    print("\nEnter a question (or number 1-4) or 'quit':\n")

    while True:
        try:
            inp = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit":
            break
        question = EXAMPLE_QUESTIONS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        result = VotingResult(question=question)

        print(f"\n  Polling {num_voters} agents in parallel...")

        def poll_agent(persona: dict) -> AgentVote:
            r = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=persona["system"],
                messages=[{"role": "user", "content": voter_prompt(question)}],
            )
            try:
                return parse_vote(r.content[0].text, persona["id"], persona["persona"])
            except Exception:
                return AgentVote(persona["id"], persona["persona"], "unknown", 0.5)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            votes = list(executor.map(poll_agent, personas))

        result.votes = votes

        maj, strength, breakdown = majority_vote(votes)
        result.majority_answer = maj
        result.weighted_answer = weighted_vote(votes)
        result.consensus_strength = strength
        result.vote_breakdown = breakdown

        result.display()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--voters", type=int, default=5, choices=[3, 4, 5])
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(num_voters=args.voters)
