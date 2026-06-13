"""
Demo: ReAct
Usage:
    python demo.py --mock
    python demo.py --real
"""

import argparse
import os
import sys

from experiment import (
    QUESTIONS, TOOLS, tool_descriptions, dispatch,
    react_prompt, parse_react_response,
    ReActTrace, Thought, Action, Observation, FinalAnswer,
)

MAX_STEPS = 8


def mock_react(question: str) -> None:
    print(f"\n  Question: {question}")

    # Scripted trace for the first question
    if "everest" in question.lower():
        steps = [
            ("Thought", "I need the height of Mount Everest.", None, None),
            ("Action",  None, "search", "mount everest height"),
            ("Obs",     "Mount Everest is 8,849 meters (29,032 feet).", None, None),
            ("Thought", "Now I need the cruising altitude of commercial aircraft.", None, None),
            ("Action",  None, "search", "commercial aircraft altitude"),
            ("Obs",     "Commercial aircraft cruise at 35,000–42,000 feet.", None, None),
            ("Thought", "Everest is 29,032 ft. Aircraft cruise at 35,000+ ft. Aircraft fly higher.", None, None),
            ("Final",   "No — Mount Everest (29,032 ft) is below the cruising altitude of commercial aircraft (35,000–42,000 ft).", None, None),
        ]
    else:
        steps = [
            ("Thought", "Let me look this up.", None, None),
            ("Action", None, "search", question[:40]),
            ("Obs", "[simulated search result]", None, None),
            ("Final", f"[simulated answer to: {question[:50]}]", None, None),
        ]

    for kind, content, tool, tool_input in steps:
        if kind == "Thought": print(f"\n  Thought: {content}")
        elif kind == "Action": print(f"  Action: {tool}[{tool_input}]")
        elif kind == "Obs":    print(f"  Observation: {content}")
        elif kind == "Final":  print(f"\n  Final Answer: {content}")


def real_react(question: str) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    trace = ReActTrace(question=question)
    print(f"\n  Question: {question}")

    for step in range(MAX_STEPS):
        system, prompt = react_prompt(question, tool_descriptions(), trace)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        thought, tool, tool_input, final_answer = parse_react_response(text)

        if thought:
            print(f"\n  Thought: {thought}")
            trace.add(Thought(thought))

        if final_answer:
            print(f"\n  Final Answer: {final_answer}")
            trace.add(FinalAnswer(final_answer))
            break

        if tool and tool_input:
            print(f"  Action: {tool}[{tool_input}]")
            trace.add(Action(tool, tool_input))
            obs = dispatch(tool, tool_input)
            print(f"  Observation: {obs}")
            trace.add(Observation(obs))
        else:
            print(f"  [could not parse action — stopping]")
            break
    else:
        print(f"\n  [max steps ({MAX_STEPS}) reached]")


def run_demo(use_mock: bool) -> None:
    mode = "MOCK" if use_mock else "REAL (Claude)"
    print(f"\n=== ReAct Demo [{mode}] ===")
    print("Thought → Action → Observation loop.\n")
    print("Available tools:", list(TOOLS.keys()))
    print("\nExample questions:")
    for q in QUESTIONS:
        print(f"  • {q}")
    print("\nType a question or 'quit' to exit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not q or q.lower() == "quit": break
        if use_mock:
            mock_react(q)
        else:
            real_react(q)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    run_demo(use_mock=args.mock)
