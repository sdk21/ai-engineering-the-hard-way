"""
Demo: Autonomous Task Solver (Capstone)
Usage:
    python demo.py --mock
    python demo.py --real [--verbose]
"""

import argparse
import os
import sys

from agent import SolverSession, SubGoal, SolverTask, TaskStatus, run_solver


# ---------------------------------------------------------------------------
# Mock demo
# ---------------------------------------------------------------------------

def mock_demo() -> None:
    print("\n=== Autonomous Task Solver [MOCK] ===")
    print("Goal: Design and recommend a tech stack for a real-time chat application\n")

    session = SolverSession(goal="Design and recommend a tech stack for a real-time chat application")

    # Mock hierarchical plan
    sg1 = SubGoal("sg1", "Research backend technologies")
    sg1.tasks = [
        SolverTask("t1", "Research WebSocket support in Python frameworks", "sg1",
                   tool="search", tool_input="fastapi websocket",
                   status=TaskStatus.DONE, result="FastAPI has built-in WebSocket support via Starlette. Lightweight and performant."),
        SolverTask("t2", "Research message broker options", "sg1",
                   tool="search", tool_input="redis message broker",
                   status=TaskStatus.DONE, result="Redis Pub/Sub is ideal for real-time messaging. Low latency, easy horizontal scaling."),
    ]

    sg2 = SubGoal("sg2", "Research database and storage options")
    sg2.tasks = [
        SolverTask("t3", "Research database for chat history", "sg2",
                   tool="search", tool_input="postgresql",
                   status=TaskStatus.DONE, result="PostgreSQL is a robust choice for chat history with full-text search capabilities."),
        SolverTask("t4", "Estimate storage needs", "sg2",
                   tool="calculate", tool_input="1000 * 100 * 365 * 0.5 / 1024 / 1024",
                   status=TaskStatus.DONE, result="17.39 GB/year for 1000 users × 100 messages/day × 0.5KB each"),
    ]

    sg3 = SubGoal("sg3", "Synthesize recommendation")
    sg3.tasks = [
        SolverTask("t5", "Write tech stack recommendation", "sg3",
                   status=TaskStatus.DONE,
                   result="Recommended: FastAPI (backend) + Redis Pub/Sub (messaging) + PostgreSQL (storage) + React (frontend)"),
    ]

    session.subgoals = [sg1, sg2, sg3]
    for sg in session.subgoals:
        for t in sg.tasks:
            session.all_tasks[t.id] = t

    session.draft_answer = "For a real-time chat application, I recommend: FastAPI with WebSockets for the backend, Redis Pub/Sub for real-time message delivery, PostgreSQL for persistent chat history, and React on the frontend. This stack is modern, performant, and well-supported."
    session.critique = "• Missing mention of authentication/security considerations.\n• No mention of deployment strategy."
    session.final_answer = "For a real-time chat application, I recommend:\n\n• **Backend**: FastAPI with WebSockets — async-native, lightweight, built-in WebSocket support\n• **Message broker**: Redis Pub/Sub — sub-millisecond latency for real-time delivery, easy horizontal scaling\n• **Database**: PostgreSQL — robust chat history storage with full-text search; ~17 GB/year for 1000 active users\n• **Frontend**: React with a WebSocket client library (Socket.IO or native)\n• **Auth**: JWT tokens with refresh rotation\n• **Deployment**: Docker containers on any cloud provider; Redis and PostgreSQL as managed services\n\nThis stack is production-proven, well-documented, and straightforward to deploy."

    session.display()
    print(f"\n  Techniques used: hierarchical planning, task graph, tool dispatch, self-reflection")


# ---------------------------------------------------------------------------
# Real demo
# ---------------------------------------------------------------------------

EXAMPLE_GOALS = [
    "Design and recommend a tech stack for a real-time chat application.",
    "Compare FastAPI and Flask and recommend one for a new REST API project.",
    "Explain the CAP theorem and when to choose CP vs AP systems.",
    "Outline a migration plan from a monolith to microservices.",
]


def real_demo(verbose: bool) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("\n=== Autonomous Task Solver [REAL] ===")
    print("This agent combines: hierarchical planning + task graph + plan-and-execute + self-reflection + replanning\n")
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

        print(f"\n  Solving: {goal}")
        print("  [This will make several API calls...]\n")

        try:
            session = run_solver(goal, client, verbose=verbose)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        if not verbose:
            # Show compact summary
            total_tasks = sum(len(sg.tasks) for sg in session.subgoals)
            done_tasks = sum(
                1 for sg in session.subgoals for t in sg.tasks
                if t.status == TaskStatus.DONE
            )
            print(f"  Completed {done_tasks}/{total_tasks} tasks across {len(session.subgoals)} sub-goals")
            if session.replans:
                print(f"  Replans: {session.replans}")
            if session.critique and "NO_ISSUES" not in session.critique.upper():
                print(f"  Reflection: improved after critique")
        else:
            session.display()

        print(f"\n  Final Answer:\n  {session.final_answer}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Show all planning steps")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(verbose=args.verbose)
