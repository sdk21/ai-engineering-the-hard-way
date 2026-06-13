"""
Demo: Parallel Fan-out
Usage:
    python demo.py --mock
    python demo.py --real [--agents 2|3|4]
"""

import argparse
import os
import sys
import time
import concurrent.futures

from experiment import (
    EXAMPLE_GOALS, FanoutTask, FanoutSession,
    PERSPECTIVE_AGENTS, AGGREGATOR_SYSTEM,
    aggregator_prompt, mock_fanout_session,
)


def mock_demo() -> None:
    print("\n=== Parallel Fan-out Demo [MOCK] ===")
    session = mock_fanout_session()
    session.display()


def real_demo(num_agents: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Parallel Fan-out Demo [REAL, agents={num_agents}] ===")
    print("Example topics:")
    for i, g in enumerate(EXAMPLE_GOALS, 1):
        print(f"  {i}. {g}")
    print("\nEnter a topic (or number 1-4) or 'quit':\n")

    while True:
        try:
            inp = input("Topic: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit":
            break
        goal = EXAMPLE_GOALS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        agents = PERSPECTIVE_AGENTS[:num_agents]
        session = FanoutSession(goal=goal)

        for agent in agents:
            session.tasks.append(FanoutTask(
                id=agent["name"],
                agent_name=agent["name"],
                instruction=f"Analyze: {goal}",
                system_prompt=agent["system"],
            ))

        print(f"\n  Launching {len(session.tasks)} agents in parallel...")
        wall_start = time.time()

        def run_task(task: FanoutTask) -> FanoutTask:
            t_start = time.time()
            r = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=task.system_prompt,
                messages=[{"role": "user", "content": f"Topic: {goal}"}],
            )
            task.result = r.content[0].text.strip()
            task.status = "done"
            task.duration_ms = (time.time() - t_start) * 1000
            return task

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(run_task, task): task for task in session.tasks}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                    print(f"    ✓ [{task.agent_name}] done ({task.duration_ms:.0f}ms)")
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    print(f"    ✗ [{task.agent_name}] failed: {e}")

        session.total_wall_time_ms = (time.time() - wall_start) * 1000
        session.sequential_estimate_ms = sum(t.duration_ms for t in session.tasks)

        # Aggregate
        print("\n  [Aggregating perspectives...]")
        r2 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=AGGREGATOR_SYSTEM,
            messages=[{"role": "user", "content": aggregator_prompt(goal, session.tasks)}],
        )
        session.aggregated_result = r2.content[0].text.strip()
        session.display()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--agents", type=int, default=4, choices=[2, 3, 4])
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(num_agents=args.agents)
