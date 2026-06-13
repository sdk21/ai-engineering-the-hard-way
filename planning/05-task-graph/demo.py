"""
Demo: Task Graph
Usage:
    python demo.py --mock
    python demo.py --real [--execute]
"""

import argparse
import os
import sys
import concurrent.futures

from experiment import (
    EXAMPLE_GOALS, TaskStatus, mock_webapp_graph,
    GRAPH_SYSTEM, parse_task_graph,
)


def simulate_task(task) -> str:
    import time, random
    time.sleep(random.uniform(0.05, 0.15))  # simulate work
    return f"Completed in ~{task.estimated_minutes}min (simulated)"


def mock_demo() -> None:
    print("\n=== Task Graph Demo [MOCK] ===\n")
    graph = mock_webapp_graph()
    graph.display()

    cp = graph.critical_path()
    print(f"\n  Critical path: {' → '.join(cp)}")

    layers = graph.topological_layers()
    total_sequential = sum(graph.tasks[tid].estimated_minutes for layer in layers for tid in layer)
    critical_time = sum(graph.tasks[tid].estimated_minutes for tid in cp)
    print(f"  Sequential time: {total_sequential} min")
    print(f"  Parallel time (critical path): {critical_time} min")
    print(f"  Speedup potential: {total_sequential/critical_time:.1f}×")

    print("\n  Simulating parallel execution by layer:")
    for i, layer in enumerate(layers):
        names = [graph.tasks[tid].description[:30] for tid in layer]
        parallel_note = " [PARALLEL]" if len(layer) > 1 else ""
        print(f"  Layer {i+1}{parallel_note}: {', '.join(names)}")
        for tid in layer:
            graph.tasks[tid].status = TaskStatus.DONE
            graph.tasks[tid].result = "done"

    print(f"\n  Complete: {graph.is_complete()}")


def real_demo(execute: bool) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Task Graph Demo [REAL] ===")
    print("Example goals:")
    for i, g in enumerate(EXAMPLE_GOALS, 1):
        print(f"  {i}. {g}")
    print("\nEnter a goal (or number 1-4) or 'quit':\n")

    while True:
        try:
            inp = input("Goal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit": break
        goal = EXAMPLE_GOALS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=GRAPH_SYSTEM,
            messages=[{"role": "user", "content": f"Goal: {goal}"}],
        )
        try:
            graph = parse_task_graph(goal, response.content[0].text)
        except Exception as e:
            print(f"  Failed to parse plan: {e}\n  Raw: {response.content[0].text[:200]}")
            continue

        graph.display()
        cp = graph.critical_path()
        print(f"\n  Critical path: {' → '.join(cp)}")
        layers = graph.topological_layers()
        print(f"  Execution layers: {len(layers)}")

        if execute:
            print("\n  Simulating parallel execution...")
            for i, layer in enumerate(layers):
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    futures = {graph.tasks[tid]: ex.submit(simulate_task, graph.tasks[tid]) for tid in layer}
                    for task, fut in futures.items():
                        task.result = fut.result()
                        task.status = TaskStatus.DONE
                        print(f"    ✓ {task.id}: {task.result}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock: mock_demo()
    else: real_demo(execute=args.execute)
