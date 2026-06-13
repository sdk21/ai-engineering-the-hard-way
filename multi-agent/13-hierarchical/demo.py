"""
Demo: Hierarchical Multi-Agent
Usage:
    python demo.py --mock
    python demo.py --real [--goal 1|2|3|4]
"""

import argparse
import os
import sys
import concurrent.futures

from experiment import (
    EXAMPLE_GOALS, HierarchyNode, HierarchicalSession,
    EXECUTIVE_SYSTEM, MANAGER_SYSTEM_TEMPLATE, WORKER_SYSTEM_TEMPLATE, REPORTER_SYSTEM,
    manager_prompt, worker_prompt, reporter_prompt,
    parse_executive_plan, parse_manager_plan,
    mock_hierarchy,
)


def mock_demo() -> None:
    print("\n=== Hierarchical Multi-Agent Demo [MOCK] ===")
    session = mock_hierarchy()
    session.display()


def real_demo(goal_idx: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    goal = EXAMPLE_GOALS[goal_idx - 1]
    session = HierarchicalSession(goal=goal)

    print(f"\n=== Hierarchical Multi-Agent Demo [REAL] ===")
    print(f"  Goal: {goal}")

    # Level 0: Executive decomposes goal
    print("\n  [CTO] Decomposing goal into team objectives...")
    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=EXECUTIVE_SYSTEM,
        messages=[{"role": "user", "content": f"Goal: {goal}"}],
    )
    try:
        plan = parse_executive_plan(r.content[0].text)
    except Exception as e:
        print(f"  Failed to parse executive plan: {e}"); return

    domains = [
        ("backend", plan.get("backend_objective", "Handle backend implementation")),
        ("frontend", plan.get("frontend_objective", "Handle frontend implementation")),
        ("qa", plan.get("qa_objective", "Handle testing and quality")),
    ]

    print(f"  Delegated to {len(domains)} teams")

    team_results = {}
    manager_nodes = []

    def run_team(domain_info):
        domain, objective = domain_info
        print(f"\n  [{domain.upper()} LEAD] Decomposing: {objective[:60]}...")

        r2 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=MANAGER_SYSTEM_TEMPLATE.format(domain=domain),
            messages=[{"role": "user", "content": manager_prompt(domain, objective)}],
        )
        try:
            w1_task, w2_task = parse_manager_plan(r2.content[0].text)
        except Exception:
            w1_task = f"Implement {domain} component 1"
            w2_task = f"Implement {domain} component 2"

        # Workers run in parallel within each team
        worker_configs = [
            (f"{domain}-worker-1", w1_task),
            (f"{domain}-worker-2", w2_task),
        ]

        def run_worker(wc):
            wid, wtask = wc
            r3 = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=WORKER_SYSTEM_TEMPLATE.format(specialty=f"{domain} specialist"),
                messages=[{"role": "user", "content": worker_prompt(wtask)}],
            )
            return wid, wtask, r3.content[0].text.strip()

        with concurrent.futures.ThreadPoolExecutor() as ex:
            worker_results = list(ex.map(run_worker, worker_configs))

        # Manager synthesizes worker results
        combined = " | ".join(r for _, _, r in worker_results)
        team_results[domain] = combined

        # Build node tree
        workers = [
            HierarchyNode(wid, f"{domain} Worker", 2,
                         WORKER_SYSTEM_TEMPLATE.format(specialty=f"{domain} specialist"),
                         assigned_task=wtask, result=wresult, status="done")
            for wid, wtask, wresult in worker_results
        ]
        mgr = HierarchyNode(f"{domain}-lead", f"{domain.upper()} Lead", 1,
                            MANAGER_SYSTEM_TEMPLATE.format(domain=domain),
                            assigned_task=objective, result=combined[:200],
                            status="done", children=workers)
        print(f"  [{domain.upper()} LEAD] Done")
        return mgr

    # All teams run in parallel
    print("\n  [Running all teams in parallel...]")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        manager_nodes = list(executor.map(run_team, domains))

    # Executive synthesizes
    print("\n  [CTO] Compiling executive report...")
    r4 = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=REPORTER_SYSTEM,
        messages=[{"role": "user", "content": reporter_prompt(goal, team_results)}],
    )
    session.final_report = r4.content[0].text.strip()

    executive = HierarchyNode("executive", "CTO (Executive)", 0,
                              EXECUTIVE_SYSTEM, assigned_task=goal,
                              result=session.final_report[:200],
                              status="done", children=manager_nodes)
    session.executive = executive
    session.display()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--goal", type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(goal_idx=args.goal)
