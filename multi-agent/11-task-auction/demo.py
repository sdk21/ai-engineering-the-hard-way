"""
Demo: Task Auction
Usage:
    python demo.py --mock
    python demo.py --real
"""

import argparse
import os
import sys
import concurrent.futures

from experiment import (
    AuctionSession, AGENTS, MOCK_TASKS,
    BIDDER_SYSTEM, EXECUTOR_SYSTEM,
    bidder_prompt, executor_prompt, parse_bid, award_tasks, mock_auction,
)


def mock_demo() -> None:
    print("\n=== Task Auction Demo [MOCK] ===")
    session = mock_auction()
    session.display()


def real_demo() -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    tasks = [
        type(MOCK_TASKS[0])(t.id, t.description, t.requirements)
        for t in MOCK_TASKS
    ]
    # Re-create fresh tasks
    from experiment import Task
    tasks = [Task(t.id, t.description, list(t.requirements)) for t in MOCK_TASKS]

    session = AuctionSession(tasks=tasks, agents=AGENTS)

    print("\n=== Task Auction Demo [REAL] ===")
    print(f"\n  Tasks:")
    for t in tasks:
        print(f"    [{t.id}] {t.description}")
    print(f"\n  Agents:")
    for a in AGENTS:
        print(f"    [{a.id}] specialties: {', '.join(a.specialties[:3])}")

    # Bidding phase — all agents bid on all tasks in parallel
    print(f"\n  [Bidding phase — {len(AGENTS)} agents × {len(tasks)} tasks in parallel...]")

    def get_bid(agent_task_pair):
        agent, task = agent_task_pair
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=256,
            system=BIDDER_SYSTEM,
            messages=[{"role": "user", "content": bidder_prompt(agent, task, tasks)}],
        )
        try:
            bid = parse_bid(r.content[0].text, agent.id, task.id)
        except Exception:
            from experiment import Bid
            bid = Bid(agent.id, task.id, 5.0, "default bid")
        return bid

    pairs = [(agent, task) for agent in AGENTS for task in tasks]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        bids = list(executor.map(get_bid, pairs))
    session.bids = bids

    # Award phase
    print("\n  [Awarding tasks to highest bidders...]")
    session.awards = award_tasks(tasks, bids)
    for task_id, agent_id in session.awards.items():
        task = next(t for t in tasks if t.id == task_id)
        winning_bid = next(b for b in bids if b.task_id == task_id and b.agent_id == agent_id)
        print(f"    [{task_id}] → {agent_id} (bid: {winning_bid.capability_score:.1f}/10)")

    # Execution phase
    print("\n  [Execution phase...]")
    for task in tasks:
        agent_id = session.awards.get(task.id)
        if not agent_id:
            continue
        agent = next(a for a in AGENTS if a.id == agent_id)
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=agent.system_prompt + "\n\n" + EXECUTOR_SYSTEM,
            messages=[{"role": "user", "content": executor_prompt(agent, task)}],
        )
        task.result = r.content[0].text.strip()
        task.status = "done"
        session.results[task.id] = task.result
        print(f"    ✓ [{task.id}] {task.result[:80]}")

    session.display()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo()
