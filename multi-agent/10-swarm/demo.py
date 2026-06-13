"""
Demo: Swarm
Usage:
    python demo.py --mock
    python demo.py --real [--workers 2|3|5] [--doc 1|2]
"""

import argparse
import os
import sys
import time
import threading
import concurrent.futures
import json
import re

from experiment import (
    WorkItem, SwarmSession, EXAMPLE_DOCUMENT_CHUNKS,
    WORKER_SYSTEM, AGGREGATOR_SYSTEM,
    worker_prompt, aggregator_prompt, mock_swarm_session,
)


def mock_demo() -> None:
    print("\n=== Swarm Demo [MOCK] ===")
    session = mock_swarm_session()
    session.display()


def real_demo(num_workers: int, doc_idx: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    chunks = EXAMPLE_DOCUMENT_CHUNKS[doc_idx - 1]
    session = SwarmSession(
        task_description=f"Annotate {len(chunks)} document chunks",
        num_workers=num_workers,
    )
    for chunk_id, content in chunks:
        session.work_items.append(WorkItem(id=chunk_id, content=content))

    print(f"\n=== Swarm Demo [REAL, workers={num_workers}] ===")
    print(f"  {len(session.work_items)} chunks, {num_workers} workers processing in parallel")

    # Work queue — thread-safe using a lock
    queue = list(session.work_items)
    queue_lock = threading.Lock()

    def pick_item() -> WorkItem | None:
        with queue_lock:
            for item in queue:
                if item.status == "pending":
                    item.status = "processing"
                    return item
        return None

    def worker(worker_id: str) -> None:
        while True:
            item = pick_item()
            if item is None:
                break
            item.worker_id = worker_id
            r = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=WORKER_SYSTEM,
                messages=[{"role": "user", "content": worker_prompt(item)}],
            )
            item.annotation = r.content[0].text.strip()
            item.status = "done"
            print(f"    ✓ worker-{worker_id} annotated [{item.id}]")

    wall_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, str(i+1)) for i in range(num_workers)]
        concurrent.futures.wait(futures)
    session.wall_time_ms = (time.time() - wall_start) * 1000

    # Aggregate
    print("\n  [Aggregating...]")
    r2 = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=AGGREGATOR_SYSTEM,
        messages=[{"role": "user", "content": aggregator_prompt(session.task_description, session.work_items)}],
    )
    session.aggregated_result = r2.content[0].text.strip()
    session.display()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--workers", type=int, default=3, choices=[2, 3, 5])
    parser.add_argument("--doc", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(num_workers=args.workers, doc_idx=args.doc)
