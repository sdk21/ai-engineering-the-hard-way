"""
Demo: Tree of Thoughts
Usage:
    python demo.py --mock
    python demo.py --real [--strategy bfs|dfs] [--depth 3] [--breadth 3] [--threshold 6.0]
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_PROBLEMS, ThoughtNode, ThoughtTree, NodeStatus,
    THOUGHT_GEN_SYSTEM, EVALUATOR_SYSTEM, SOLUTION_CHECK_SYSTEM,
    thought_gen_prompt, evaluator_prompt, solution_check_prompt,
    parse_thoughts, parse_score, parse_solution_check,
    mock_tot_tree,
)


def mock_demo() -> None:
    print("\n=== Tree of Thoughts Demo [MOCK] ===")
    tree = mock_tot_tree()
    tree.display()
    print(f"\n  Total nodes explored: {tree.total_nodes()}")
    print(f"  Pruned branches: {tree.pruned_nodes()}")


def path_from_root(node: ThoughtNode, all_nodes: dict) -> list[str]:
    """Collect thought strings from root to this node."""
    path = []
    current = node
    while current.parent_id and current.parent_id in all_nodes:
        path.append(current.thought)
        current = all_nodes[current.parent_id]
    path.reverse()
    return path


def real_demo(strategy: str, max_depth: int, breadth: int, threshold: float) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Tree of Thoughts Demo [REAL, strategy={strategy}, depth={max_depth}, breadth={breadth}] ===")
    print("Example problems:")
    for i, p in enumerate(EXAMPLE_PROBLEMS, 1):
        print(f"  {i}. {p}")
    print("\nEnter a problem (or number 1-4) or 'quit':\n")

    while True:
        try:
            inp = input("Problem: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit":
            break
        problem = EXAMPLE_PROBLEMS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        tree = ThoughtTree(problem=problem)
        root = ThoughtNode("root", f"Problem: {problem}", depth=0, status=NodeStatus.OPEN)
        tree.root = root
        tree.add_node(root)

        node_counter = [0]

        def new_id() -> str:
            node_counter[0] += 1
            return f"n{node_counter[0]}"

        def expand(node: ThoughtNode) -> list[ThoughtNode]:
            path = path_from_root(node, tree.all_nodes)
            if node.depth > 0:
                path = path + [node.thought]
            sys_prompt, user_msg = thought_gen_prompt(problem, path, k=breadth)
            r = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=1024,
                system=sys_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            try:
                thoughts = parse_thoughts(r.content[0].text)
            except Exception:
                return []
            children = []
            for t in thoughts:
                child = ThoughtNode(new_id(), t, depth=node.depth + 1, parent_id=node.id)
                node.children.append(child)
                tree.add_node(child)
                children.append(child)
            return children

        def evaluate(node: ThoughtNode) -> float:
            path = path_from_root(node, tree.all_nodes)
            r = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=EVALUATOR_SYSTEM,
                messages=[{"role": "user", "content": evaluator_prompt(problem, path, node.thought)}],
            )
            try:
                score, _ = parse_score(r.content[0].text)
                return score
            except Exception:
                return 5.0

        def check_solution(node: ThoughtNode) -> tuple[bool, str]:
            path = path_from_root(node, tree.all_nodes) + [node.thought]
            r = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=SOLUTION_CHECK_SYSTEM,
                messages=[{"role": "user", "content": solution_check_prompt(problem, path)}],
            )
            try:
                return parse_solution_check(r.content[0].text)
            except Exception:
                return False, ""

        print("\n  [Exploring thought tree...]")

        if strategy == "bfs":
            frontier = [root]
            for depth in range(max_depth):
                if not frontier:
                    break
                next_frontier = []
                for node in frontier:
                    children = expand(node)
                    for child in children:
                        score = evaluate(child)
                        child.score = score
                        if score >= threshold:
                            child.status = NodeStatus.PROMISING
                            # Check if solution
                            is_sol, answer = check_solution(child)
                            if is_sol:
                                child.status = NodeStatus.SOLUTION
                                tree.solution = child
                                tree.solution.thought = answer or child.thought
                                break
                            next_frontier.append(child)
                        else:
                            child.status = NodeStatus.DEAD_END
                    if tree.solution:
                        break
                frontier = next_frontier
                if tree.solution:
                    break
        else:  # dfs
            stack = [root]
            while stack and not tree.solution:
                node = stack.pop()
                if node.depth >= max_depth:
                    continue
                children = expand(node)
                for child in children:
                    score = evaluate(child)
                    child.score = score
                    if score >= threshold:
                        child.status = NodeStatus.PROMISING
                        is_sol, answer = check_solution(child)
                        if is_sol:
                            child.status = NodeStatus.SOLUTION
                            tree.solution = child
                            tree.solution.thought = answer or child.thought
                            break
                        stack.append(child)
                    else:
                        child.status = NodeStatus.DEAD_END

        tree.display()
        print(f"\n  Nodes explored: {tree.total_nodes()}")
        print(f"  Pruned: {tree.pruned_nodes()}")
        if tree.solution:
            print(f"  Solution found: {tree.solution.thought}")
        else:
            print("  No definitive solution found within budget.")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--strategy", choices=["bfs", "dfs"], default="bfs")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--breadth", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=6.0)
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(strategy=args.strategy, max_depth=args.depth, breadth=args.breadth, threshold=args.threshold)
