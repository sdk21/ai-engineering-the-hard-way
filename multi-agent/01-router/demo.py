"""
Demo: Agent Router
Usage:
    python demo.py --mock
    python demo.py --real [--strategy rule|llm]
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_MESSAGES, Route, AGENTS, RoutingResult,
    ROUTER_SYSTEM, llm_router_prompt, parse_routing_result,
    rule_based_route, mock_route_and_respond,
)


def mock_demo() -> None:
    print("\n=== Agent Router Demo [MOCK] ===\n")
    for msg in EXAMPLE_MESSAGES:
        result = mock_route_and_respond(msg)
        print(f"  Message: {msg}")
        print(f"  → Routed to: {result.agent_name} [{result.routing_result.confidence} confidence]")
        print(f"  → Response: {result.response[:100]}...")
        print()


def real_demo(strategy: str) -> None:
    import anthropic
    import json, re
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n=== Agent Router Demo [REAL, strategy={strategy}] ===")
    print("Example messages:")
    for i, m in enumerate(EXAMPLE_MESSAGES, 1):
        print(f"  {i}. {m}")
    print("\nEnter a message (or number 1-4) or 'quit':\n")

    while True:
        try:
            inp = input("Message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit":
            break
        message = EXAMPLE_MESSAGES[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        # Route
        if strategy == "rule":
            routing = rule_based_route(message)
        else:
            r = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=ROUTER_SYSTEM,
                messages=[{"role": "user", "content": llm_router_prompt(message)}],
            )
            try:
                routing = parse_routing_result(r.content[0].text)
            except Exception:
                routing = rule_based_route(message)
                routing.strategy = "rule (fallback)"

        agent_config = AGENTS[routing.route]
        print(f"\n  Routed to: {agent_config['name']} [{routing.confidence} confidence]")
        if routing.reasoning:
            print(f"  Reason: {routing.reasoning}")

        # Dispatch to subagent
        r2 = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=agent_config["system"],
            messages=[{"role": "user", "content": message}],
        )
        print(f"\n  [{agent_config['name']}]: {r2.content[0].text.strip()}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--strategy", choices=["rule", "llm"], default="llm")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(strategy=args.strategy)
