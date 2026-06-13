"""
Demo: Research Team (Capstone)
Usage:
    python demo.py --mock
    python demo.py --real [--topic 1|2|3|4] [--hitl] [--verbose]
"""

import argparse
import os
import sys

from agent import ResearchSession, make_blackboard, run_research_team


# ---------------------------------------------------------------------------
# Mock demo
# ---------------------------------------------------------------------------

def mock_demo() -> None:
    print("\n=== Research Team Capstone [MOCK] ===")
    print("Topic: The rise of WebAssembly (Wasm) in web development\n")

    bb = make_blackboard("The rise of WebAssembly (Wasm) in web development")
    bb["sub_questions"] = [
        "How does WebAssembly work technically and how does it compare to JavaScript performance-wise?",
        "What are the real-world use cases and adoption of WebAssembly in production?",
        "What is the history of WebAssembly and why was it created?",
    ]
    bb["technical"] = "WebAssembly is a binary instruction format designed as a compilation target for high-level languages. It runs in a stack-based virtual machine inside the browser at near-native speed. WASM modules are sandboxed and communicate with JavaScript via a JS API. Benchmarks show 10-50× speedups over JavaScript for compute-intensive tasks like image processing, video encoding, and cryptography."
    bb["business"] = "Major adopters include Figma (2× performance gain on canvas rendering), Google Earth, AutoCAD Web, and Unity games. The WASM ecosystem grew significantly after 2022 with WASI (WebAssembly System Interface) enabling server-side use in edge computing (Fastly, Cloudflare Workers). Most major languages now have WASM compilation targets."
    bb["context"] = "WebAssembly was developed by a W3C community group with input from Mozilla, Google, Microsoft, and Apple, officially released in 2017. It was born from the limitations of asm.js (Mozilla, 2013) — a faster but unwieldy JavaScript subset. The goal: a universally agreed-upon binary format that all browsers could implement without vendor-specific extensions."
    bb["draft_report"] = "WebAssembly (Wasm) is a binary instruction format that runs at near-native speed inside browser VMs, designed as a compilation target for languages like C++, Rust, and Go. Technically, it's sandboxed, stack-based, and interops with JavaScript via a typed API — benchmarks show 10-50× speedups for compute-heavy workloads. In production, Figma, Google Earth, AutoCAD Web, and Unity games all rely on Wasm for performance-critical rendering. Beyond the browser, WASI (2022) extended Wasm to server-side edge computing, with Cloudflare and Fastly running Wasm workloads. Historically, Wasm emerged in 2017 from a W3C cross-vendor collaboration, succeeding asm.js and providing a vendor-neutral binary standard that all major browsers agreed to implement."
    bb["critique"] = "No significant issues"
    bb["validated_claims"] = [
        {"claim": "WebAssembly is widely used in industry", "valid": True, "confidence": 0.9},
        {"claim": "WebAssembly has both advantages and disadvantages", "valid": True, "confidence": 0.95},
    ]
    bb["final_report"] = bb["draft_report"]

    session = ResearchSession(topic="The rise of WebAssembly (Wasm) in web development", bb=bb)
    session.human_approved = True
    session.steps_log = [
        "[Orchestrator] Decomposing research topic...",
        "[Research Team] Researching in parallel...",
        "[Blackboard] Three research sections written",
        "[Drafter] Writing draft report...",
        "[Critic] Reviewing draft...",
        "[Consensus] Validating key claims...",
        "[Done] Final report ready",
    ]

    print("  Agents involved:")
    print("    Orchestrator → Technical researcher + Business researcher + Context researcher (parallel)")
    print("    → Drafter → Critic → Consensus voters → [HITL] → Final report")
    print()
    session.display()
    print(f"\n  Techniques used: orchestration, parallel fan-out, shared blackboard, critic agent,")
    print(f"  consensus voting, (HITL available with --hitl flag)")


# ---------------------------------------------------------------------------
# HITL handler for interactive mode
# ---------------------------------------------------------------------------

def interactive_hitl(draft: str) -> tuple[bool, str]:
    print(f"\n  ┌─ HITL CHECKPOINT ─────────────────────────────────────")
    print(f"  │ Draft report ready for review:")
    print(f"  │")
    for line in draft.split(". "):
        print(f"  │   {line.strip()[:80]}")
    print(f"  │")
    print(f"  │ Options: [a]pprove / [r]eject / [e]dit (provide feedback)")
    print(f"  └──────────────────────────────────────────────────────")
    while True:
        try:
            choice = input("  Decision: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False, "interrupted"
        if choice in ("a", "approve"):
            return True, ""
        elif choice in ("r", "reject"):
            return False, ""
        elif choice in ("e", "edit"):
            feedback = input("  Feedback: ").strip()
            return True, feedback
        print("  Enter 'a', 'r', or 'e'")


def auto_hitl(draft: str) -> tuple[bool, str]:
    print("  [HITL] Auto-approving...")
    return True, ""


# ---------------------------------------------------------------------------
# Real demo
# ---------------------------------------------------------------------------

EXAMPLE_TOPICS = [
    "The rise of WebAssembly (Wasm) in web development",
    "Why Rust is replacing C++ in systems programming",
    "The impact of large language models on software development workflows",
    "How edge computing is changing web application architecture",
]


def real_demo(topic_idx: int, use_hitl: bool, verbose: bool) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    topic = EXAMPLE_TOPICS[topic_idx - 1]
    hitl_handler = interactive_hitl if use_hitl else auto_hitl

    print(f"\n=== Research Team Capstone [REAL] ===")
    print(f"  Topic: {topic}")
    print(f"  Techniques: orchestration + parallel fan-out + shared blackboard + critic + consensus + {'interactive HITL' if use_hitl else 'auto HITL'}")
    print()

    session = run_research_team(topic, client, verbose=verbose, hitl_handler=hitl_handler)

    if verbose:
        session.display()
    else:
        print(f"\n  Steps completed: {len(session.steps_log)}")
        print(f"  Human approved: {session.human_approved}")
        if session.bb.get("validated_claims"):
            for vc in session.bb["validated_claims"]:
                validity = "✓" if vc["valid"] else "✗"
                print(f"  {validity} Claim validated ({vc['confidence']:.0%}): {vc['claim'][:60]}")

    if session.bb.get("final_report"):
        print(f"\n  Final Report:\n  {session.bb['final_report']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--topic", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--hitl", action="store_true", help="Enable interactive human review")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(topic_idx=args.topic, use_hitl=args.hitl, verbose=args.verbose)
