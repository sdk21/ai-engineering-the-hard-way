"""
Demo: Human-in-the-Loop
Usage:
    python demo.py --mock
    python demo.py --real

Try asking:
    "List the files in /home/user and read notes.txt"    (auto-approved)
    "Write a summary to /home/user/summary.txt"          (requires approval)
    "Send an email to boss@example.com about Q3 results" (requires approval)
    "Delete /home/user/config.json"                      (blocked by policy)
"""

import argparse
import os
import sys

from experiment import (
    TOOLS, ToolCall, execute_tool, get_audit_log, TOOL_POLICIES, ApprovalPolicy
)


# ---------------------------------------------------------------------------
# Mock: scripted scenarios showing the approval flow
# ---------------------------------------------------------------------------

MOCK_SCENARIOS = [
    {
        "label": "Auto-approved read (no prompt)",
        "call": ToolCall(name="read_file", arguments={"path": "/home/user/notes.txt"}, call_id="t1"),
    },
    {
        "label": "Auto-approved directory listing",
        "call": ToolCall(name="list_directory", arguments={"path": "/home/user"}, call_id="t2"),
    },
    {
        "label": "CONFIRM tool (auto-approved in mock mode)",
        "call": ToolCall(name="write_file", arguments={"path": "/home/user/memo.txt", "content": "Q4 plan"}, call_id="t3"),
    },
    {
        "label": "CONFIRM tool — send email (auto-approved in mock mode)",
        "call": ToolCall(name="send_email", arguments={"to": "alice@example.com", "subject": "Update", "body": "Here is the Q4 update."}, call_id="t4"),
    },
    {
        "label": "BLOCK tool — delete file (always denied)",
        "call": ToolCall(name="delete_file", arguments={"path": "/home/user/config.json"}, call_id="t5"),
    },
]


def mock_demo() -> None:
    print("\n=== Human-in-the-Loop Demo [MOCK] ===\n")
    print("Approval policies:")
    for tool_name, policy in TOOL_POLICIES.items():
        print(f"  {tool_name}: {policy.value}")
    print()

    for scenario in MOCK_SCENARIOS:
        label = scenario["label"]
        call = scenario["call"]
        print(f"  Scenario: {label}")
        print(f"  → {call.name}({call.arguments})")

        result, was_executed, decision = execute_tool(
            call, interactive=False, auto_approve_all=False
        )

        status = "[EXECUTED]" if was_executed else "[DENIED]"
        policy_str = f"({decision.policy.value})"
        print(f"  {status} {policy_str} {result[:80]}{'...' if len(result) > 80 else ''}")
        print()

    print("Audit log:")
    for entry in get_audit_log():
        status = "✓" if entry.approved else "✗"
        print(f"  {status} {entry.tool_name} ({entry.policy.value})")
        if entry.denial_reason:
            print(f"    reason: {entry.denial_reason}")


# ---------------------------------------------------------------------------
# Real: interactive session with live approval prompts
# ---------------------------------------------------------------------------

def real_session(user_message: str) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system = (
        "You are a helpful file management assistant. You can read files, list directories, "
        "search files, write files, and send emails. "
        "Some operations require human approval before you can proceed. "
        "If a tool call is denied, explain to the user what happened and why you couldn't complete the task. "
        "If a tool call is blocked by policy, explain that the operation is not permitted and suggest alternatives."
    )

    messages = [{"role": "user", "content": user_message}]
    print(f"\nUser: {user_message}")

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            for block in assistant_content:
                if hasattr(block, "text"):
                    print(f"Assistant: {block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    call = ToolCall(
                        name=block.name,
                        arguments=block.input,
                        call_id=block.id,
                    )
                    result, was_executed, decision = execute_tool(
                        call, interactive=True, auto_approve_all=False
                    )

                    if was_executed:
                        print(f"  [executed] {result[:80]}{'...' if len(result) > 80 else ''}")
                    else:
                        print(f"  [denied] {result}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        **({"is_error": True} if not was_executed else {}),
                    })
            messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "List the files in /home/user and read notes.txt",
    "Write a brief summary to /home/user/summary.txt",
    "Send an email to boss@example.com with a status update",
    "Delete /home/user/config.json",
]


def run_demo(use_mock: bool) -> None:
    if use_mock:
        mock_demo()
        return

    mode = "REAL (Claude)"
    print(f"\n=== Human-in-the-Loop Demo [{mode}] ===")
    print("READ and SEARCH tools auto-execute.")
    print("WRITE and SEND tools require your approval (y/n prompt).")
    print("DELETE is blocked by policy regardless of approval.\n")
    print("Example queries:")
    for q in EXAMPLE_QUERIES:
        print(f"  • {q}")
    print("\nType a request or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input or user_input.lower() == "quit":
            break
        real_session(user_input)

        # Show audit summary after each request
        log = get_audit_log()
        if log:
            last = log[-1]
            status_char = "✓" if last.approved else "✗"
            print(f"\n  [audit] {status_char} {last.tool_name} ({last.policy.value})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--real", action="store_true")
    args = parser.parse_args()

    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_demo(use_mock=args.mock)
