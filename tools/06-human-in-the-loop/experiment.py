"""
Human-in-the-Loop Tool Use
--------------------------
Some tool calls should not execute automatically — they have side effects,
are irreversible, or carry risk. This experiment intercepts tool calls
before execution and routes them through a human approval gate.

Key concepts:
- Risk classification: read-only vs. write vs. destructive
- Approval policies: auto-approve safe tools, require confirmation for risky ones
- Dry-run mode: simulate execution without side effects for preview
- Audit logging: record all tool calls with approval status
- Rejection handling: what the model does when a tool call is denied

Three approval modes:
  AUTO   — executes immediately (read-only tools)
  CONFIRM — prints a preview and waits for human y/n
  BLOCK  — always denied with explanation (used for dangerous tools)

Simulated tools with varying risk:
  read_file(path)               → AUTO   (safe, reversible)
  list_directory(path)          → AUTO   (safe, reversible)
  write_file(path, content)     → CONFIRM (write, reversible with backup)
  send_email(to, subject, body) → CONFIRM (side effect, not reversible)
  delete_file(path)             → BLOCK  (destructive)
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

class ApprovalPolicy(Enum):
    AUTO = "auto"          # Execute without asking
    CONFIRM = "confirm"    # Ask human before executing
    BLOCK = "block"        # Never execute, explain why


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str = ""


@dataclass
class ApprovalDecision:
    approved: bool
    policy: ApprovalPolicy
    reason: str = ""
    modified_args: dict = field(default_factory=dict)  # optional arg overrides


@dataclass
class AuditEntry:
    timestamp: float
    tool_name: str
    arguments: dict
    policy: ApprovalPolicy
    approved: bool
    result: str = ""
    denial_reason: str = ""


# ---------------------------------------------------------------------------
# Approval policies per tool
# ---------------------------------------------------------------------------

TOOL_POLICIES: dict[str, ApprovalPolicy] = {
    "read_file": ApprovalPolicy.AUTO,
    "list_directory": ApprovalPolicy.AUTO,
    "search_files": ApprovalPolicy.AUTO,
    "write_file": ApprovalPolicy.CONFIRM,
    "send_email": ApprovalPolicy.CONFIRM,
    "delete_file": ApprovalPolicy.BLOCK,
    "execute_command": ApprovalPolicy.BLOCK,
}


# ---------------------------------------------------------------------------
# Fake file system for the demo
# ---------------------------------------------------------------------------

FAKE_FS: dict[str, str] = {
    "/home/user/notes.txt": "Meeting notes from Monday:\n- Review Q3 targets\n- Discuss hiring plan\n- Schedule retrospective",
    "/home/user/config.json": '{"debug": false, "theme": "dark", "notifications": true}',
    "/home/user/passwords.txt": "[REDACTED - this file should not be readable by AI]",
}

FAKE_DIRS: dict[str, list[str]] = {
    "/home/user": ["notes.txt", "config.json", "passwords.txt", "reports/"],
    "/home/user/reports": ["q1.pdf", "q2.pdf", "q3.pdf"],
}

# Simulated write store (doesn't persist — this is a demo)
_WRITTEN_FILES: dict[str, str] = {}
_SENT_EMAILS: list[dict] = []
_AUDIT_LOG: list[AuditEntry] = []


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    if path in FAKE_FS:
        return FAKE_FS[path]
    if path in _WRITTEN_FILES:
        return _WRITTEN_FILES[path]
    return f"Error: file not found: {path}"


def list_directory(path: str) -> str:
    items = FAKE_DIRS.get(path)
    if not items:
        return f"Error: directory not found: {path}"
    return f"Contents of {path}:\n" + "\n".join(f"  {item}" for item in items)


def search_files(query: str, directory: str = "/home/user") -> str:
    results = []
    for path, content in {**FAKE_FS, **_WRITTEN_FILES}.items():
        if path.startswith(directory) and query.lower() in content.lower():
            results.append(path)
    if not results:
        return f"No files matching '{query}' in {directory}"
    return f"Files matching '{query}':\n" + "\n".join(f"  {r}" for r in results)


def write_file(path: str, content: str) -> str:
    _WRITTEN_FILES[path] = content
    return f"File written: {path} ({len(content)} bytes)"


def send_email(to: str, subject: str, body: str) -> str:
    _SENT_EMAILS.append({"to": to, "subject": subject, "body": body, "sent_at": time.time()})
    return f"Email sent to {to}: '{subject}'"


def delete_file(path: str) -> str:
    # This should never be called — BLOCK policy prevents it
    return f"DANGER: Deleted {path}"


def execute_command(command: str) -> str:
    # This should never be called — BLOCK policy prevents it
    return f"DANGER: Executed: {command}"


TOOL_FN: dict[str, Callable] = {
    "read_file": read_file,
    "list_directory": list_directory,
    "search_files": search_files,
    "write_file": write_file,
    "send_email": send_email,
    "delete_file": delete_file,
    "execute_command": execute_command,
}


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------

def get_approval(
    tool_call: ToolCall,
    interactive: bool = True,
    auto_approve_all: bool = False,
) -> ApprovalDecision:
    """
    Determine whether a tool call should proceed.

    In non-interactive (mock) mode, CONFIRM tools auto-approve.
    In interactive (real) mode, CONFIRM tools prompt the user.
    BLOCK tools are always denied.
    """
    policy = TOOL_POLICIES.get(tool_call.name, ApprovalPolicy.CONFIRM)

    if policy == ApprovalPolicy.BLOCK:
        return ApprovalDecision(
            approved=False,
            policy=policy,
            reason=f"Tool '{tool_call.name}' is blocked by policy — it performs destructive operations.",
        )

    if policy == ApprovalPolicy.AUTO or auto_approve_all:
        return ApprovalDecision(approved=True, policy=policy)

    # CONFIRM — ask the human
    if not interactive:
        # Non-interactive: auto-approve for demo purposes
        return ApprovalDecision(approved=True, policy=policy, reason="(auto-approved in non-interactive mode)")

    # Show preview
    print(f"\n  ┌─ APPROVAL REQUIRED ────────────────────────────────────┐")
    print(f"  │ Tool:      {tool_call.name}")
    for k, v in tool_call.arguments.items():
        v_str = str(v)[:60] + ("..." if len(str(v)) > 60 else "")
        print(f"  │ {k}: {v_str}")
    print(f"  │ Risk:      {_risk_label(tool_call.name)}")
    print(f"  └────────────────────────────────────────────────────────┘")

    while True:
        answer = input("  Approve? [y/n/e(edit)] > ").strip().lower()
        if answer in ("y", "yes"):
            return ApprovalDecision(approved=True, policy=policy)
        if answer in ("n", "no"):
            reason = input("  Reason for denial (optional): ").strip()
            return ApprovalDecision(
                approved=False,
                policy=policy,
                reason=reason or "User declined.",
            )
        # 'e' could allow editing args in a full implementation


def _risk_label(tool_name: str) -> str:
    labels = {
        "write_file": "WRITE — creates or modifies a file",
        "send_email": "EXTERNAL — sends an email that cannot be unsent",
        "delete_file": "DESTRUCTIVE — permanently deletes a file",
        "execute_command": "DESTRUCTIVE — runs arbitrary shell commands",
    }
    return labels.get(tool_name, "UNKNOWN")


# ---------------------------------------------------------------------------
# Execution with gate
# ---------------------------------------------------------------------------

def execute_tool(
    tool_call: ToolCall,
    interactive: bool = True,
    auto_approve_all: bool = False,
) -> tuple[str, bool, ApprovalDecision]:
    """
    Returns (result, was_executed, decision).
    was_executed=False means the call was blocked or denied.
    """
    decision = get_approval(tool_call, interactive=interactive, auto_approve_all=auto_approve_all)

    entry = AuditEntry(
        timestamp=time.time(),
        tool_name=tool_call.name,
        arguments=tool_call.arguments,
        policy=decision.policy,
        approved=decision.approved,
    )

    if not decision.approved:
        result = f"Tool call denied: {decision.reason}"
        entry.denial_reason = decision.reason
        _AUDIT_LOG.append(entry)
        return result, False, decision

    # Execute
    fn = TOOL_FN.get(tool_call.name)
    if fn is None:
        result = f"Unknown tool: {tool_call.name}"
    else:
        try:
            args = {**tool_call.arguments, **(decision.modified_args or {})}
            result = fn(**args)
        except Exception as e:
            result = f"Execution error: {e}"

    entry.result = result[:200]
    _AUDIT_LOG.append(entry)
    return result, True, decision


def get_audit_log() -> list[AuditEntry]:
    return list(_AUDIT_LOG)


# ---------------------------------------------------------------------------
# Tool definitions for the API
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path, e.g. '/home/user/notes.txt'"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": "List the files and subdirectories in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute directory path, e.g. '/home/user'"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": "Search for files containing a given query string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for in file contents"},
                "directory": {"type": "string", "description": "Directory to search in (default: /home/user)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Requires human approval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path to write to"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email. Requires human approval before sending. Cannot be undone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body text"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "delete_file",
        "description": "Delete a file permanently. WARNING: This is a destructive operation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path to delete"},
            },
            "required": ["path"],
        },
    },
]
