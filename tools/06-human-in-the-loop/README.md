# Lesson: Human-in-the-Loop

**Vertical:** Tools | **Difficulty:** Intermediate | **Status:** ✅ Ready

---

## Table of Contents

1. [Why Not Just Let Agents Run?](#1-why-not-just-let-agents-run)
2. [The Approval Gate Pattern](#2-the-approval-gate-pattern)
3. [Risk Classification](#3-risk-classification)
4. [Three Approval Policies](#4-three-approval-policies)
5. [What the Model Does When Denied](#5-what-the-model-does-when-denied)
6. [Audit Logging](#6-audit-logging)
7. [Dry-Run Mode](#7-dry-run-mode)
8. [Designing the Approval UX](#8-designing-the-approval-ux)
9. [Key Principles](#9-key-principles)
10. [In the Real World](#10-in-the-real-world)
11. [Running the Experiment](#11-running-the-experiment)

---

## 1. Why Not Just Let Agents Run?

The agentic loop from experiment 01 executes tool calls automatically. This is fine for read-only operations that can't cause harm. It is not fine when tools can:

- Send messages on your behalf (email, Slack, social media)
- Modify or delete data (files, database records)
- Spend money (API calls, purchases)
- Run code (arbitrary shell commands, SQL)
- Expose sensitive information externally

The defining question: **can the action be undone?** Reads can be. Writes usually can (if you have backups). Sends and deletes often cannot.

Human-in-the-loop (HITL) is the pattern of inserting a human approval gate between the model's decision to call a tool and the actual execution of that tool. The model proposes; the human approves or denies; the tool runs only if approved.

---

## 2. The Approval Gate Pattern

The gate sits in your agentic loop, between extracting a `tool_use` block and executing the tool:

```python
# Standard agentic loop (no gate)
for block in tool_use_blocks:
    result = execute_tool(block.name, block.input)

# With approval gate
for block in tool_use_blocks:
    decision = request_approval(block)   # <-- gate
    if decision.approved:
        result = execute_tool(block.name, block.input)
    else:
        result = f"Tool call denied: {decision.reason}"
    # Return result to model either way
```

The model always gets a result — either the actual tool output or an explanation of why the call was denied. This allows the model to respond to the user appropriately.

```
User: "Send an email to boss@company.com"
Model: [tool_use: send_email(to=boss@company.com, ...)]
Gate: "CONFIRM required — show preview and ask human"
Human: denies
Model: [tool_result: "Tool call denied: User declined."]
Model: "I wasn't able to send the email — you declined the approval. Let me know if you'd like to try again."
```

---

## 3. Risk Classification

Classify every tool by risk level before building the approval policy:

| Risk Level | Characteristics | Examples |
|-----------|----------------|---------|
| Safe | Read-only, reversible, no side effects | `read_file`, `search`, `list_directory` |
| Low | Write, but reversible (with backup) | `write_file`, `create_record` |
| Medium | Side effects, not easily reversible | `send_email`, `post_message`, `charge_card` |
| High | Destructive, irreversible | `delete_file`, `drop_table`, `execute_command` |

**The key insight:** Risk is about reversibility, not just scope. Writing a file is lower risk than sending an email because you can always overwrite the file. You can't unsend an email.

---

## 4. Three Approval Policies

**AUTO** — execute immediately, no prompt:
```python
# Safe for read-only, idempotent, reversible tools
"read_file": ApprovalPolicy.AUTO
"list_directory": ApprovalPolicy.AUTO
```

**CONFIRM** — show a preview and ask the human:
```python
# Required for writes and side-effecting tools
"write_file": ApprovalPolicy.CONFIRM
"send_email": ApprovalPolicy.CONFIRM
```

The CONFIRM prompt shows the tool name and arguments, lets the human review what will happen, and accepts y/n.

**BLOCK** — always deny with an explanation:
```python
# For tools the agent should never be allowed to call autonomously
"delete_file": ApprovalPolicy.BLOCK
"execute_command": ApprovalPolicy.BLOCK
```

BLOCK is different from not including a tool at all — the tool exists and the model can attempt to call it, but the gate always denies it. This lets the model explain to the user what it tried to do and why it was stopped, rather than just failing silently.

---

## 5. What the Model Does When Denied

When a tool call is denied, you return the denial reason as a `tool_result` (typically with `is_error: True`):

```python
{
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": "Tool call denied: delete_file is blocked by policy — it performs destructive operations.",
    "is_error": True,
}
```

The model incorporates this into its response. Well-implemented denial messages lead to helpful model responses:

- *"I tried to delete the file but this operation is blocked for safety reasons. If you need to delete files, you'll need to do it directly outside the assistant."*
- *"You declined the email send. I've kept the draft — would you like to review it or send it now?"*
- *"I don't have permission to run shell commands. I can help you write the command instead."*

---

## 6. Audit Logging

Every tool call — approved or denied — should be logged. This serves three purposes:

1. **Accountability** — who authorized what, when
2. **Debugging** — reproduce the exact sequence of calls that led to a bug
3. **Policy refinement** — identify which tools are frequently denied and why

```python
@dataclass
class AuditEntry:
    timestamp: float
    tool_name: str
    arguments: dict
    policy: ApprovalPolicy
    approved: bool
    result: str = ""         # truncated result for approved calls
    denial_reason: str = ""  # reason for denied calls
```

Log before execution (not after) to capture calls that fail mid-execution.

---

## 7. Dry-Run Mode

For CONFIRM tools, showing a meaningful preview can require a dry run — executing the tool logic without the side effect:

```python
def write_file_preview(path: str, content: str) -> str:
    """Shows what write_file WOULD do, without doing it."""
    existing = read_file(path)
    if existing:
        return f"Would overwrite {path} ({len(existing)} → {len(content)} bytes)"
    return f"Would create {path} ({len(content)} bytes)"
```

Dry runs help humans make informed approval decisions. They're especially valuable for:
- File writes (show diff from current content)
- API calls (show the request that would be sent)
- Database mutations (show the SQL that would run)

---

## 8. Designing the Approval UX

For CLI approvals (this experiment), a minimal but useful prompt:
```
  ┌─ APPROVAL REQUIRED ────────────────────────────────────┐
  │ Tool:      write_file
  │ path:      /home/user/report.txt
  │ content:   Q3 Summary: Revenue up 12%...
  │ Risk:      WRITE — creates or modifies a file
  └────────────────────────────────────────────────────────┘
  Approve? [y/n/e(edit)] >
```

For production UIs, consider:
- **Inline approval in chat** — a message with approve/deny buttons
- **Email/Slack approval** — async approval for non-urgent actions
- **Mobile push notification** — for time-sensitive actions
- **Batch approval** — show multiple pending actions at once

**Timeout handling:** If the human doesn't respond in N seconds, auto-deny with a message: "Approval timed out — action not taken."

---

## 9. Key Principles

> **Principle 1 — The model proposes; the human approves.**
> The model is excellent at knowing what to do. Humans are better at judging whether to do it in this specific context. HITL separates these roles cleanly.

> **Principle 2 — Classify tools by reversibility, not just scope.**
> Reading a user's entire database is lower risk than sending one email if the read has no side effects. Risk classification should be based on what can go wrong, not how much data is involved.

> **Principle 3 — Always return a result to the model.**
> Whether a tool call is approved or denied, the model gets a tool_result. A denial result lets the model explain the situation to the user; no result leaves the model confused.

> **Principle 4 — BLOCK is better than not exposing the tool.**
> If you expose a tool, the model can tell the user it tried and was stopped. If you don't expose it, the model can't even attempt the action and may hallucinate that it succeeded.

> **Principle 5 — Log everything.**
> An agentic system that acts in the world must be auditable. Every tool call, approved or not, should produce a log entry with enough context to reconstruct what happened and why.

---

## 10. In the Real World

**Claude's Computer Use**
Anthropic's computer use feature operates in a HITL loop by design for the initial release — the user watches the agent act and can interrupt at any time. Side-effecting actions (filling forms, clicking buttons) are visible to the user before they happen.

**GitHub Copilot Workspace**
Code changes proposed by Copilot are shown as diffs before being applied. The developer approves each file change. This is HITL at the code-write level.

**Zapier AI Actions**
By default, AI-triggered Zapier actions require human approval in Slack before executing. The approval message shows exactly what action will run. This is the CONFIRM pattern at the workflow level.

**AWS CloudFormation (ChangeSets)**
Before applying infrastructure changes, CloudFormation generates a ChangeSet (a preview of what will be created, modified, or deleted). An administrator approves the ChangeSet before execution. This is dry-run + CONFIRM for infrastructure.

**Google Workspace AI Features**
Gmail's "Help me draft" inserts text into the compose box but requires the user to click Send. Google Meet's auto-summary requires the user to approve before sharing. Both are HITL gates on side-effecting actions.

**Devin (Cognition AI)**
Devin, the autonomous software engineering agent, notifies the user before taking major actions (creating PRs, running deploys, modifying CI configs) and offers an interrupt mechanism throughout. Full autonomy is opt-in, not default.

---

## 11. Running the Experiment

```bash
# From the project root

# Mock mode — runs scripted scenarios showing each approval policy
uv run python tools/06-human-in-the-loop/demo.py --mock

# Real mode — interactive agent with live approval prompts
ANTHROPIC_API_KEY=sk-... uv run python tools/06-human-in-the-loop/demo.py --real
```

**Suggested queries for real mode:**
- `"List the files in /home/user and read notes.txt"` — auto-approved reads
- `"Write a brief project summary to /home/user/summary.txt"` — approval prompt for write
- `"Send an email to boss@example.com about the Q3 results"` — approval prompt for send
- `"Delete /home/user/config.json"` — blocked by policy regardless of approval
- `"Read notes.txt then write a summary back to the same directory"` — mixed auto/confirm

**Suggested exercises:**
1. Add a `DEFER` policy that queues the tool call for later review instead of blocking.
2. Implement a timeout: if the human doesn't respond to a CONFIRM prompt within 30 seconds, auto-deny.
3. Add argument editing in the approval prompt — let the human modify the email subject before approving.
4. Implement a dry-run preview for `write_file` that shows a diff against the current file content.

---

*Previous: [Tool Result Context](../05-tool-result-context/) · Next: [Dynamic Tool Selection](../07-dynamic-tool-selection/)*
