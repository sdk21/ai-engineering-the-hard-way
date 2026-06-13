"""
Hierarchical Multi-Agent
-------------------------
Agents organized in a hierarchy of authority:
  Level 0: Executive agent   — receives high-level goal, delegates to managers
  Level 1: Manager agents    — receive sub-goals, delegate to workers
  Level 2: Worker agents     — execute concrete tasks

Unlike the flat orchestrator (exp 02) which has one coordinator, hierarchical
systems have multiple tiers of coordination, each with different authority
and scope.

Benefits:
  - Scalability: executive doesn't need to track 50 individual tasks
  - Specialization at every level: managers specialize in their domain
  - Isolation: a failure in one manager's team doesn't affect others
  - Parallel execution across teams

This experiment implements a software development hierarchy:
  Executive: CTO agent
    └── Manager: Backend Lead
          └── Workers: API developer, Database developer
    └── Manager: Frontend Lead
          └── Workers: UI developer, State developer
    └── Manager: QA Lead
          └── Workers: Test writer, Security reviewer
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import re
import concurrent.futures


@dataclass
class HierarchyNode:
    id: str
    role: str
    level: int                          # 0=executive, 1=manager, 2=worker
    system_prompt: str
    children: list[HierarchyNode] = field(default_factory=list)
    assigned_task: str = ""
    result: str = ""
    status: str = "idle"                # idle | working | done

    def display(self, indent: int = 0) -> None:
        prefix = "  " * indent
        connector = "└─ " if indent > 0 else ""
        icon = "✓" if self.status == "done" else "○"
        task_str = f": {self.assigned_task[:50]}" if self.assigned_task else ""
        print(f"{prefix}{connector}{icon} [{self.role}]{task_str}")
        if self.result:
            print(f"{prefix}   → {self.result[:80]}")
        for child in self.children:
            child.display(indent + 1)


@dataclass
class HierarchicalSession:
    goal: str
    executive: Optional[HierarchyNode] = None
    final_report: str = ""

    def display(self) -> None:
        print(f"\n  Goal: {self.goal}")
        if self.executive:
            self.executive.display()
        if self.final_report:
            print(f"\n  Executive Report:\n  {self.final_report}")


# ---------------------------------------------------------------------------
# Agent hierarchy definitions
# ---------------------------------------------------------------------------

EXECUTIVE_SYSTEM = """You are a CTO-level executive agent. You receive a high-level software development goal
and delegate it to three team leads: backend, frontend, and qa.

Decompose the goal into team-level objectives. Return JSON:
{
  "backend_objective": "...",
  "frontend_objective": "...",
  "qa_objective": "..."
}"""

MANAGER_SYSTEM_TEMPLATE = """You are a {domain} team lead. You receive a team objective and delegate it
to two specialist workers.

Decompose the objective into two worker tasks. Return JSON:
{{
  "worker1_task": "...",
  "worker2_task": "..."
}}"""

WORKER_SYSTEM_TEMPLATE = """You are a {specialty} specialist. Complete your assigned task concisely and specifically.
Return only the result (2-4 sentences)."""

REPORTER_SYSTEM = """You are an executive summarizer. Given outputs from backend, frontend, and QA teams,
write a brief executive summary (3-4 sentences) covering what was built, how it was tested, and readiness status."""


def manager_prompt(domain: str, objective: str) -> str:
    return f"Team objective ({domain}): {objective}\n\nDecompose into two worker tasks."


def worker_prompt(task: str, context: str = "") -> str:
    return f"{task}{(chr(10) + chr(10) + 'Context: ' + context) if context else ''}"


def reporter_prompt(goal: str, team_results: dict) -> str:
    lines = [f"Project goal: {goal}", "", "Team outputs:"]
    for team, result in team_results.items():
        lines.append(f"  [{team}]: {result[:150]}")
    return "\n".join(lines)


def parse_executive_plan(json_text: str) -> dict:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    return json.loads(json_text)


def parse_manager_plan(json_text: str) -> tuple[str, str]:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    return data.get("worker1_task", ""), data.get("worker2_task", "")


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

MOCK_GOAL = "Build a user authentication feature for a REST API"


def mock_hierarchy() -> HierarchicalSession:
    session = HierarchicalSession(goal=MOCK_GOAL)

    # Workers
    api_worker = HierarchyNode("w1", "API Developer", 2,
                               WORKER_SYSTEM_TEMPLATE.format(specialty="API development"),
                               assigned_task="Implement POST /auth/login and POST /auth/register endpoints with JWT token generation",
                               result="Implemented /auth/login (validates credentials, returns JWT) and /auth/register (hashes password with bcrypt, stores user, returns 201). Both endpoints validate input with Pydantic schemas.",
                               status="done")

    db_worker = HierarchyNode("w2", "Database Developer", 2,
                              WORKER_SYSTEM_TEMPLATE.format(specialty="database"),
                              assigned_task="Design users table schema and write migration for auth tokens",
                              result="Created users table (id, email, password_hash, created_at, is_active) and auth_tokens table (id, user_id, token_hash, expires_at). Added indexes on email and token_hash for fast lookups.",
                              status="done")

    ui_worker = HierarchyNode("w3", "UI Developer", 2,
                              WORKER_SYSTEM_TEMPLATE.format(specialty="frontend UI"),
                              assigned_task="Build login form component with validation and error states",
                              result="Built LoginForm component with email/password fields, client-side validation (email format, min 8-char password), loading state during submit, and error message display for invalid credentials.",
                              status="done")

    state_worker = HierarchyNode("w4", "State Developer", 2,
                                 WORKER_SYSTEM_TEMPLATE.format(specialty="frontend state management"),
                                 assigned_task="Implement auth state management: token storage, route guards, and logout",
                                 result="Implemented useAuth hook with JWT storage in httpOnly cookies (not localStorage), route guards via ProtectedRoute component, and logout clearing cookies + redirecting to /login.",
                                 status="done")

    test_worker = HierarchyNode("w5", "Test Engineer", 2,
                                WORKER_SYSTEM_TEMPLATE.format(specialty="testing"),
                                assigned_task="Write integration tests for auth endpoints and unit tests for JWT logic",
                                result="Wrote 12 integration tests covering: successful login, invalid password, non-existent user, expired token, and registration validation. Unit tests for JWT creation, validation, and expiry. 100% coverage on auth module.",
                                status="done")

    security_worker = HierarchyNode("w6", "Security Reviewer", 2,
                                    WORKER_SYSTEM_TEMPLATE.format(specialty="security"),
                                    assigned_task="Review auth implementation for security vulnerabilities",
                                    result="Review complete. Issues found and addressed: (1) Rate limiting added to /login (5 attempts/minute), (2) Confirmed bcrypt cost factor ≥12, (3) JWT expiry set to 1 hour with refresh token pattern, (4) HTTPS-only cookie flag verified.",
                                    status="done")

    # Managers
    backend_mgr = HierarchyNode("m1", "Backend Lead", 1,
                                MANAGER_SYSTEM_TEMPLATE.format(domain="backend"),
                                assigned_task="Implement secure REST API endpoints for user authentication with JWT",
                                result="Login and register endpoints implemented with JWT, bcrypt hashing, and Pydantic validation. Database schema with proper indexes deployed.",
                                status="done",
                                children=[api_worker, db_worker])

    frontend_mgr = HierarchyNode("m2", "Frontend Lead", 1,
                                 MANAGER_SYSTEM_TEMPLATE.format(domain="frontend"),
                                 assigned_task="Build login UI with secure token handling and route protection",
                                 result="LoginForm component with validation and error states. Auth state via httpOnly cookies with route guards.",
                                 status="done",
                                 children=[ui_worker, state_worker])

    qa_mgr = HierarchyNode("m3", "QA Lead", 1,
                           MANAGER_SYSTEM_TEMPLATE.format(domain="QA"),
                           assigned_task="Ensure authentication feature is tested and secure",
                           result="12 integration tests + unit tests, 100% coverage. Security review complete: rate limiting, bcrypt ≥12, 1-hour JWT with refresh, HTTPS-only cookies.",
                           status="done",
                           children=[test_worker, security_worker])

    # Executive
    executive = HierarchyNode("e1", "CTO (Executive)", 0,
                              EXECUTIVE_SYSTEM,
                              assigned_task=MOCK_GOAL,
                              result="Authentication feature complete. Backend: JWT endpoints with bcrypt. Frontend: secure cookie-based token handling. QA: 100% test coverage, security review passed.",
                              status="done",
                              children=[backend_mgr, frontend_mgr, qa_mgr])

    session.executive = executive
    session.final_report = "The user authentication feature is complete and production-ready. Backend team delivered JWT-based login/register endpoints with bcrypt password hashing and proper database indexing. Frontend team built a validated login form with secure httpOnly cookie token storage and route guards. QA team achieved 100% test coverage and completed a security review, adding rate limiting and confirming all security best practices. Ready for deployment."
    return session


EXAMPLE_GOALS = [
    "Build a user authentication feature for a REST API",
    "Implement a real-time notification system",
    "Add CSV data import and export functionality",
    "Build a dashboard with analytics and charting",
]
