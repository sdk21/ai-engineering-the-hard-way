"""
ReAct: Reasoning + Acting
--------------------------
ReAct interleaves Thought and Action steps. The model alternates between
reasoning about what to do next and actually doing it (calling a tool).
Each observation feeds back into the next thought.

Pattern:
  Thought: I need to find X to answer this.
  Action: search("X")
  Observation: [search result]
  Thought: Now I know X. I need Y too.
  Action: lookup("Y")
  Observation: [lookup result]
  Thought: I have enough to answer.
  Final Answer: [synthesised answer]

Key concepts:
- Explicit Thought traces make reasoning inspectable
- Actions ground reasoning in real information (not hallucination)
- Observations update the model's belief state
- The loop terminates when the model decides it has enough information
- Thought traces enable debugging: you can see exactly why the model acted

Difference from basic tool use (experiment 01 in tools vertical):
  - Tool use: model calls tools when needed, reasoning is implicit
  - ReAct: Thought steps are explicit, named, and part of the output format
  - ReAct traces are machine-parseable (Thought/Action/Observation labels)
"""

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# ReAct trace structure
# ---------------------------------------------------------------------------

@dataclass
class Thought:
    content: str


@dataclass
class Action:
    tool: str
    input: str


@dataclass
class Observation:
    content: str


@dataclass
class FinalAnswer:
    content: str


ReActStep = Thought | Action | Observation | FinalAnswer


@dataclass
class ReActTrace:
    question: str
    steps: list[ReActStep] = field(default_factory=list)

    def add(self, step: ReActStep) -> None:
        self.steps.append(step)

    def to_prompt_suffix(self) -> str:
        """Render the trace as a prompt continuation."""
        lines = []
        for step in self.steps:
            if isinstance(step, Thought):
                lines.append(f"Thought: {step.content}")
            elif isinstance(step, Action):
                lines.append(f"Action: {step.tool}[{step.input}]")
            elif isinstance(step, Observation):
                lines.append(f"Observation: {step.content}")
            elif isinstance(step, FinalAnswer):
                lines.append(f"Final Answer: {step.content}")
        return "\n".join(lines)

    def display(self) -> None:
        print(f"\n  Question: {self.question}\n")
        for step in self.steps:
            if isinstance(step, Thought):
                print(f"  Thought: {step.content}")
            elif isinstance(step, Action):
                print(f"  Action: {step.tool}[{step.input}]")
            elif isinstance(step, Observation):
                print(f"  Observation: {step.content}")
            elif isinstance(step, FinalAnswer):
                print(f"\n  Final Answer: {step.content}")


# ---------------------------------------------------------------------------
# ReAct prompt builder
# ---------------------------------------------------------------------------

REACT_SYSTEM = """You are a reasoning agent. You have access to the following tools:

{tool_descriptions}

Use this EXACT format for every response:

Thought: [your reasoning about what to do next]
Action: tool_name[input]

OR, when you have enough information:

Thought: [your reasoning that you're done]
Final Answer: [your answer]

Rules:
- Always start with a Thought
- One Action per response
- Wait for the Observation before your next Thought
- Use Final Answer only when you have all the information needed
- Do not make up Observations — only use what you are given
"""

REACT_FEW_SHOT = """Example:

Question: What is the elevation of the tallest mountain, and is it above the cruising altitude of commercial aircraft?

Thought: I need to find the elevation of the tallest mountain first.
Action: search[tallest mountain elevation]
Observation: Mount Everest is the tallest mountain at 8,849 meters (29,032 feet).

Thought: Now I need to know the cruising altitude of commercial aircraft.
Action: search[commercial aircraft cruising altitude]
Observation: Commercial aircraft typically cruise at 35,000-42,000 feet (10,668-12,802 meters).

Thought: Everest is 29,032 feet. Aircraft cruise at 35,000+ feet. So aircraft cruise higher than Everest.
Final Answer: Mount Everest is 29,032 feet (8,849 meters) — below the cruising altitude of commercial aircraft which typically fly at 35,000-42,000 feet.

---

"""


def react_prompt(question: str, tool_descriptions: str, trace: ReActTrace = None) -> str:
    system = REACT_SYSTEM.format(tool_descriptions=tool_descriptions)
    prompt = f"{REACT_FEW_SHOT}Question: {question}\n\n"
    if trace and trace.steps:
        prompt += trace.to_prompt_suffix() + "\n"
    return system, prompt


# ---------------------------------------------------------------------------
# Simple tool implementations
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "mount everest height": "Mount Everest is 8,849 meters (29,032 feet) — the world's highest mountain.",
    "commercial aircraft altitude": "Commercial aircraft cruise at 35,000–42,000 feet (10,700–12,800 meters).",
    "speed of sound": "The speed of sound in air is approximately 343 m/s (1,235 km/h) at sea level.",
    "speed of light": "The speed of light is approximately 299,792,458 m/s (about 300,000 km/s).",
    "population of tokyo": "Tokyo's population is approximately 13.96 million in the city proper, 37.4 million in the greater metro area.",
    "population of new york": "New York City's population is approximately 8.3 million in the city, 20 million in the metro area.",
    "distance earth to moon": "The average distance from Earth to the Moon is 384,400 km (238,855 miles).",
    "python release year": "Python was first released in 1991 by Guido van Rossum.",
    "amazon river length": "The Amazon River is approximately 6,400 km (3,976 miles) long.",
    "nile river length": "The Nile River is approximately 6,650 km (4,130 miles) long.",
}


def search(query: str) -> str:
    query_lower = query.lower().strip()
    for key, value in KNOWLEDGE_BASE.items():
        if any(word in query_lower for word in key.split()):
            return value
    return f"No result found for '{query}'. Try rephrasing."


def calculate(expression: str) -> str:
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
    def ev(n):
        if isinstance(n, ast.Constant): return n.n
        if isinstance(n, ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(f"Unsupported: {n}")
    try:
        return str(ev(ast.parse(expression.strip(), mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


def compare(a: str, b: str) -> str:
    try:
        fa, fb = float(a.replace(",", "")), float(b.replace(",", ""))
        if fa > fb: return f"{a} is greater than {b}"
        if fa < fb: return f"{a} is less than {b}"
        return f"{a} equals {b}"
    except ValueError:
        return f"Cannot numerically compare '{a}' and '{b}'"


TOOLS = {
    "search": (search, "search[query] — look up a fact in the knowledge base"),
    "calculate": (calculate, "calculate[expression] — evaluate a math expression, e.g. calculate[6400 / 343]"),
    "compare": (compare, "compare[value_a, value_b] — compare two numeric values"),
}


def tool_descriptions() -> str:
    return "\n".join(f"  {desc}" for _, (_, desc) in TOOLS.items())


def dispatch(tool: str, input_str: str) -> str:
    entry = TOOLS.get(tool.lower().strip())
    if not entry:
        return f"Unknown tool '{tool}'. Available: {', '.join(TOOLS.keys())}"
    fn, _ = entry
    # Handle comma-separated args for compare
    if tool.lower() == "compare":
        parts = [p.strip() for p in input_str.split(",")]
        return fn(*parts) if len(parts) == 2 else fn(input_str, "")
    return fn(input_str)


# ---------------------------------------------------------------------------
# ReAct response parser
# ---------------------------------------------------------------------------

def parse_react_response(text: str) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Parse a model response into (thought, tool, tool_input, final_answer).
    Returns None for fields not present in this response.
    """
    thought = None
    tool = None
    tool_input = None
    final_answer = None

    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"Action:\s*(\w+)\[(.+?)\]", text, re.DOTALL)
    if action_match:
        tool = action_match.group(1).strip()
        tool_input = action_match.group(2).strip()

    answer_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if answer_match:
        final_answer = answer_match.group(1).strip()

    return thought, tool, tool_input, final_answer


# ---------------------------------------------------------------------------
# Sample questions for the demo
# ---------------------------------------------------------------------------

QUESTIONS = [
    "Is Mount Everest taller than the cruising altitude of commercial aircraft?",
    "How long would it take sound to travel from Earth to the Moon?",
    "Which is longer — the Amazon or the Nile? By how many kilometers?",
    "What was released first: Python or the World Wide Web?",
]
