"""
Chain-of-Thought Prompting
--------------------------
Eliciting step-by-step reasoning from a model before it produces a final answer.
The intermediate reasoning tokens act as computation scaffolding.

Key concepts:
- Direct prompting vs. CoT prompting on the same problems
- Zero-shot CoT: "think step by step" trigger
- Few-shot CoT: worked examples in the prompt
- Structured CoT: XML tags, scratchpad format
- Measuring the accuracy difference

Three problem categories where CoT helps most:
  1. Multi-step arithmetic / word problems
  2. Logical deduction
  3. Commonsense reasoning with hidden dependencies
"""

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Problem bank
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    id: str
    category: str
    question: str
    answer: str          # ground truth
    difficulty: str      # easy / medium / hard


PROBLEMS = [
    # Arithmetic word problems
    Problem("math_01", "arithmetic",
        "A train travels 60 mph for 2.5 hours, then 80 mph for 1.5 hours. How many miles total?",
        "270", "easy"),
    Problem("math_02", "arithmetic",
        "Alice has 3 times as many apples as Bob. Bob has 4 more than Carol. Carol has 6. How many apples does Alice have?",
        "30", "medium"),
    Problem("math_03", "arithmetic",
        "A store marks up items 40% then offers a 20% sale discount. What is the net percentage change from original price?",
        "12% increase", "medium"),

    # Logical deduction
    Problem("logic_01", "logic",
        "All mammals are warm-blooded. Whales are mammals. Snakes are not mammals. Which of these is warm-blooded: whales, snakes, both, or neither?",
        "whales", "easy"),
    Problem("logic_02", "logic",
        "If it rains, the picnic is cancelled. If the picnic is cancelled, we go to the museum. It is raining. Do we go to the museum?",
        "yes", "easy"),
    Problem("logic_03", "logic",
        "Five people sit in a row: A, B, C, D, E. A is not next to B. C is between D and E. B is at one end. What position is A?",
        "position 4 (second from right)", "hard"),

    # Commonsense with hidden dependencies
    Problem("common_01", "commonsense",
        "I put an ice cube in a glass of warm water and sealed the glass. An hour later, is the water warmer or cooler than it started?",
        "cooler", "easy"),
    Problem("common_02", "commonsense",
        "A farmer has 17 sheep. All but 9 die. How many are left?",
        "9", "medium"),  # classic trick question
    Problem("common_03", "commonsense",
        "You're in a race and you overtake the person in second place. What place are you now in?",
        "second", "medium"),  # another classic trick
]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def direct_prompt(problem: Problem) -> str:
    """No reasoning scaffolding — model jumps straight to answer."""
    return f"""Answer the following question with just the final answer (no explanation needed):

Question: {problem.question}

Answer:"""


def zero_shot_cot_prompt(problem: Problem) -> str:
    """Zero-shot CoT: trigger phrase only, no examples."""
    return f"""Answer the following question. Think step by step before giving your final answer.

Question: {problem.question}

Let's think step by step:"""


def few_shot_cot_prompt(problem: Problem) -> str:
    """Few-shot CoT: worked examples demonstrate the reasoning format."""
    examples = """Question: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many tennis balls does he have now?
Reasoning: Roger starts with 5 balls. He buys 2 × 3 = 6 more balls. Total: 5 + 6 = 11 balls.
Answer: 11

Question: The cafeteria had 23 apples. They used 20 for lunch and bought 6 more. How many do they have?
Reasoning: Start with 23. Used 20, so 23 - 20 = 3 remain. Bought 6 more: 3 + 6 = 9 apples.
Answer: 9

"""
    return f"""{examples}Question: {problem.question}
Reasoning:"""


def structured_cot_prompt(problem: Problem) -> str:
    """Structured CoT: XML tags separate reasoning from answer."""
    return f"""Answer the following question. Put your reasoning inside <thinking> tags, then give your final answer.

Question: {problem.question}

<thinking>"""


# ---------------------------------------------------------------------------
# Mock CoT simulation (for --mock mode)
# ---------------------------------------------------------------------------

MOCK_COT_RESPONSES = {
    "math_01": {
        "direct": "150 miles.",
        "cot": "First leg: 60 mph × 2.5 hours = 150 miles. Second leg: 80 mph × 1.5 hours = 120 miles. Total: 150 + 120 = 270 miles. Answer: 270 miles.",
    },
    "math_02": {
        "direct": "15 apples.",  # wrong — shows where direct fails
        "cot": "Carol has 6 apples. Bob has 6 + 4 = 10 apples. Alice has 3 × 10 = 30 apples. Answer: 30.",
    },
    "common_02": {
        "direct": "8 sheep.",  # wrong — classic trick
        "cot": "'All but 9 die' means 9 survive. The phrase 'all but 9' means everything except 9. Answer: 9.",
    },
    "common_03": {
        "direct": "first place.",  # wrong — classic trick
        "cot": "If I overtake the person in 2nd place, I take their position. I am now in 2nd place. I didn't overtake 1st. Answer: second place.",
    },
}


def mock_response(problem: Problem, mode: str) -> str:
    """Return a pre-scripted response showing CoT vs direct difference."""
    responses = MOCK_COT_RESPONSES.get(problem.id, {})
    if mode == "direct":
        return responses.get("direct", f"[direct] {problem.answer}")
    return responses.get("cot", f"[cot] Step 1: ... Step 2: ... Answer: {problem.answer}")


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def check_answer(response: str, ground_truth: str) -> bool:
    """Rough check — does the response contain the correct answer?"""
    return ground_truth.lower() in response.lower()


@dataclass
class EvalResult:
    problem_id: str
    category: str
    difficulty: str
    mode: str
    response: str
    correct: bool
    ground_truth: str
