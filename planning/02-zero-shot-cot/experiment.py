"""
Zero-Shot Chain-of-Thought
--------------------------
Zero-shot CoT adds a reasoning trigger to the prompt without any worked
examples. The model generates reasoning steps from the trigger phrase alone.

Key concepts:
- The "Let's think step by step" trigger and why it works
- Trigger phrase variants and their relative effectiveness
- Two-stage prompting: separate reasoning call + answer extraction call
- Self-consistency: sample multiple reasoning paths, majority-vote the answer
- When zero-shot CoT matches few-shot CoT (and when it doesn't)

The two-stage pattern:
  Stage 1 → "Let's think step by step: [question]"
            Model generates: reasoning chain
  Stage 2 → "[question] [reasoning] Therefore, the answer is:"
            Model extracts: final answer only

This separation prevents the model from generating a long reasoning chain
and then giving an answer that contradicts it.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Trigger phrase library
# ---------------------------------------------------------------------------

TRIGGERS = {
    "original":    "Let's think step by step.",
    "careful":     "Let's think through this carefully, step by step.",
    "verify":      "Let's solve this step by step and verify each step.",
    "expert":      "As an expert, let me reason through this systematically.",
    "decompose":   "Let me break this problem into smaller parts.",
    "plan":        "First, let me plan my approach, then execute it step by step.",
    "none":        "",  # baseline — no trigger
}


def zero_shot_cot_stage1(question: str, trigger: str = "original") -> str:
    """Stage 1: elicit reasoning."""
    trigger_text = TRIGGERS.get(trigger, trigger)
    if trigger_text:
        return f"Q: {question}\nA: {trigger_text}"
    return f"Q: {question}\nA:"


def zero_shot_cot_stage2(question: str, reasoning: str) -> str:
    """Stage 2: extract the final answer from the reasoning."""
    return (
        f"Q: {question}\n"
        f"A: {reasoning}\n\n"
        f"Therefore, the final answer (just the answer, no explanation) is:"
    )


def zero_shot_cot_single_stage(question: str, trigger: str = "original") -> str:
    """Single-stage: ask for both reasoning and a clearly labelled answer."""
    trigger_text = TRIGGERS.get(trigger, trigger)
    suffix = trigger_text + " " if trigger_text else ""
    return (
        f"Q: {question}\n"
        f"A: {suffix}"
        f"Show your reasoning, then write 'Final Answer:' followed by just the answer."
    )


# ---------------------------------------------------------------------------
# Self-consistency sampling
# ---------------------------------------------------------------------------

@dataclass
class ReasoningPath:
    reasoning: str
    extracted_answer: str


def majority_vote(paths: list[ReasoningPath]) -> str:
    """Return the most common extracted answer across multiple reasoning paths."""
    from collections import Counter
    counts = Counter(p.extracted_answer.strip().lower() for p in paths)
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Problem bank (harder problems where zero-shot trigger matters most)
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    id: str
    question: str
    answer: str
    category: str


PROBLEMS = [
    Problem("p1", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "5 cents", "trick_arithmetic"),
    Problem("p2", "If you have a 3-gallon jug and a 5-gallon jug and need exactly 4 gallons of water, how do you do it?",
            "fill 5-gallon, pour into 3-gallon, empty 3-gallon, pour remaining 2 gallons into 3-gallon, fill 5-gallon again, pour into 3-gallon until full (1 gallon goes in), leaving 4 gallons in 5-gallon jug",
            "logic_puzzle"),
    Problem("p3", "In a tournament, every team plays every other team exactly once. With 8 teams, how many total games are played?",
            "28", "combinatorics"),
    Problem("p4", "A snail is at the bottom of a 10-foot well. Each day it climbs 3 feet, each night it slides back 2 feet. How many days to escape?",
            "8 days", "trick_sequence"),
    Problem("p5", "You have 12 balls, one is heavier. Using a balance scale, what is the minimum number of weighings needed to find it?",
            "3 weighings", "logic_puzzle"),
]


# ---------------------------------------------------------------------------
# Mock responses showing trigger effect
# ---------------------------------------------------------------------------

MOCK_RESPONSES = {
    "p1": {
        "none":     "The ball costs 10 cents.",  # wrong — classic cognitive bias
        "original": "Let's think step by step. Let ball = x. Bat = x + 1.00. Together: x + (x + 1.00) = 1.10. 2x = 0.10. x = 0.05. The ball costs 5 cents.",
    },
    "p4": {
        "none":     "10 days.",  # wrong
        "original": "Let's think step by step. After day 1: 3-2=1ft. After day 2: 2ft. ... After day 7: 7ft. On day 8 it climbs to 10ft and escapes before sliding back. Answer: 8 days.",
    },
    "p3": {
        "none":     "56 games.",  # wrong
        "original": "Let's think step by step. Each pair of teams plays once. Number of pairs from 8 teams = 8×7/2 = 28 games.",
    },
}
