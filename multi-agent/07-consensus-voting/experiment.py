"""
Consensus Voting
-----------------
N independent agents each produce an answer. The most common answer wins
(majority vote). Weighted voting assigns higher weight to more confident agents.

Multi-agent analog of self-consistency (planning exp 02):
  Self-consistency: one model sampled N times with high temperature
  Consensus voting: N different agents (different prompts, possibly different models)

Why multiple agents instead of sampling?
  - Agents can have different specializations or perspectives
  - Agents can use different prompts engineered for different approaches
  - Agents can be from different model families (true diversity)
  - Disagreement between agents is more informative than sampling variance

Aggregation strategies:
  1. Majority vote: pick the answer that appears most often
  2. Weighted vote: each agent's vote is weighted by its stated confidence
  3. Veto: if any agent has very high confidence in a different answer, override majority
  4. Confidence-weighted: final answer = answer with highest total weighted confidence

Use cases:
  - Factual QA where individual agents may be uncertain
  - Classification tasks (sentiment, intent, category)
  - Code correctness judgments
  - Medical/legal/financial decisions requiring high confidence
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional
import json
import re


@dataclass
class AgentVote:
    agent_id: str
    agent_persona: str
    answer: str
    confidence: float          # 0.0 - 1.0
    reasoning: str = ""


@dataclass
class VotingResult:
    question: str
    votes: list[AgentVote] = field(default_factory=list)
    majority_answer: str = ""
    weighted_answer: str = ""
    consensus_strength: str = ""    # "strong", "moderate", "weak"
    vote_breakdown: dict = field(default_factory=dict)

    def display(self) -> None:
        print(f"\n  Question: {self.question}")
        print(f"\n  Individual votes:")
        for v in self.votes:
            print(f"    [{v.agent_id}] ({v.agent_persona}) → '{v.answer}' (confidence: {v.confidence:.0%})")
            if v.reasoning:
                print(f"        Reasoning: {v.reasoning[:80]}")
        print(f"\n  Majority answer: '{self.majority_answer}'")
        print(f"  Weighted answer: '{self.weighted_answer}'")
        print(f"  Consensus: {self.consensus_strength}")
        if self.vote_breakdown:
            for answer, count in sorted(self.vote_breakdown.items(), key=lambda x: -x[1]):
                print(f"    '{answer}': {count} vote(s)")


# ---------------------------------------------------------------------------
# Voter agents — different personas/approaches to the same question
# ---------------------------------------------------------------------------

VOTER_PERSONAS = [
    {
        "id": "logical",
        "persona": "logical reasoner",
        "system": """You are a logical reasoner. Answer questions by breaking them down step by step.
Follow the logic strictly. Return JSON: {"answer": "...", "confidence": 0.9, "reasoning": "brief"}""",
    },
    {
        "id": "intuitive",
        "persona": "intuitive expert",
        "system": """You are an experienced expert who uses pattern recognition and intuition.
Answer based on your deep familiarity with similar problems.
Return JSON: {"answer": "...", "confidence": 0.8, "reasoning": "brief"}""",
    },
    {
        "id": "skeptical",
        "persona": "skeptical analyst",
        "system": """You are a skeptical analyst. Question assumptions, look for edge cases.
Only give high confidence if you've considered alternative interpretations.
Return JSON: {"answer": "...", "confidence": 0.7, "reasoning": "brief"}""",
    },
    {
        "id": "conservative",
        "persona": "conservative estimator",
        "system": """You are a conservative estimator. When uncertain, lean toward the safer/more common answer.
Explicitly note uncertainty.
Return JSON: {"answer": "...", "confidence": 0.6, "reasoning": "brief"}""",
    },
    {
        "id": "creative",
        "persona": "lateral thinker",
        "system": """You are a lateral thinker who considers unconventional interpretations.
Challenge obvious answers. Still provide the most defensible answer.
Return JSON: {"answer": "...", "confidence": 0.75, "reasoning": "brief"}""",
    },
]


def voter_prompt(question: str) -> str:
    return f"Question: {question}"


def parse_vote(json_text: str, agent_id: str, persona: str) -> AgentVote:
    match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if match:
        json_text = match.group(0)
    data = json.loads(json_text)
    return AgentVote(
        agent_id=agent_id,
        agent_persona=persona,
        answer=str(data.get("answer", "")).strip(),
        confidence=float(data.get("confidence", 0.5)),
        reasoning=data.get("reasoning", ""),
    )


def majority_vote(votes: list[AgentVote]) -> tuple[str, str, dict]:
    """Returns (majority_answer, consensus_strength, vote_breakdown)."""
    answers = [v.answer.lower().strip() for v in votes]
    counts = Counter(answers)
    majority = counts.most_common(1)[0][0]
    majority_count = counts[majority]
    total = len(votes)

    if majority_count / total >= 0.8:
        strength = "strong"
    elif majority_count / total >= 0.6:
        strength = "moderate"
    else:
        strength = "weak"

    return majority, strength, dict(counts)


def weighted_vote(votes: list[AgentVote]) -> str:
    """Returns the answer with highest total weighted confidence."""
    weights: dict[str, float] = {}
    for v in votes:
        key = v.answer.lower().strip()
        weights[key] = weights.get(key, 0) + v.confidence
    return max(weights, key=weights.get)


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

MOCK_QUESTION = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"

MOCK_VOTES = [
    AgentVote("logical", "logical reasoner", "$0.05", 0.95,
              "Let ball=x, bat=x+1.00. x + x+1.00 = 1.10 → 2x = 0.10 → x = $0.05"),
    AgentVote("intuitive", "intuitive expert", "$0.10", 0.60,
              "Instinct says $0.10, but the phrasing is a classic trick question."),
    AgentVote("skeptical", "skeptical analyst", "$0.05", 0.90,
              "The intuitive answer is $0.10, but algebra confirms $0.05. The question is designed to trigger System 1."),
    AgentVote("conservative", "conservative estimator", "$0.05", 0.85,
              "Algebra: 2x + 1.00 = 1.10, x = 0.05. Conservative answer matches the calculation."),
    AgentVote("creative", "lateral thinker", "$0.05", 0.80,
              "The 'trick' is that $0.10 feels right but is wrong. Algebraic proof confirms $0.05."),
]


def mock_voting_result() -> VotingResult:
    result = VotingResult(question=MOCK_QUESTION, votes=MOCK_VOTES)
    maj, strength, breakdown = majority_vote(MOCK_VOTES)
    result.majority_answer = "$0.05"
    result.weighted_answer = "$0.05"
    result.consensus_strength = "strong"
    result.vote_breakdown = {"$0.05": 4, "$0.10": 1}
    return result


EXAMPLE_QUESTIONS = [
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "A lily pad doubles in size every day. On day 48, it covers half the pond. On what day does it cover the whole pond?",
    "Is Python or JavaScript better for building a REST API backend? (Answer: Python or JavaScript)",
]
