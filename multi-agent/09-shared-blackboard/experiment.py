"""
Shared Blackboard
------------------
A shared data structure (the "blackboard") that multiple agents can read from
and write to. Agents work asynchronously, each contributing their expertise
when they have something to add.

Classical AI: the Blackboard Architecture (1970s-80s) used for speech
recognition (HEARSAY) — multiple specialist knowledge sources (KS) all
reading from and writing partial solutions to a central blackboard.

Modern LLM version:
  - Blackboard is a structured dict / JSON object
  - Each agent has read access to the full blackboard
  - Each agent writes only to its designated section
  - A controller decides which agent runs next (or runs all in sequence)
  - Agents can build on each other's contributions

Key advantage over sequential pipeline:
  Sequential: A → B → C (each sees only the previous output)
  Blackboard: all agents see ALL contributions from ALL other agents
              before writing their own section

Use case here: collaborative research report
  - researcher: finds facts, writes to blackboard['facts']
  - analyst: reads facts, writes to blackboard['analysis']
  - critic: reads facts + analysis, writes to blackboard['critique']
  - synthesizer: reads everything, writes final blackboard['report']
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import re


# The blackboard is just a dict — shared, mutable, visible to all agents
Blackboard = dict[str, Any]


def display_blackboard(bb: Blackboard) -> None:
    print("\n  [Blackboard State]")
    for key, value in bb.items():
        if value:
            print(f"    [{key}]: {str(value)[:120]}")
        else:
            print(f"    [{key}]: (empty)")


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

RESEARCHER_SYSTEM = """You are a research agent contributing to a shared blackboard.
Your role: gather and write key facts about the given topic.
Other agents will read your facts to do their analysis.
Write 3-5 specific, factual points. Be precise."""

ANALYST_SYSTEM = """You are an analysis agent contributing to a shared blackboard.
Your role: analyze the facts written by the researcher and draw conclusions.
Read the 'facts' section carefully. Write 2-3 analytical insights.
Focus on patterns, implications, and significance."""

CRITIC_SYSTEM = """You are a critical review agent contributing to a shared blackboard.
Your role: read the facts AND analysis, then identify gaps, weaknesses, or counterarguments.
Be specific. Write 2-3 critical observations that strengthen the overall report."""

SYNTHESIZER_SYSTEM = """You are a synthesis agent. You have access to the full blackboard:
- facts (from researcher)
- analysis (from analyst)
- critique (from critic)

Write a coherent, balanced final report (3-5 sentences) that integrates all three sections.
Address the critique in your synthesis."""


def researcher_prompt(topic: str) -> str:
    return f"Topic: {topic}\n\nResearch and write key facts about this topic."


def analyst_prompt(topic: str, bb: Blackboard) -> str:
    return f"Topic: {topic}\n\nFacts from researcher:\n{bb.get('facts', '(none)')}\n\nWrite your analysis."


def critic_prompt(topic: str, bb: Blackboard) -> str:
    return (f"Topic: {topic}\n\n"
            f"Facts:\n{bb.get('facts', '(none)')}\n\n"
            f"Analysis:\n{bb.get('analysis', '(none)')}\n\n"
            f"Write your critical observations.")


def synthesizer_prompt(topic: str, bb: Blackboard) -> str:
    return (f"Topic: {topic}\n\n"
            f"Facts:\n{bb.get('facts', '(none)')}\n\n"
            f"Analysis:\n{bb.get('analysis', '(none)')}\n\n"
            f"Critique:\n{bb.get('critique', '(none)')}\n\n"
            f"Write the final synthesized report.")


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

MOCK_TOPIC = "The rise of vector databases in AI applications"


def mock_blackboard_session() -> tuple[str, Blackboard]:
    bb: Blackboard = {
        "topic": MOCK_TOPIC,
        "facts": (
            "1. Vector databases store high-dimensional embeddings (typically 768-1536 dimensions) rather than structured rows.\n"
            "2. Key players: Pinecone, Weaviate, Chroma, Qdrant, pgvector (PostgreSQL extension).\n"
            "3. Core operation: approximate nearest neighbor (ANN) search using algorithms like HNSW or IVF.\n"
            "4. Adoption accelerated with LLM popularity: RAG pipelines require fast similarity search.\n"
            "5. Latency benchmarks show sub-10ms search over 10M+ vectors for HNSW indexes."
        ),
        "analysis": (
            "Vector databases fill a fundamental gap: relational databases were designed for exact-match queries, "
            "not semantic similarity. The LLM wave created massive demand for 'find things that mean the same thing' "
            "rather than 'find things with the same value'. The proliferation of providers suggests the market is "
            "still consolidating — expect acquisitions and pgvector commoditizing the lower end."
        ),
        "critique": (
            "Missing context: (1) Vector databases are often overkill — for <1M vectors, in-memory libraries "
            "(FAISS, Annoy) may suffice without the operational overhead. (2) The analysis doesn't address "
            "hybrid search (vector + keyword), which is often more practical than pure vector search. "
            "(3) Data freshness is a real challenge: re-indexing at scale is expensive."
        ),
        "report": (
            "Vector databases have emerged as critical infrastructure for AI applications, particularly RAG pipelines "
            "that need fast semantic similarity search over millions of embeddings. They address a fundamental gap "
            "that relational databases weren't designed for. However, the space is still maturing: for smaller "
            "datasets, in-memory alternatives may be more practical, and hybrid search (combining vector and keyword "
            "retrieval) is often more effective than pure vector search. Expect the market to consolidate, with "
            "managed services (Pinecone) and embedded solutions (pgvector) competing at different tiers."
        ),
    }
    return MOCK_TOPIC, bb


EXAMPLE_TOPICS = [
    "The rise of vector databases in AI applications",
    "Why Rust is gaining adoption in systems programming",
    "The tradeoffs of server-side rendering vs client-side rendering",
    "How LLM context windows changed AI application architecture",
]
