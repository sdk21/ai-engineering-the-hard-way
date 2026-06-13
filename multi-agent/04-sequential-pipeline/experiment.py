"""
Sequential Pipeline
--------------------
A chain of agents where each agent's output is the next agent's input.
Each agent transforms, enriches, or filters the data as it passes through.

Unlike the orchestrator (exp 02) which has one agent decomposing work,
a pipeline has a fixed sequence of transformation stages.

Pattern:
  Input → [Agent A] → [Agent B] → [Agent C] → Output

Real-world examples:
  - Document processing: extract → clean → classify → summarize
  - Code review: parse → lint → security scan → suggest improvements
  - Content moderation: detect language → translate → classify → flag
  - Data enrichment: parse → normalize → enrich → validate → store

Key concepts:
  - Each stage has a single, clear responsibility
  - Stages are composable and independently testable
  - Data flows in one direction (no loops in a simple pipeline)
  - A stage can be skipped (conditional routing) or retried on failure
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import re


@dataclass
class PipelineStage:
    name: str
    system: str
    description: str


@dataclass
class PipelineResult:
    stage_name: str
    input_text: str
    output_text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineRun:
    input_text: str
    stages: list[PipelineResult] = field(default_factory=list)

    def final_output(self) -> str:
        return self.stages[-1].output_text if self.stages else self.input_text

    def display(self) -> None:
        print(f"\n  Input: {self.input_text[:80]}")
        for i, stage in enumerate(self.stages):
            print(f"\n  Stage {i+1} [{stage.stage_name}]:")
            print(f"    → {stage.output_text[:120]}")
        print(f"\n  Final output: {self.final_output()[:200]}")


# ---------------------------------------------------------------------------
# Document processing pipeline
# Content: raw user feedback → structured insight
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    PipelineStage(
        name="cleaner",
        description="Clean and normalize the text",
        system="""You are a text cleaner. Fix spelling errors, normalize formatting,
remove noise (URLs, emojis unless meaningful, excessive punctuation).
Return only the cleaned text, nothing else.""",
    ),
    PipelineStage(
        name="classifier",
        description="Classify the feedback type and sentiment",
        system="""You are a feedback classifier. Analyze the text and return JSON:
{
  "type": "bug_report|feature_request|praise|complaint|question",
  "sentiment": "positive|neutral|negative",
  "urgency": "low|medium|high",
  "topic": "one word topic (e.g. 'performance', 'ui', 'billing')"
}""",
    ),
    PipelineStage(
        name="extractor",
        description="Extract key entities and action items",
        system="""You are an entity extractor. Given customer feedback, extract:
- The specific problem or request (1 sentence)
- Any features or components mentioned
- Suggested action for the product team (1 sentence)

Return JSON:
{"problem": "...", "components": ["...", "..."], "action": "..."}""",
    ),
    PipelineStage(
        name="summarizer",
        description="Write a structured summary for the product team",
        system="""You are a product feedback summarizer. Given cleaned feedback + classification + extracted entities,
write a concise structured summary for the product team. Format:

[TYPE] [URGENCY] — [TOPIC]
Problem: <one sentence>
Action: <one sentence>""",
    ),
]


def stage_prompt(stage: PipelineStage, current_text: str, all_results: list[PipelineResult]) -> str:
    if not all_results:
        return current_text
    # Pass prior stage outputs as context
    context = "\n\nPrior stage outputs:\n" + "\n".join(
        f"  [{r.stage_name}]: {r.output_text[:200]}" for r in all_results
    )
    return f"{current_text}{context}"


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

MOCK_INPUT = "The app is SOO slow when i try to export my data!!! its been like this for weeks and nobody has fixed it. I tried using chrome and safari but same thing. Please fix ASAP!!!"

MOCK_PIPELINE = [
    PipelineResult("cleaner", MOCK_INPUT,
                   "The app is very slow when I try to export my data. This has been happening for weeks and hasn't been fixed. I tried using Chrome and Safari but the issue persists. Please fix this as soon as possible."),
    PipelineResult("classifier", "...",
                   '{"type": "bug_report", "sentiment": "negative", "urgency": "high", "topic": "performance"}'),
    PipelineResult("extractor", "...",
                   '{"problem": "Data export is consistently slow across Chrome and Safari for multiple weeks.", "components": ["export", "data export", "cross-browser"], "action": "Investigate and profile the export endpoint; prioritize fix given multi-week duration."}'),
    PipelineResult("summarizer", "...",
                   "[BUG_REPORT] [HIGH] — performance\nProblem: Data export is consistently slow across Chrome and Safari, unresolved for multiple weeks.\nAction: Investigate and profile the export endpoint; prioritize fix given multi-week duration and cross-browser impact."),
]


def mock_pipeline_run() -> PipelineRun:
    run = PipelineRun(input_text=MOCK_INPUT, stages=MOCK_PIPELINE)
    return run


EXAMPLE_INPUTS = [
    "The app is SOO slow when i try to export my data!!! its been like this for weeks.",
    "Would love a dark mode option! My eyes hurt using the app at night 😢",
    "Your billing page is confusing, I accidentally upgraded to Pro instead of canceling",
    "Everything works great, the new dashboard is super intuitive. Love it!",
]
