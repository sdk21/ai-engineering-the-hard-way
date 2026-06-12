# Contributing Guidelines

This document captures all conventions and principles for this project. Follow them when adding new experiments, whether you are a human contributor or an AI agent.

---

## Project Goal

Build a hands-on AI engineering experiment lab that teaches the underlying principles and concepts of each topic — from simple to complex to frontier — across all major AI engineering verticals.

Every experiment should help someone understand **why** a technique exists, **how** it works at a mechanical level, and **where** it is used in real products.

---

## Verticals

The project is organized into verticals. Current verticals:

- `memory/` — Conversation memory, episodic, semantic, working memory
- `tools/` — Function calling, tool use, code execution, tool chaining
- `planning/` — Chain-of-thought, ReAct, task decomposition, self-reflection
- `multi-agent/` — Orchestration, routing, swarms, debate, parallelism
- `rag/` — Retrieval-augmented generation, chunking, reranking, hybrid search
- `eval/` — LLM-as-judge, benchmarks, regression testing, human eval

New verticals can be added as the project grows. Each vertical gets its own top-level directory.

---

## Experiment Numbering and Progression

Within each vertical, experiments are numbered sequentially: `01-`, `02-`, `03-`, etc.

**Experiments must progress from simple → complex → frontier.** Do not add an advanced experiment if the foundational one doesn't exist yet. The learning path should be walkable from first principles.

Example progression for `memory/`:
```
01-conversation-buffer    ← simplest possible memory
02-sliding-window         ← bounded memory
03-summarization          ← compressing old context
04-vector-memory          ← semantic retrieval of past turns
05-episodic-memory        ← structured long-term memory
```

---

## Experiment File Structure

Every experiment lives at `<vertical>/<NN>-<experiment-name>/` and must contain exactly these files:

```
<experiment-name>/
├── README.md       # Exhaustive lesson (see README format below)
├── experiment.py   # Core implementation — pure logic, no I/O
└── demo.py         # Runnable demo supporting --mock and --real
```

Do not create additional files unless strictly necessary. Do not create a per-experiment `requirements.txt` — add dependencies to the root `pyproject.toml`.

---

## `experiment.py` Conventions

- Contains the core implementation: classes, functions, data structures
- **No I/O** — no `input()`, no `print()`, no `argparse`
- Importable by `demo.py` and by tests
- Well-commented where the logic is non-obvious
- No hardcoded API keys or environment variable reads

---

## `demo.py` Conventions

Every demo **must** support two modes via CLI flags:

```bash
python demo.py --mock    # No API calls. Deterministic output. No API key needed.
python demo.py --real    # Calls the actual LLM via Anthropic API.
```

### Mock mode

- Uses a local fake/stub implementation — no network calls
- Produces deterministic, predictable output that illustrates the concept
- Should simulate realistic behavior (not just return a fixed string for everything)
- Must work with zero configuration — no environment variables required

### Real mode

- Calls Claude via the Anthropic Python SDK
- Default model: `claude-haiku-4-5-20251001` (fast and cheap for learning)
- Must check for `ANTHROPIC_API_KEY` and exit with a clear error if missing
- Should not hardcode model names as magic strings — define them as constants

### Running

```bash
# From the project root
uv run python <vertical>/<experiment>/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python <vertical>/<experiment>/demo.py --real
```

---

## README Format

Each experiment's README must be an **exhaustive, lesson-style document** — not a quick summary. Think of it as a chapter in a technical book. The goal is that someone can read the README alone and come away with a deep understanding of the topic.

### Required sections (in order)

1. **Header** — Experiment name, vertical, difficulty, status
2. **Table of Contents** — Numbered, with anchor links to every section
3. **The Problem** — What gap or limitation does this technique address? Start here, before explaining the solution.
4. **What Is X?** — Define the concept clearly and concisely
5. **How It Works — Step by Step** — Mechanical walkthrough with concrete examples, diagrams (ASCII), or code snippets
6. **Key sub-topics** — Variations, design parameters, tradeoffs (varies by experiment)
7. **Failure Modes** — A dedicated section on exactly how and why this technique breaks
8. **Key Principles** — 4–6 distilled principles, formatted as blockquotes
9. **In the Real World** — Named real products/tools with specific detail (see below)
10. **Running the Experiment** — Commands, suggested exercises
11. **Navigation** — Links to previous and next experiments

### Header format

```markdown
# Lesson: <Experiment Name>

**Vertical:** <Vertical> | **Difficulty:** <Beginner|Beginner–Intermediate|Intermediate|Advanced> | **Status:** <✅ Ready|🔜 Coming Soon>
```

### Difficulty levels

| Level | Meaning |
|-------|---------|
| Beginner | Core concept, no prior AI engineering knowledge needed |
| Beginner–Intermediate | Builds on a beginner concept, introduces one new complexity |
| Intermediate | Requires understanding of 1–2 prior experiments in the vertical |
| Advanced | Frontier technique, assumes solid grasp of the vertical |

### Status values

- `✅ Ready` — `experiment.py` and `demo.py` are fully implemented
- `🔜 Coming Soon` — README exists but code is not yet written

### "In the Real World" section

This is one of the most important sections. Requirements:

- List **6–8 named, real products or tools** (not generic "many companies do X")
- For each entry: name the product, explain specifically how it uses the concept from this experiment
- Include a mix of: open-source libraries (LangChain, LlamaIndex), commercial products (ChatGPT, Cursor), and cloud services (AWS, Azure)
- Be specific — "LangChain has a class literally named `ConversationBufferMemory`" is better than "LangChain supports memory"
- Do not include products you cannot verify actually use the technique

### Key Principles section

Format each principle as a blockquote with a bold title:

```markdown
> **Principle 1 — Name of principle.**
> One to three sentences explaining the principle and why it matters.
```

### Navigation links

End every README with navigation to adjacent experiments:

```markdown
*Previous: [Experiment Name](../NN-name/) | Next: [Experiment Name](../NN-name/)*
```

Use "coming soon" for experiments that don't exist yet:

```markdown
*Next: Summarization Memory (coming soon)*
```

---

## Dependency Management

This project uses [uv](https://github.com/astral-sh/uv).

- Add new dependencies with `uv add <package>`
- Do not manually edit `pyproject.toml` dependency lists
- Do not create per-experiment `requirements.txt` files
- Commit both `pyproject.toml` and `uv.lock`

To install and run:

```bash
uv sync
uv run python <path>/demo.py --mock
```

---

## Root README

`README.md` at the project root must stay in sync with the experiments:

- Every experiment gets a row in its vertical's table
- Columns: `#`, `Experiment` (linked to folder), `Difficulty`, `Status`
- Status must match the experiment's README header status
- When adding a new experiment, update the root README in the same commit

---

## What Not to Do

- **Do not skip difficulty levels.** If experiment 03 requires knowledge of 01 and 02, build 01 and 02 first.
- **Do not add features beyond the experiment's scope.** Each experiment teaches one concept. Scope creep confuses the learning path.
- **Do not use `eval()` or other unsafe execution patterns** in tool implementations. Always validate model-provided inputs.
- **Do not commit `.env` files, API keys, or secrets.**
- **Do not hardcode model names** as string literals scattered through the code — define them as named constants.
- **Do not summarize at the end of READMEs.** The sections speak for themselves.
- **Do not add comments to code that is self-evident.** Only comment non-obvious logic.

---

## Commit Style

Write commits that explain *why*, not just *what*:

```
# Good
Add sliding window memory with pinned summary variation

# Too vague
Update memory files

# Just describes what git diff already shows
Add experiment.py and demo.py to 02-sliding-window
```

---

## Quick Reference

| Thing | Convention |
|-------|-----------|
| Package manager | `uv` |
| Default model | `claude-haiku-4-5-20251001` |
| Demo flags | `--mock` and `--real` (mutually exclusive, one required) |
| Experiment numbering | `NN-kebab-case-name/` |
| Difficulty order | Beginner → Beginner–Intermediate → Intermediate → Advanced |
| README style | Exhaustive lesson, not a summary |
| "In the Real World" | 6–8 named real products, specific details |
