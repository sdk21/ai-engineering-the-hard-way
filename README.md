# AI Engineering the Hard Way

A hands-on experiment lab for learning AI engineering from first principles — simple to frontier topics across all major verticals.

---

## Verticals

- [Memory](#memory)
- [Tools](#tools)
- [Planning](#planning)
- [Multi-Agent](#multi-agent)
- [RAG](#rag)
- [Eval](#eval)

---

## Memory

Conversation memory, episodic memory, semantic memory, working memory.

| # | Experiment | Difficulty | Status |
|---|-----------|-----------|--------|
| 01 | [Conversation Buffer](memory/01-conversation-buffer/) | Beginner | ✅ Ready |
| 02 | [Sliding Window](memory/02-sliding-window/) | Beginner–Intermediate | ✅ Ready |
| 03 | [Summarization Memory](memory/03-summarization-memory/) | Intermediate | ✅ Ready |
| 04 | [Vector Memory](memory/04-vector-memory/) | Intermediate | ✅ Ready |
| 05 | [Entity Memory](memory/05-entity-memory/) | Intermediate | ✅ Ready |

---

## Tools

Function calling, tool use, code execution, tool chaining.

| # | Experiment | Difficulty | Status |
|---|-----------|-----------|--------|
| 01 | [Basic Function Calling](tools/01-basic-function-calling/) | Beginner | ✅ Ready |

---

## Planning

Chain-of-thought, ReAct, task decomposition, self-reflection.

| # | Experiment | Difficulty | Status |
|---|-----------|-----------|--------|
| 01 | [Chain-of-Thought](planning/01-chain-of-thought/) | Beginner | 🔜 Coming Soon |

---

## Multi-Agent

Orchestration, routing, swarms, debate, parallelism.

| # | Experiment | Difficulty | Status |
|---|-----------|-----------|--------|
| 01 | [Agent Router](multi-agent/01-router/) | Beginner | 🔜 Coming Soon |

---

## RAG

Retrieval-augmented generation, chunking, reranking, hybrid search.

| # | Experiment | Difficulty | Status |
|---|-----------|-----------|--------|
| 01 | [Naive RAG](rag/01-naive-rag/) | Beginner | 🔜 Coming Soon |

---

## Eval

LLM-as-judge, benchmarks, regression testing, human eval.

| # | Experiment | Difficulty | Status |
|---|-----------|-----------|--------|
| 01 | [LLM-as-Judge](eval/01-llm-as-judge/) | Beginner | 🔜 Coming Soon |

---

## Structure

Each experiment lives in `<vertical>/<experiment-name>/` and contains:

```
<experiment-name>/
├── README.md       # What this experiment teaches and how it works
├── experiment.py   # Core implementation
└── demo.py         # Runnable demo (supports --mock and --real flags)
```

## Running Demos

Every demo supports two modes:

```bash
# Mock mode — no API calls, deterministic output, great for learning the structure
uv run python <vertical>/<experiment>/demo.py --mock

# Real mode — calls the actual LLM
ANTHROPIC_API_KEY=sk-... uv run python <vertical>/<experiment>/demo.py --real
```

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY=your_key_here
```
