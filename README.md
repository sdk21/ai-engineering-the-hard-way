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
| 06 | [Episodic Memory](memory/06-episodic-memory/) | Intermediate | ✅ Ready |
| 07 | [Knowledge Graph Memory](memory/07-knowledge-graph-memory/) | Intermediate–Advanced | ✅ Ready |
| 08 | [Provenance & Confidence Memory](memory/08-provenance-confidence-memory/) | Intermediate–Advanced | ✅ Ready |
| 09 | [Conflict Resolution Memory](memory/09-conflict-resolution-memory/) | Advanced | ✅ Ready |
| 10 | [Layered Memory](memory/10-layered-memory/) | Advanced | ✅ Ready |
| C | [**Capstone: Personal Assistant**](memory/capstone/personal-assistant/) | Capstone | ✅ Ready |

---

## Tools

Function calling, tool use, code execution, tool chaining.

| # | Experiment | Difficulty | Status |
|---|-----------|-----------|--------|
| 01 | [Basic Function Calling](tools/01-basic-function-calling/) | Beginner | ✅ Ready |
| 02 | [Tool Chaining](tools/02-tool-chaining/) | Beginner–Intermediate | ✅ Ready |
| 03 | [Parallel Tool Calls](tools/03-parallel-tool-calls/) | Beginner–Intermediate | ✅ Ready |
| 04 | [Tool Error Handling](tools/04-tool-error-handling/) | Intermediate | ✅ Ready |
| 05 | [Tool Result Context](tools/05-tool-result-context/) | Intermediate | ✅ Ready |
| 06 | [Human-in-the-Loop](tools/06-human-in-the-loop/) | Intermediate | ✅ Ready |
| 07 | [Dynamic Tool Selection](tools/07-dynamic-tool-selection/) | Intermediate | ✅ Ready |
| 08 | [Programmatic Tool Generation](tools/08-programmatic-tool-generation/) | Intermediate | ✅ Ready |
| 09 | [Streaming with Tools](tools/09-streaming-tools/) | Intermediate | ✅ Ready |
| 10 | [Tool Use with Memory](tools/10-tool-use-with-memory/) | Intermediate–Advanced | ✅ Ready |
| 11 | [Multi-Tool Agent](tools/11-multi-tool-agent/) | Advanced | ✅ Ready |
| 12 | [Tool Composition](tools/12-tool-composition/) | Advanced | ✅ Ready |
| C | [**Capstone: CLI Agent**](tools/capstone/cli-agent/) | Capstone | ✅ Ready |

---

## Planning

Chain-of-thought, ReAct, task decomposition, hierarchical planning, self-reflection, tree of thoughts, replanning.

| # | Experiment | Difficulty | Status |
|---|-----------|-----------|--------|
| 01 | [Chain-of-Thought](planning/01-chain-of-thought/) | Beginner | ✅ Ready |
| 02 | [Zero-Shot CoT](planning/02-zero-shot-cot/) | Beginner | ✅ Ready |
| 03 | [ReAct](planning/03-react/) | Intermediate | ✅ Ready |
| 04 | [Task Decomposition](planning/04-task-decomposition/) | Beginner | ✅ Ready |
| 05 | [Task Graph](planning/05-task-graph/) | Intermediate | ✅ Ready |
| 06 | [Hierarchical Planning](planning/06-hierarchical-planning/) | Intermediate | ✅ Ready |
| 07 | [Plan-and-Execute](planning/07-plan-and-execute/) | Intermediate | ✅ Ready |
| 08 | [Self-Reflection](planning/08-self-reflection/) | Intermediate | ✅ Ready |
| 09 | [Tree of Thoughts](planning/09-tree-of-thoughts/) | Advanced | ✅ Ready |
| 10 | [Backtracking](planning/10-backtracking/) | Advanced | ✅ Ready |
| 11 | [Adaptive Planning](planning/11-adaptive-planning/) | Advanced | ✅ Ready |
| 12 | [Replanning](planning/12-replanning/) | Advanced | ✅ Ready |
| 🏆 | [Capstone: Autonomous Task Solver](planning/capstone/autonomous-task-solver/) | Advanced | ✅ Ready |

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
