# Lesson: Naive RAG

**Vertical:** RAG | **Difficulty:** Beginner | **Status:** 🔜 Coming Soon

---

## Table of Contents

1. [The Problem: LLMs Don't Know Your Data](#1-the-problem-llms-dont-know-your-data)
2. [What Is RAG?](#2-what-is-rag)
3. [The Naive RAG Pipeline](#3-the-naive-rag-pipeline)
4. [Step 1: Document Ingestion and Chunking](#4-step-1-document-ingestion-and-chunking)
5. [Step 2: Embedding and Indexing](#5-step-2-embedding-and-indexing)
6. [Step 3: Retrieval](#6-step-3-retrieval)
7. [Step 4: Augmented Generation](#7-step-4-augmented-generation)
8. [Chunking Strategies](#8-chunking-strategies)
9. [Why Naive RAG Fails](#9-why-naive-rag-fails)
10. [Failure Modes](#10-failure-modes)
11. [Key Principles](#11-key-principles)
12. [In the Real World](#12-in-the-real-world)
13. [Running the Experiment](#13-running-the-experiment)

---

## 1. The Problem: LLMs Don't Know Your Data

LLMs are trained on data with a fixed cutoff date. They know nothing about:

- Your company's internal documents
- Data created after their training cutoff
- Private, proprietary, or niche content that wasn't in the training corpus
- Real-time information (prices, availability, news)

You cannot solve this by telling the model facts in the system prompt — context windows have limits, and stuffing the entire company wiki into every request is expensive, slow, and often exceeds token limits.

Fine-tuning the model on your data is expensive, slow (hours to days), doesn't update well, and doesn't prevent hallucinations about the fine-tuned content.

RAG (Retrieval-Augmented Generation) is the standard solution: retrieve only the relevant parts of your data at query time, and inject them into the model's context.

---

## 2. What Is RAG?

RAG combines two things:
- **Retrieval** — Given a query, find the most relevant documents (or chunks of documents) from a corpus
- **Augmented Generation** — Give those documents to the model as context, and ask it to answer based on them

```
User query
    │
    ▼
[Retriever] ← searches indexed documents
    │
    ▼
Relevant chunks
    │
    ▼
[LLM] ← "Answer using this context: [chunks] \n\n Question: [query]"
    │
    ▼
Grounded answer
```

The model doesn't need to memorize your data. It just needs to read the relevant pieces when answering. This shifts the problem from "how do I get knowledge into the model" to "how do I find the right knowledge to give the model."

---

## 3. The Naive RAG Pipeline

Naive RAG is the simplest possible implementation of this idea:

**Offline (indexing):**
1. Load documents from source (files, URLs, database)
2. Split documents into fixed-size chunks
3. Embed each chunk using an embedding model
4. Store chunks and embeddings in a vector database

**Online (query time):**
1. Embed the user's query with the same embedding model
2. Find the top-k chunks nearest to the query vector (cosine similarity)
3. Concatenate the chunks into a context string
4. Prompt the model: "Answer the question based on this context: [chunks]"
5. Return the model's response

This is called "naive" not because it's poorly designed, but because it makes no attempt to handle the many ways this simple pipeline fails. It is the baseline — the thing every improvement is compared against.

---

## 4. Step 1: Document Ingestion and Chunking

Documents come in many formats (PDF, HTML, Markdown, database rows). The first step is normalizing them to plain text.

Chunking splits that text into pieces small enough to embed meaningfully and fit into the LLM context alongside other chunks:

```
Document (10,000 tokens)
    ↓ chunk(size=500, overlap=50)
Chunk 1: tokens 0–500
Chunk 2: tokens 450–950   ← 50-token overlap with previous
Chunk 3: tokens 900–1400
...
Chunk N: tokens 9500–10000
```

**Why overlap?** To avoid splitting a sentence or concept across chunk boundaries. The overlap ensures the boundary area appears in at least two chunks.

**Chunk size** is a hyperparameter with significant impact on retrieval quality. Too small: each chunk lacks enough context for the model to interpret it. Too large: retrieval is imprecise, and the chunk may contain irrelevant content alongside the relevant part.

---

## 5. Step 2: Embedding and Indexing

An **embedding model** converts text to a dense vector (a list of floats, typically 768–3072 dimensions). Texts with similar meaning produce similar vectors.

```
"How do I reset my password?"  → [0.12, -0.34, 0.89, ...]
"Steps to recover account access" → [0.11, -0.31, 0.91, ...]  ← similar vector
"What is the capital of France?" → [-0.78, 0.22, -0.15, ...] ← different vector
```

The embedding is the mathematical representation of the text's *meaning*, not its exact words. This enables semantic search — finding relevant content even when the exact words don't match.

After embedding all chunks, you store the vectors in a **vector database** (or just a numpy array for small corpora). The vector database supports fast approximate nearest-neighbor (ANN) search — given a query vector, find the k most similar stored vectors efficiently.

---

## 6. Step 3: Retrieval

At query time:
1. Embed the query
2. Compute cosine similarity between the query vector and all chunk vectors
3. Return the top-k chunks by similarity score

```python
query_vec = embed("What is the refund policy?")
similarities = cosine_similarity(query_vec, all_chunk_vecs)
top_k_indices = argsort(similarities)[-k:]
retrieved_chunks = [chunks[i] for i in top_k_indices]
```

**Cosine similarity** measures the angle between vectors, not their magnitude. Two vectors pointing in the same direction in high-dimensional space represent semantically similar texts, regardless of length.

`k` (number of chunks to retrieve) is another hyperparameter. More chunks give the model more context but increase token cost and can dilute relevance. Typical values: 3–10 chunks.

---

## 7. Step 4: Augmented Generation

Combine the retrieved chunks with the query into a prompt:

```
You are a helpful assistant. Answer the user's question using only 
the provided context. If the answer is not in the context, say so.

Context:
---
[Chunk 1 text]
---
[Chunk 2 text]
---
[Chunk 3 text]
---

Question: What is the refund policy?
Answer:
```

The model reads the retrieved context and synthesizes an answer grounded in the actual documents. Key prompt design choices:

- **"Use only the context"** — Reduces hallucination by explicitly constraining the model to the provided information.
- **"If not in the context, say so"** — Prevents the model from inventing an answer when the retrieved chunks don't contain the relevant information.
- **Source attribution** — Include document titles or IDs in the chunks so the model can cite sources.

---

## 8. Chunking Strategies

Chunking has an outsized effect on RAG quality. The naive fixed-size approach often produces poor chunks:

**Fixed size (naive)** — Split every N tokens. Fast and simple, but splits sentences mid-thought and separates related content.

**Sentence/paragraph splitting** — Split at natural language boundaries. Better semantic coherence, but variable chunk sizes.

**Recursive character splitting** — Try to split at paragraph breaks, then sentences, then words, with a target size. This is LangChain's default splitter and works well for most text.

**Semantic chunking** — Embed sentences, compute similarity between consecutive sentences, and split where similarity drops sharply (topic boundaries). More expensive but produces semantically coherent chunks.

**Document structure splitting** — Use document structure (headers, sections) as chunk boundaries. Ideal for structured documents like wikis, legal docs, or technical manuals.

The right strategy depends on your document type. A PDF of meeting notes has different optimal chunking than a legal contract or a code repository.

---

## 9. Why Naive RAG Fails

Naive RAG breaks down in predictable ways that motivated an entire field of "advanced RAG" techniques:

**Retrieval misses** — The query and the relevant chunk use different vocabulary. "How do I cancel?" matches poorly against a chunk titled "Account Termination Procedures."

**Chunk boundary problems** — The answer spans multiple chunks. Retrieval finds only one half of the answer.

**Irrelevant retrieval** — The top-k chunks are semantically similar but not actually helpful. The model hallucinates anyway because the retrieved content doesn't contain the answer.

**Context poisoning** — Retrieved chunks contain contradictory information from different versions of a document. The model picks one and may be wrong.

**Lost in the middle** — Research shows LLMs are better at using context at the beginning and end of the prompt than the middle. Stuffing 10 chunks into the middle of the context degrades performance.

Every advanced RAG technique (reranking, HyDE, query expansion, multi-hop retrieval, chunk parent retrieval) is a direct response to one of these failure modes.

---

## 10. Failure Modes

**Retrieval hallucination** — The model answers correctly even when the retrieved chunks don't contain the answer, by drawing on training knowledge. Users think RAG is working when it isn't; evaluation is misleading.

**Stale index** — Documents change but the index isn't updated. Users get answers based on old information. Naive RAG has no concept of document freshness.

**Embedding model mismatch** — The embedding model used at indexing time differs from the one used at query time, causing semantic space incompatibility and degraded retrieval.

**Overly long chunks drowning signal** — A 1000-token chunk retrieved for a specific fact means 995 tokens of noise surrounding the 5-token answer. The model may miss the answer or extract an incorrect nearby fact.

---

## 11. Key Principles

> **Principle 1 — Retrieval quality is the ceiling for RAG quality.**
> No matter how good your LLM, if the retriever returns the wrong chunks, the model can't give the right answer. Invest in retrieval before investing in generation.

> **Principle 2 — Chunking is a design decision, not a default.**
> The chunk size and strategy profoundly affect what the retriever can find. Match your chunking strategy to your document structure and query patterns.

> **Principle 3 — Ground the model in the context.**
> Tell the model explicitly to use only the provided context and to say "I don't know" when the answer isn't there. This is the primary lever for reducing hallucination in RAG systems.

> **Principle 4 — Naive RAG is the baseline, not the destination.**
> Naive RAG is fast to build and works well for simple corpora and straightforward queries. But it has known failure modes. Learn them now; they'll motivate every advanced technique.

---

## 12. In the Real World

**Notion AI / Confluence AI**
Knowledge management tools use RAG to answer questions about workspace content. The document corpus is the user's wiki; the retriever finds relevant pages; the model synthesizes an answer. Chunking follows document/section structure.

**GitHub Copilot — code search**
Copilot uses a form of RAG to include relevant code from the current repository in its context. When you ask it to implement a function, it retrieves similar functions and usages from your codebase as examples. The "chunks" are code snippets; the "embedding model" is trained on code.

**Cursor**
Cursor's "codebase indexing" feature is RAG for code. It indexes your entire repo, embeds files/functions, and retrieves relevant code when you ask questions or request changes. This is why it can answer "how does authentication work in this repo?" without you providing the files.

**Perplexity AI**
Perplexity runs a search query through web retrieval, retrieves current web pages, chunks and embeds the results, and augments the LLM's context with the retrieved content. It is RAG with live web retrieval as the document source — essentially a real-time, web-scale RAG system.

**LlamaIndex**
LlamaIndex is fundamentally a library for building RAG pipelines. Its entire API surface (`VectorStoreIndex`, `QueryEngine`, `RetrieverQueryEngine`) is structured around the ingest → embed → retrieve → generate pipeline. It is the primary open-source library for RAG system construction.

**LangChain — `RetrievalQA`**
LangChain's `RetrievalQA` chain implements the exact naive RAG pipeline: load documents, split, embed, store in a vector store, retrieve, generate. It is one of the most commonly used chains in the LangChain ecosystem.

**AWS Bedrock Knowledge Bases**
Amazon's managed RAG service: you point it at an S3 bucket, it ingests, chunks, embeds, and indexes your documents. You query it via API and get grounded responses. It is naive RAG as a fully managed product.

**Glean (enterprise search)**
Glean indexes all of an organization's SaaS tools (Slack, Confluence, Jira, Drive, email) and uses RAG to answer questions across the entire company knowledge graph. The "documents" are messages, tickets, pages, and files; the retriever must handle cross-source multi-hop queries — far beyond naive RAG.

---

## 13. Running the Experiment

```bash
# From the project root (experiment coming soon)

uv run python rag/01-naive-rag/demo.py --mock
ANTHROPIC_API_KEY=sk-... uv run python rag/01-naive-rag/demo.py --real
```

**Planned exercises:**
1. Index a small set of text documents and query them. Observe what gets retrieved.
2. Change chunk size from 100 to 500 to 1000 tokens and observe how retrieval quality changes.
3. Ask a question whose answer spans two chunks — confirm naive retrieval fails.
4. Ask a question not covered by your documents — confirm the model says "I don't know" rather than hallucinating.

---

*Next experiment: Advanced RAG — Reranking and Query Expansion (coming soon)*
