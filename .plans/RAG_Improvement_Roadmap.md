
# RAG Roadmap


# Part 1 — High-Priority Improvements (Implement These)

These are the improvements that will teach the largest amount about RAG systems while making the project significantly stronger.

---

## 1. Replace FAISS with PostgreSQL + pgvector — Accomplished

Status: **Accomplished**

Implemented:

- PostgreSQL + pgvector backend
- pgvector Docker Compose setup
- Alembic migration for vector schema
- `embedding_collections`, `documents`, and `chunks` tables
- Gemini embedding model integration
- Collection validation by provider, model, and dimension
- Explicit ingestion command separated from API startup
- pgvector cosine-distance retrieval
- JSONB metadata storage with promoted metadata columns

### Why

Move from a local vector index to a real database-backed architecture.

Learn:

- pgvector
- Vector indexes
- Cosine similarity search
- Persistent storage
- JSONB metadata
- Database design
- SQL vector queries

Store:

- chunk_id
- document_id
- source
- page
- content
- metadata (JSONB)
- embedding
- timestamps

Expected architecture:

```
Document
    ↓
Chunking
    ↓
Embedding
    ↓
PostgreSQL + pgvector
    ↓
Similarity search
```

---

## 2. Add B-Tree Indexing For pgvector Tables

### Why

The pgvector backend currently filters chunks by collection before vector search:

```sql
WHERE chunks.collection_id = ...
ORDER BY chunks.embedding <=> query_embedding
LIMIT k
```

A B-tree index helps PostgreSQL quickly find rows for a specific collection, document, or source instead of scanning the full table.

Learn:

- PostgreSQL B-tree indexes
- query planning
- `EXPLAIN ANALYZE`
- database performance basics
- indexed joins and filters
- retrieval scalability

Add indexes such as:

```sql
CREATE INDEX idx_documents_collection_source
ON documents(collection_id, source);

CREATE INDEX idx_chunks_collection_id
ON chunks(collection_id);

CREATE INDEX idx_chunks_document_id
ON chunks(document_id);
```

Use `EXPLAIN ANALYZE` to compare query plans before and after indexing.

Important distinction:

- B-tree indexes speed up relational filters and joins.
- pgvector HNSW or IVFFlat indexes speed up vector nearest-neighbor search.

This should be implemented before partitioning. Partitioning is only needed once table size or collection count makes indexed single-table search too slow.

---

## 3. Add Metadata Filtering

Currently retrieval ignores metadata.

Support filters such as:

- language
- document_type
- source
- tags
- category

Future-ready fields:

- tenant
- department

Learn:

- filtered retrieval
- pre-filter vs post-filter
- retrieval architecture
- enterprise RAG

---

## 4. Add Reranking

Current:

```
Query
 ↓
Vector Search
 ↓
LLM
```

Target:

```
Query
 ↓
Vector Search (Top 30-50)
 ↓
Reranker
 ↓
Top 5-8
 ↓
LLM
```

Learn:

- candidate retrieval
- cross encoders
- relevance scoring
- retrieval precision

Possible implementations:

- CrossEncoder (SentenceTransformers)
- Cohere Rerank
- Jina AI
- Voyage
- LLM-based reranking

---

## 5. Add Hybrid Search

Current system performs semantic search only.

Implement:

```
Semantic Search
        +
BM25 Keyword Search
        ↓
Merge
        ↓
Reranker
```

Learn:

- BM25
- lexical retrieval
- Reciprocal Rank Fusion
- hybrid pipelines

Understand when hybrid retrieval is better than semantic retrieval alone.

---

## 6. Add Similarity Thresholds

Instead of always sending retrieved chunks to the LLM:

```
retrieve
    ↓
filter by score
    ↓
if nothing relevant:
    "I don't have enough information."
```

Learn:

- calibrated thresholds
- abstention
- hallucination reduction

---

## 7. Return Retrieval Scores

Every retrieved chunk should include:

- similarity score
- reranker score
- final ranking position

Useful for:

- debugging
- evaluation
- threshold tuning
- visual inspection

---

## 8. Stable Chunk IDs

Every chunk should have a deterministic ID.

Example:

```
refund_policy.pdf::page3::chunk5
```

Learn:

- traceability
- citation systems
- incremental indexing
- debugging

---

## 9. Incremental Indexing

Avoid rebuilding everything.

Implement:

- document hashing
- chunk hashing
- modified documents
- deleted documents
- new documents

Learn:

- ingestion pipelines
- synchronization
- index maintenance

---

## 10. Move Ingestion Outside API Startup

Separate ingestion from serving.

Target:

```
ingest.py
    ↓
build vector index

API
    ↓
load existing index
```

Learn:

- deployment architecture
- offline indexing
- serving vs ingestion separation

---

## 11. Source Citations

Current project returns source metadata.

Improve by attaching citations directly to answers.

Example:

```
Answer...

[Source: refund_policy.pdf, page 4]
```

Learn:

- grounded generation
- explainability
- answer traceability

---

## 12. Prompt Injection Defenses

Treat retrieved documents as untrusted input.

Learn:

- prompt injection
- secure prompting
- context isolation
- retrieval security

---

## 13. Retrieval Evaluation Framework

One of the biggest improvements.

Create:

```
eval/
    questions.json
```

Example:

- question
- expected source
- expected answer keywords

Measure:

- Recall@k
- Precision@k
- MRR
- nDCG
- latency
- groundedness
- answer correctness

Learn how professional RAG systems are evaluated.

---

## 14. Better Chunking Strategies

Compare:

- RecursiveCharacterTextSplitter
- semantic chunking
- markdown-aware chunking
- sentence chunking

Experiment with:

- chunk size
- overlap
- structure-aware splitting

Learn that chunking is one of the most important design decisions in RAG.

---

## 15. Better Provider Abstraction

Currently one provider config selects both LLM and embeddings.

Separate:

- embedding provider
- generation provider
- reranker provider

Learn modular AI architecture.

---

## 16. Logging and Observability

Log:

- retrieved chunks
- similarity scores
- reranker scores
- latency
- tokens
- provider
- failures

Useful for debugging and evaluation.

---

# Suggested Order

1. PostgreSQL + pgvector — accomplished
2. B-tree indexing for pgvector tables
3. Metadata filtering
4. Retrieval scores
5. Similarity thresholds
6. Reranking
7. Hybrid search
8. Stable chunk IDs
9. Source citations
10. Better chunking
11. Incremental indexing
12. Retrieval evaluation
13. Prompt injection defenses
14. Better provider abstraction
15. Observability
16. Separate ingestion pipeline

---

# Part 2 — Nice to Have

These are valuable but not essential for demonstrating strong RAG knowledge.

---

## Conversation History

Maintain previous interactions.

Learn:

- memory
- query rewriting
- conversational retrieval

---

## Query Rewriting

Rewrite vague user questions before retrieval.

Example:

```
"How does it work?"
↓

"How does the document ingestion pipeline work?"
```

---

## Streaming Responses

Return tokens progressively.

Useful for UX.

---

## Multi-query Retrieval

Generate several search queries from one user question.

Retrieve with all of them.

Merge results.

---

## Context Compression

Compress retrieved chunks before generation.

Useful when documents are long.

---

## Parent-Child Retrieval

Retrieve child chunks but provide larger parent sections.

Improves context quality.

---

## Contextual Compression Retriever

Filter irrelevant sentences inside retrieved chunks before sending them to the LLM.

---

## Multiple Retrieval Strategies

Allow switching between:

- semantic
- hybrid
- keyword
- reranked

Compare evaluation results.

---

## Multiple Embedding Models

Experiment with:

- Gemini
- OpenAI
- Voyage
- BAAI BGE
- Nomic
- Jina

Compare retrieval quality.

---

## Approximate Search

Experiment with pgvector HNSW indexes instead of exact search.

Learn scalability trade-offs.

---

## Advanced Citation Verification

Verify that every generated claim is actually supported by retrieved chunks.

---

## Dashboard

Build a small interface showing:

- retrieved chunks
- scores
- reranker output
- final prompt
- final answer

Very useful for demos and debugging.

---

# Final Goal

By implementing the **Part 1** improvements you will learn nearly every major component of modern RAG systems:

- ingestion
- chunking
- embeddings
- vector databases
- metadata filtering
- retrieval
- hybrid search
- reranking
- prompt engineering
- grounded generation
- evaluation
- observability
- production-oriented architecture

