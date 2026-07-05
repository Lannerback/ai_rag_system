# RAG System Overview

## Architecture Summary

This project is a FastAPI RAG system using PostgreSQL with pgvector as the vector database.
Documents are partitioned into typed elements (Unstructured, `fast` PDF strategy), cleaned
of layout noise, chunked section-aware with `chunk_by_title`, legal-section tagged, embedded
with Gemini (asymmetric document/query task types), and stored in pgvector with their
original text and metadata.

At query time the user question is embedded with the same model, then a **three-stage
retrieval pipeline** runs: wide vector recall by cosine similarity → similarity floor →
cross-encoder reranking (BGE) → optional MMR diversity (currently disabled). The final
chunks are passed to the LLM as grounding context.

See `architect-choices.md` for the rationale behind each choice below.

## Current Production-Like Settings

```text
Vector backend: pgvector
Database: PostgreSQL + pgvector extension
Collection: default
Embedding provider: gemini
Embedding model: models/gemini-embedding-001
Embedding dimension: 3072
Embedding task types: RETRIEVAL_DOCUMENT (docs) / RETRIEVAL_QUERY (queries)
Retrieval metric: cosine similarity (pgvector cosine_distance)

Retrieval pipeline:
  candidate_k     120   (wide vector recall)
  min_similarity  0.5   (cosine floor on the pool)
  reranker        BGE   (BAAI/bge-reranker-v2-m3, enabled)
  rerank_k        30    (kept after reranking)
  diversity       MMR   (implemented, DISABLED in config)
  final_k         12    (chunks sent to the LLM)

LLM model: gemini-2.5-pro
LLM top_p: 0.95
LLM temperature: 0.1
```

## Retrieval Pipeline (the `/ask` flow)

Implemented in `src/ai/rag_service.py:_retrieve`:

```text
question
-> embed query with Gemini (RETRIEVAL_QUERY task type)
-> pgvector cosine-similarity search -> top candidate_k (120) chunks
-> drop candidates below min_similarity (0.5)
-> BGE cross-encoder reranks the pool -> keep rerank_k (30)
-> [MMR diversity selection]   (skipped: diversity block disabled)
-> take top final_k (12)
-> inject the 12 chunks into the LLM prompt
-> answer with deduplicated sources
```

Each stage is timed and logged (`[TIMING] <stage> | start=... end=... duration=...`) via
`src/common/timing.py`.

## Startup Explanation

Startup does not ingest documents. It validates readiness.

Flow:

```text
FastAPI lifespan
-> initialize_vector_store()
-> pgvector initializer
-> database reachability check
-> collection lookup
-> provider/model/dimension validation
-> chunk count logging
-> initialize RAG facade (wires embedder, vector store, reranker, diversity selector)
```

If startup fails with `embedding_collections does not exist`, migrations were not run.

Fix:

```bash
/opt/miniconda3/envs/ai_rag/bin/python -m alembic upgrade head
```

## Ingestion Explanation

Ingestion is explicit:

```bash
/opt/miniconda3/envs/ai_rag/bin/python -m src.ingestion --collection default
```

Chunking algorithm (text loader):

```text
partition file into typed elements (partition_pdf fast strategy for PDFs)
-> ElementCleaner drops headers/footers/page numbers/markers/short fragments
-> chunk_by_title (section-aware, max 1200 chars, overlap 150)
-> LegalSectionTagger prepends the governing Article heading to continuation chunks (PDFs)
```

Storage algorithm:

```text
load documents
group chunks by source file
create or validate collection
compute SHA-256 checksum for each source document
skip unchanged documents
embed changed/new chunks (RETRIEVAL_DOCUMENT task type)
replace old chunks for changed documents
prune removed sources
```

Why this is good:

```text
idempotent ingestion
no duplicate chunks on rerun
no re-embedding unchanged documents
collection prevents model/dimension mismatch
layout noise removed before embedding
every legal chunk anchored to its Article
```

## Tables

```text
embedding_collections
  Stores collection identity: name, provider, model, dimensions.

documents
  Stores one row per source file: source path, checksum, document-level metadata.

chunks
  Stores one row per chunk: text, embedding, page, language, loader, extra metadata.

alembic_version
  Tracks applied migrations.
```

## Metadata

Text loader keeps an allowlist of provenance keys (`source`, `filename`, `page`,
`language`); everything else Unstructured emits is dropped.

```text
source    original file path
filename  Unstructured filename
page      page_number
language  first detected language
```

OCR loader metadata:

```text
source
page
lang -> language
ocr: True -> loader = ocr
```

LLM extractor metadata:

```text
source
page
lang -> language
llm_extracted: True -> loader = llm_extractor
```

Storage mapping:

```text
source -> documents.source
page -> chunks.page
language/lang -> chunks.language
loader/ocr/llm_extracted -> chunks.loader
extra fields -> chunks.metadata JSONB
```

## Retrieval Explanation

The vector-search stage (stage 1 of the pipeline):

```python
order_by(Chunk.embedding.cosine_distance(query_embedding)).limit(k)
```

Meaning:

```text
lower cosine distance = more semantically similar
score returned to the pipeline = 1.0 - cosine_distance
k at this stage = candidate_k = 120
```

The full multi-stage pipeline that runs after this vector search is described in the
"Retrieval Pipeline" section above and, in depth with rationale, in `architect-choices.md`.

Note: `llm.default_k` (20) is a **legacy** raw-recall knob. It no longer drives `/ask`; it
is retained only for the raw vector-search regression tests in
`tests/test_retrieval_ranking.py`.
