# RAG System Overview

## Architecture Summary

This project is a FastAPI RAG system using PostgreSQL with pgvector as the vector database. Documents are loaded from configured folders, split into overlapping chunks, embedded with Gemini, and stored in pgvector with their original text and metadata. At query time, the user question is embedded with the same model, pgvector retrieves the nearest chunks using cosine distance, and those chunks are passed to the LLM as context for the final answer.

## Current Production-Like Settings

```text
Vector backend: pgvector
Database: PostgreSQL + pgvector extension
Collection: default
Embedding provider: gemini
Embedding model: models/gemini-embedding-001
Embedding dimension: 3072
Retrieval metric: cosine distance
Retrieval k: 20
LLM model: gemini-2.5-pro
LLM top_p: 0.95
```

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
-> initialize RAG facade
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

Algorithm:

```text
load documents
split into chunks
group chunks by source file
create or validate collection
compute SHA-256 checksum for each source document
skip unchanged documents
embed changed/new chunks
replace old chunks for changed documents
prune removed sources
```

Why this is good:

```text
idempotent ingestion
no duplicate chunks on rerun
no re-embedding unchanged documents
collection prevents model/dimension mismatch
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

Text loader metadata:

```text
source
page
language
loader inferred as text
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

Retrieval flow:

```text
question
-> embed query with Gemini embedding model
-> validate query dimension is 3072
-> pgvector cosine-distance search in chunks table
-> return top k chunks
-> inject chunks into LLM prompt
-> answer with sources
```

Actual search:

```python
order_by(Chunk.embedding.cosine_distance(query_embedding)).limit(k)
```

Meaning:

```text
lower cosine distance = more semantically similar
k = number of chunks returned
```

Current `k`:

```text
20
```
