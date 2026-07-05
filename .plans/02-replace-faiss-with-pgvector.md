# Replace FAISS with PostgreSQL + pgvector

## Summary

Replace the FAISS index and pickle metadata store with PostgreSQL, SQLAlchemy, Alembic, and pgvector. Documents will be ingested through an explicit idempotent command, while `/ask` keeps its existing contract. Existing FAISS data will not be imported; PostgreSQL will be rebuilt from source documents.

## Database Structure

### `embedding_collections`

This table does not contain embeddings. It identifies a logical collection and records how that collection's embeddings were produced.

- `id UUID`
- unique `name`
- embedding `provider`
- embedding `model`
- embedding `dimensions`
- creation and update timestamps

A collection is permanently associated with one embedding model and dimension. This prevents querying Gemini embeddings with an incompatible Azure query vector and allows differently configured collections to coexist.

### `documents`

Each row represents a complete source document, such as a PDF, DOCX, or Markdown file. It stores information shared by all chunks from that document.

- `id UUID`
- `collection_id`
- canonical `source`
- content checksum
- document-level `metadata JSONB`
- creation and update timestamps

The `(collection_id, source)` pair must be unique. The table avoids repeating document-level data for every chunk, supports change detection, and makes it possible to replace or delete every chunk belonging to one document.

### `chunks`

Each row represents one chunk and contains that chunk's embedding.

- `id UUID`
- `document_id`
- `collection_id`
- `chunk_index`
- promoted metadata fields: `page`, `language`, and `loader`
- `content`
- content checksum
- remaining `metadata JSONB`
- `embedding vector`
- embedding dimensions
- creation and update timestamps

The `(document_id, chunk_index)` pair must be unique. Enforce `vector_dims(embedding) = embedding_dimensions` and ensure the dimensions match the associated collection.

The relationship is:

```text
embedding_collections
└── documents
    └── chunks + embeddings
```

Add B-tree indexes for collection, document, and source lookup, plus a GIN index for JSONB metadata. Use exact cosine search initially, so no HNSW or IVFFlat index is created yet.

## Implementation Changes

- Add SQLAlchemy models, repositories, connection/session management, and Alembic migrations that enable the `vector` extension and create the schema.
- Configure the database exclusively through environment variables such as `DATABASE_URL`; add a collection name to application configuration without storing credentials.
- Add an explicit ingestion command such as:

  ```bash
  python -m src.ingestion --collection <name>
  ```

- Support `--rebuild` for replacing a collection's documents when the embedding model has not changed.
- Make ingestion idempotent:
  - Group chunks by canonical source.
  - Skip unchanged documents using checksums.
  - Re-embed and transactionally replace changed documents.
  - Remove database documents no longer present after a fully successful source scan.
  - Reject model or dimension changes for an existing collection; use a new collection and re-ingest instead.
- Replace `FaissVectorStore` with `PgVectorStore` behind `BaseVectorStore`, preserving the existing `search(query, k)` result shape.
- During search:
  - Embed the query using the collection's configured model.
  - Validate the query vector's dimension.
  - Restrict the query to the configured collection.
  - Order chunks by cosine distance with `embedding <=> query_embedding`.
  - Return `content` and reconstructed metadata, with canonical columns taking precedence over duplicate JSONB keys.
- Remove FAISS initialization, disk loading and saving, pickle handling, FAISS dependencies, local vector-store configuration, and FAISS-specific tests.
- Add a pgvector-enabled PostgreSQL service for local development and update setup, migration, and ingestion documentation.
- Leave existing untracked FAISS artifacts for manual deletion rather than deleting user data automatically.

## Public Interfaces and Configuration

- Keep the `/ask` request and response contracts unchanged.
- Add `DATABASE_URL` and a configured collection name.
- Require the selected embedder to match the collection's provider, model, and dimensions; startup and ingestion must fail clearly on mismatch.
- Add `sqlalchemy`, `alembic`, `psycopg`, and `pgvector` to both dependency manifests.
- Remove `faiss-cpu` from both dependency manifests.

## Test Plan

- Verify the Alembic migration creates the extension, tables, constraints, and metadata indexes.
- Verify ingestion creates documents and chunks and can be rerun without duplicates.
- Verify changed sources are updated and removed sources are deleted.
- Verify model and dimension mismatches fail before insertion or retrieval.
- Verify exact cosine retrieval returns correctly ordered chunks and respects collection isolation.
- Verify metadata survives database round trips and promoted fields are reconstructed consistently.
- Verify failed document ingestion preserves its previous database version.
- Verify `/ask` continues returning the existing answer and sources structure.
- Run focused unit tests and integration tests against a real pgvector PostgreSQL container.

## Assumptions

- Source document directories remain local; only embeddings and their persisted metadata move to PostgreSQL.
- Existing FAISS and pickle data are disposable and will be regenerated from source documents.
- Exact search is appropriate for the current corpus size. HNSW can be introduced later through dimension-specific expression indexes when query latency requires it.
