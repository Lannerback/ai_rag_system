# 02 — Add pgvector as a Selectable Vector-Store Backend (Code Plan)

> On approval this document will be persisted to `.plans/02-pgvector-implementation-plan.md`
> (companion to the existing `.plans/02-replace-faiss-with-pgvector.md` design doc).

## Context

Vectors are persisted today with `faiss.IndexFlatIP` + a pickle metadata file
(`src/ai/vector_store_service/faiss/`). We want PostgreSQL + pgvector
(`collections → documents → chunks`, idempotent ingestion) **as a second backend**, selected by
configuration. **FAISS stays — its classes, files and logic are untouched and remain fully usable;
it just becomes one option that can be disabled by config.** The `/ask` contract is unchanged.

Target DB (already running, empty):
```
DATABASE_URL = postgresql+psycopg://admin:admin@localhost:5433/rag
```

## Design Patterns Used (explicit)

The whole point is to choose store + startup behavior from `config.yaml` **without `if/else`**.
Patterns, and why each:

| Pattern | Where | Why |
|---------|-------|-----|
| **Abstract Factory** | `VectorStoreProvider` ABC → `FaissVectorStoreProvider`, `PgVectorStoreProvider` | Each backend is a *family* = a store **plus** its matching initializer. The factory produces a consistent pair so FAISS and pgvector can differ in both store and startup logic. |
| **Registry** | `VECTOR_STORE_PROVIDERS = {"faiss": ..., "pgvector": ...}` in `ServiceFactory` | Maps the `vector_store.backend` config string → provider class via **dict lookup**. Adding a backend = one registry entry. Mirrors the existing `PROVIDERS` dict. No branching. |
| **Strategy** | `BaseVectorStoreInitializer` ABC → `FaissVectorStoreInitializer`, `PgVectorStoreInitializer` | Startup behavior differs (FAISS: load-or-build from disk; pgvector: validate + expect explicit ingestion). `startup.py` calls `initializer.initialize()` polymorphically — selection already happened in the registry, so **zero `if/else` at the call site**. |
| **Singleton + Factory** (existing) | `ServiceFactory` | Reused as-is for caching the chosen store/initializer. |
| **Facade** (existing) | `VectorStoreFacade`, `RagFacade` | Already hide the concrete store — **no change needed**. |
| **Repository / DAO** | pgvector repositories | Isolate persistence; raise domain-neutral exceptions (`LookupError`, SDK errors) per DAO standard. |
| **Mapper** | `ChunkMetadataMapper` | Promote/reconstruct canonical metadata ↔ JSONB. |

**Branch-free selection flow:**
```
config.vector_store.backend ──► VECTOR_STORE_PROVIDERS[backend]  (Registry: dict lookup)
                                        │
                                        ▼
                              VectorStoreProvider            (Abstract Factory)
                              ├── create_store(embedder)      ──► BaseVectorStore
                              └── create_initializer(store,…)  ──► BaseVectorStoreInitializer
                                                                        │
                                                          startup → initializer.initialize()  (Strategy / polymorphism)
```

---

## Architecture Overview

```
Presentation:   src/api.py (unchanged)        |   src/ingestion/__main__.py (NEW CLI)
Service:        RagService (unchanged)         |   IngestionService (NEW, pgvector only)
Selection:      ServiceFactory + VECTOR_STORE_PROVIDERS registry + VectorStoreProvider factories
Store impls:    FaissVectorStore (KEPT)        |   PgVectorStore(BaseVectorStore) (NEW)
Init strategy:  FaissVectorStoreInitializer (KEPT logic) | PgVectorStoreInitializer (NEW)
DAO (pgvector): CollectionRepository / DocumentRepository / ChunkRepository
ORM/Infra:      models.py (declarative) | database.py (engine+Session singleton)
Cross-cutting:  CollectionValidator, ChunkMetadataMapper, CollectionWriter, DocumentChangeDetector
```

`search(query, k)` keeps returning `[{"content", "metadata"}]`, so `VectorStoreFacade`,
`RagService`, `RagFacade`, and `/ask` need **zero changes** regardless of backend.

---

## Files to CREATE

### Backend-selection scaffolding — `src/ai/vector_store_service/`

| File | Responsibility | Pattern |
|------|----------------|---------|
| `base_vector_store_initializer.py` | ABC `BaseVectorStoreInitializer` with `initialize()`. | Strategy interface |
| `vector_store_provider.py` | ABC `VectorStoreProvider` with `create_store(embedder)` + `create_initializer(store, doc_loader_facade)`; concrete `FaissVectorStoreProvider`, `PgVectorStoreProvider`. | Abstract Factory |

### pgvector backend — `src/ai/vector_store_service/pgvector/`

| File | Responsibility | Key classes |
|------|----------------|-------------|
| `database.py` | Lazy singleton engine + `sessionmaker`; reads `DATABASE_URL`. | `Database`, `get_session()` |
| `models.py` | Declarative models + constraints. | `Base`, `EmbeddingCollection`, `Document`, `Chunk` (`pgvector.sqlalchemy.Vector`) |
| `repositories/collection_repository.py` | get-or-create + identity lookup. | `CollectionRepository` |
| `repositories/document_repository.py` | per-source upsert, checksum lookup, delete-by-source, list-sources. | `DocumentRepository` |
| `repositories/chunk_repository.py` | bulk insert, delete-by-document, cosine KNN (`embedding <=> q`). | `ChunkRepository` |
| `chunk_metadata_mapper.py` | promote `page/language/loader` ↔ JSONB; reconstruct with `source`. | `ChunkMetadataMapper` |
| `collection_validator.py` | assert embedder provider/model/dimension == collection; fail clearly. | `CollectionValidator` |
| `collection_writer.py` | reusable write path (embed → build chunks → persist for one document); shared by store + ingestion. | `CollectionWriter` |
| `pgvector_store.py` | `BaseVectorStore` impl bound to a collection; `search()` + `add_documents()` (delegates to `CollectionWriter`). | `PgVectorStore` |
| `pgvector_store_initializer.py` | `BaseVectorStoreInitializer` impl: verify connection + resolve/validate collection; log guidance if empty (does NOT build). | `PgVectorStoreInitializer` |

### Ingestion — `src/ingestion/`

| File | Responsibility |
|------|----------------|
| `__main__.py` | CLI `python -m src.ingestion --collection <name> [--rebuild]` → `IngestionService`. |
| `ingestion_service.py` | idempotent ingest: load via `DocumentLoaderFacade`, group by `source`, checksum-diff, transactional replace, prune removed sources. |
| `document_change_detector.py` | sha256 checksum + compare vs stored. |

### Migrations — `alembic/`

`alembic.ini`, `alembic/env.py` (reads `DATABASE_URL`, targets `models.Base.metadata`),
`alembic/versions/0001_init_pgvector.py`: `CREATE EXTENSION vector`; 3 tables; unique
`(collection_id, source)` & `(document_id, chunk_index)`; B-tree (collection/document/source);
GIN on JSONB; `CHECK (vector_dims(embedding) = embedding_dimensions)`. No HNSW/IVFFlat yet.

### Tests — `tests/`

`test_pgvector_store.py` (KNN order, collection isolation, dim validation, metadata round-trip),
`test_ingestion.py` (idempotency, change replace, prune, mismatch reject, failed-doc preserves prior),
`test_migration.py` (extension/tables/constraints/indexes), `conftest.py` (pgvector session fixture,
rollback per test). **Existing FAISS tests are kept.**

---

## Files to EDIT

| File | Change |
|------|--------|
| `src/ai/embedders/base_embedder.py` | Add abstract read-only props `provider: str`, `model: str` (collection identity). Keep `dimension`. |
| `src/ai/embedders/azure/azure_embedder.py` | `provider="azure"`, `model` from `CONFIG["azure"]["embedding_deployment"]`. |
| `src/ai/embedders/gemini/gemini_embedder.py` | `provider="gemini"`, `model` from `CONFIG["gemini"]["embedding_model"]`. |
| `src/ai/vector_store_service/faiss/faiss_vector_store_initializer.py` | Make existing `VectorStoreInitializer` subclass `BaseVectorStoreInitializer` and rename to `FaissVectorStoreInitializer` (alias kept for safety). **Logic unchanged.** |
| `src/ai/service_factory.py` | Add `VECTOR_STORE_PROVIDERS` registry; add `get_vector_store()` + `get_vector_store_initializer()` that resolve provider via `CONFIG["vector_store"]["backend"]` (dict lookup, no `if/else`); `get_rag_facade()` uses `get_vector_store()`. Keep `get_faiss_vector_store()` delegating, for back-compat. |
| `src/startup.py` | `initialize_vector_store()` calls `ServiceFactory.get_vector_store_initializer().initialize()` — polymorphic, no branching. |
| `config.yaml` | Add `vector_store.backend: "pgvector"` (set `"faiss"` to fall back), keep existing `index_path`/`metadata_path` for FAISS, add `vector_store.collection: "default"` + `pgvector` search opts. |
| `.env.example` | Add `DATABASE_URL=postgresql+psycopg://admin:admin@localhost:5433/rag`. |
| `requirements.txt` / `environment.yml` | **Add** `sqlalchemy`, `alembic`, `psycopg[binary]`, `pgvector`. **Keep** `faiss-cpu` and `numpy`. |
| `README.md` | Document backend switch, `alembic upgrade head`, ingestion command. |

## Files DELETED

**None.** FAISS store, initializer, config keys, dependency, and tests all remain.

---

## Key Interface Decisions

- **`BaseVectorStore` preserved.** `PgVectorStore` implements `add_documents` (delegates to
  `CollectionWriter`, grouping by `source`) and `search`. FAISS implementation untouched.
- **Idempotency in `IngestionService`** (pgvector path), not in `add_documents`; both share
  `CollectionWriter` to avoid duplication.
- **Collection binding** from `CONFIG["vector_store"]["collection"]`; `CollectionValidator` enforces
  embedder provider/model/dimension match on ingest + query; startup fails clearly on mismatch.
- **Metadata fidelity** via `ChunkMetadataMapper`: promotes `page`, `language` (from existing `lang`),
  `loader` (from `ocr`/`llm_extracted`); reconstruction re-injects `source`+`page` so `RagService`
  keeps working.

---

## Verification

1. **Backend switch**: `vector_store.backend: "faiss"` → existing flow works unchanged;
   `"pgvector"` → DB path active. No `if/else` in `startup.py`/`service_factory` selection (grep).
2. **Schema**: `alembic upgrade head`; `\d+ chunks` shows vector column, constraints, indexes, extension.
3. **Ingestion**: `python -m src.ingestion --collection default`; rerun → counts unchanged; edit a source
   → only its chunks replaced; remove a source → its rows pruned.
4. **Mismatch guard**: change `llm.provider` vs an existing collection → ingest + startup fail clearly
   before any write/query.
5. **Retrieval**: `pytest tests/test_pgvector_store.py` — order, isolation, metadata round-trip.
6. **End-to-end**: start app on each backend; `POST /ask` → identical `{answer, sources}` shape.
7. **Lint + suite**: `ruff` + full `pytest` (FAISS tests still green; pgvector tests against live DB).

## Defaults (objection-only)

- Collection name `"default"`; `psycopg` v3; `faiss-cpu`/`numpy` retained.
