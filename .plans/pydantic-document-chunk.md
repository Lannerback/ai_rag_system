# Plan: Introduce Pydantic `DocumentChunk` to bind content + metadata

## Context

Chunk text and metadata currently travel the entire **write path** as two positional lists —
`Tuple[List[str], List[dict]]` — produced by every loader and re-zipped at every hop:

- `base_document_loader.load_documents` → `Tuple[List[str], List[dict]]`
- `document_loader_facade.load_all_documents` → `extend()` both lists in lockstep
- `source_grouping.group_by_source(texts, metadatas)` → `zip()`
- `ingestion_service._ingest_one` / `pgvector_store._replace_document` → unpack tuples back to two lists
- `collection_writer.build_chunks(texts, metadatas)` → `zip(texts, metadatas, embeddings)`
- `faiss_vector_store.add_documents(texts, metadatas)` → `zip(texts, metadatas)`

If any hop filters, sorts, or partially fails one list but not the other, the index alignment
silently breaks and the **wrong `source`/`page` is written into the DB**. The read path is already
safe (a `Chunk` row binds everything), so the fragility is strictly pre-insertion.

**Fix:** a Pydantic `DocumentChunk` that binds `content` + `metadata` into one object, carried as
`list[DocumentChunk]` from loader → facade → ingestion → writer. Misalignment becomes structurally
impossible. Add a validated `ChunkMetadata` schema and a read-side `RetrievedChunk` DTO for
`search()` output.

## Decisions locked (from Q&A)

| Decision | Choice |
|---|---|
| Scope | **Write path + read-side `RetrievedChunk` DTO** |
| FAISS | **Keep in sync** with the new `add_documents(list[DocumentChunk])` signature |
| `base_document_loader` `@property`/`@abstractmethod` bug | **Fix now** |
| `chunk_id` (roadmap item 8) | **Store field only** — deterministic, rides in JSONB, no write-path rewrite |
| `chunk_id` basis | **`hash(source + content)`** |
| Metadata strictness | **`extra="allow"`** — canonical fields validated, unknown keys preserved to JSONB |

**Explicitly out of scope:** per-chunk upsert / incremental re-embedding. `chunk_id` is stored and
returned for citations/traceability only; the write path stays delete-all-then-reinsert. The upsert
optimization is a clean follow-up that this change unblocks (no re-ingestion needed later).

## Three model families — what stays what

1. **Pydantic domain/transport models (the fix, NEW):** `DocumentChunk`, `ChunkMetadata`,
   `RetrievedChunk`. Flow: loader → facade → ingestion → writer (write); store → facade →
   rag_service (read). Live in `src/ai/document_loaders/document_chunk.py` (+ `RetrievedChunk`
   co-located or in the vector store package — see below).
2. **SQLAlchemy ORM models (UNCHANGED):** `EmbeddingCollection`, `Document`, `Chunk` in
   `pgvector/models.py`. Keep `Vector()`, `JSONB`, `CheckConstraint`, `UniqueConstraint`,
   relationships. **Do NOT convert to Pydantic.** No schema migration — `chunk_id` lands in the
   existing `metadata` JSONB column via `extra_metadata`.
3. **Read-side DTO (`RetrievedChunk`, NEW):** replaces the `{"content", "metadata"}` dict returned
   by `search()`. Requested in scope, so implemented (not optional).

## New module: `src/ai/document_loaders/document_chunk.py`

```python
from hashlib import sha256
from typing import Literal
from pydantic import BaseModel, Field, model_validator

LoaderType = Literal["text", "ocr", "llm_extractor"]

def compute_chunk_id(source: str, content: str) -> str:
    return sha256(f"{source}\0{content}".encode()).hexdigest()

class ChunkMetadata(BaseModel):
    model_config = {"extra": "allow"}          # preserve unknown keys → JSONB
    source: str
    loader: LoaderType
    page: int | None = None
    language: str | None = None
    chunk_id: str | None = None                # deterministic; filled by DocumentChunk validator

class DocumentChunk(BaseModel):
    content: str = Field(min_length=1)
    metadata: ChunkMetadata

    @model_validator(mode="after")
    def _bind_chunk_id(self) -> "DocumentChunk":
        if self.metadata.chunk_id is None:
            self.metadata.chunk_id = compute_chunk_id(self.metadata.source, self.content)
        return self
```

Notes:
- `chunk_id` is derived automatically at construction from the bound `source` + `content`, so it is
  impossible to attach the wrong id. Reuse/align the hash with `DocumentChangeDetector.checksum`
  (verify its algorithm; use the same `sha256` family for consistency).
- `content` `min_length=1` enforces the existing whitespace-skip behavior at the type level.
- `loader` is now **required** on `ChunkMetadata` — each loader must set it explicitly (see below),
  replacing the mapper's inference from `ocr`/`llm_extracted` flags.

### `RetrievedChunk` (read DTO)

Co-locate in `document_chunk.py` (or `vector_store_service/retrieved_chunk.py`):

```python
class RetrievedChunk(BaseModel):
    content: str
    metadata: ChunkMetadata          # re-validated on read; extra=allow absorbs JSONB keys
    score: float | None = None       # optional; None until ranking exposes it
```

## The single Pydantic → ORM conversion boundary

**`CollectionWriter.build_chunks` is the one and only Pydantic→ORM seam.** New signature:

```python
def build_chunks(self, collection_id, document_id, chunks: List[DocumentChunk]) -> List[Chunk]:
    embeddings = self._embedder.embed_documents([c.content for c in chunks])
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        columns = self._mapper.to_columns(chunk.metadata)   # accepts ChunkMetadata now
        Chunk(..., content=chunk.content, page=columns["page"], ...)
```

The only remaining `zip` is `chunks ↔ embeddings` (embeddings are produced 1:1 from the chunk list,
so they cannot misalign against metadata). `ChunkMetadataMapper.to_columns` changes to accept a
`ChunkMetadata` (use `metadata.model_dump()` internally, then pop canonical keys; `chunk_id` and any
extra keys remain in `extra_metadata` → JSONB). `to_metadata` stays returning a dict for the reverse
trip but is now consumed to build a `RetrievedChunk`/`ChunkMetadata`.

## File-by-file changes

### Loaders (write source of the models)
- **`base_document_loader.py`** — Fix the bug: remove `@property`, keep `@abstractmethod`, rename
  intent to a real method: `def load_documents(self) -> list[DocumentChunk]:`. Callers already
  invoke it as a method, so this only corrects the broken decorator.
- **`text_document_loader.py`** — Build `DocumentChunk(content=..., metadata=ChunkMetadata(source,
  loader="text", page, language, **extra))` instead of appending to parallel lists. Keep
  `_normalize_metadata` for key mapping (`page_number`→`page`, `languages[0]`→`language`), then feed
  the normalized dict into `ChunkMetadata`. Whitespace-skip now redundant with `min_length=1` but
  keep the explicit `continue` to avoid raising on empty chunks.
- **`ocr_document_loader.py`** — Emit `loader="ocr"`. Move the `ocr: True` and `lang` keys into
  extra (or map `lang`→`language`). Return `list[DocumentChunk]`.
- **`llm_extractor_document_loader.py`** — Emit `loader="llm_extractor"`. Replace the
  mutate-lists-by-reference pattern in `_process_pdf` / `_extract_text_from_page` with returning /
  accumulating `list[DocumentChunk]`. Map `lang`→`language`, keep `llm_extracted` in extra.
- **`document_loader_facade.py`** — `load_all_documents(self) -> list[DocumentChunk]`; `extend` one
  list. Keep the per-loader try/except (a failing loader still can't misalign anything now).

### pgvector write path
- **`source_grouping.py`** — `group_by_source(chunks: list[DocumentChunk]) ->
  OrderedDict[str, list[DocumentChunk]]`; group on `chunk.metadata.source`. `document_metadata(...)`
  takes `list[DocumentChunk]`, derives doc-level dict from `chunks[0].metadata` minus
  `page`/`source`/`chunk_id`.
- **`collection_writer.py`** — new `build_chunks(..., chunks: List[DocumentChunk])` (the conversion
  boundary, above).
- **`chunk_metadata_mapper.py`** — `to_columns(metadata: ChunkMetadata)`; internally
  `model_dump()`, pop canonical, leave `chunk_id`+extras in `extra_metadata`. `loader` inference
  simplifies (loader is now always set) but keep back-compat popping of `ocr`/`llm_extracted`/`lang`
  so JSONB stays clean.
- **`ingestion_service.py`** — `ingest`: `chunks = self._loader.load_all_documents()`;
  `grouped = group_by_source(chunks)`. `_ingest_one(..., chunks: list[DocumentChunk])` drops the
  `texts`/`metadatas` unpack; `checksum_many([c.content for c in chunks])`;
  `document_metadata(chunks)`; `build_chunks(collection_id, document.id, chunks)`.

### Vector store interface + backends
- **`base_vector_store.py`** — `add_documents(self, chunks: list[DocumentChunk]) -> None`;
  `search(self, query, k=3) -> list[RetrievedChunk]`.
- **`vector_store_facade.py`** — forward `list[DocumentChunk]` / return `list[RetrievedChunk]`.
- **`pgvector_store.py`** — `add_documents(chunks)`: drop `metadatas or [...]` default;
  `group_by_source(chunks)`; `_replace_document(..., chunks)` drops the unpack, passes chunks to
  `build_chunks`. `search`: build `RetrievedChunk(content=chunk.content,
  metadata=ChunkMetadata(**self._mapper.to_metadata(chunk, source)))`.
- **`faiss_vector_store.py`** (keep in sync) — `add_documents(chunks)`: embed
  `[c.content for c in chunks]`, store `{"content": c.content, "metadata": c.metadata.model_dump()}`.
  `search`: wrap stored dicts into `RetrievedChunk`. No index/format change on disk.
- **`vector_store_provider.py` / `base_vector_store_initializer.py`** — no signature change; verify
  no incidental references to the old tuple contract.

### RAG read side
- **`rag_service.py`** — `_get_relevant_docs -> list[RetrievedChunk]`. Update `answer_question`:
  `doc.metadata.source` / `doc.metadata.page` / `doc.content`; dedup key via
  `doc.metadata.model_dump_json()` (replaces `json.dumps(doc["metadata"], ...)`); `sources` list
  becomes `doc.metadata.model_dump()`.

## Testing

Backend suites (`test_pgvector_store.py`, `test_ingestion.py`, `test_retrieval_ranking.py`) hit the
**local pgvector DB**; `conftest.py::clean_db` TRUNCATEs `chunks/documents/embedding_collections`
per test and `test_schema_db`/`--force-rebuild` can DROP the `test_rag` schema. **Re-ingest real
docs after running the suite** since it wipes local data.

- **`fakes.py`** — `FakeDocumentLoaderFacade.load_all_documents -> list[DocumentChunk]`. Add a
  helper `make_chunks(texts, metadatas)` so existing tuple-style test data maps to chunks with one
  call.
- **`test_text_document_loader.py`** — assertions move from `metadatas[i]["page"]` to
  `chunks[i].metadata.page`, `.language`, and `.model_extra` for pruned/extra keys. Update the
  strict dict-equality test (`prunes noisy metadata`) to compare `ChunkMetadata` fields.
- **`test_ingestion.py`** — replace in-place `texts[0] = "..."` mutations (lines ~56, ~88) with
  rebuilding the `DocumentChunk` list; slicing (`[texts[0]],[metadatas[0]]`) → `[chunks[0]]`.
- **`test_pgvector_store.py`** — `add_documents(make_chunks(...))`; results are `RetrievedChunk`:
  `r.content`, `r.metadata.source`, `.page`, `.language`, `.model_extra["custom_key"]`.
- **`test_rag_service.py`** — `FakeVectorStoreFacade` returns `list[RetrievedChunk]`; test docs built
  as `RetrievedChunk`. Verify nested-metadata dedup still holds via `model_dump_json`.
- **`test_retrieval_ranking.py`** — fixture feeds chunks; `doc["content"]` → `doc.content`.
- **`test_rebuild.py`, `test_migration.py`** — schema/SQL only; no change expected. Confirm green.
- **New tests** (add): (a) `DocumentChunk` rejects empty `content` (`min_length=1`); (b) `chunk_id`
  is deterministic for same `(source, content)` and differs when either changes; (c) unknown
  metadata keys survive the round-trip into `extra_metadata` JSONB and back via `search()`;
  (d) `ChunkMetadata` rejects an invalid `loader` literal; (e) content↔metadata stay bound through
  `group_by_source` (grouping a mixed list keeps each content with its own source).

## Migration / compatibility

- **No DB migration.** ORM `models.py` untouched; `chunk_id` and any extra keys ride the existing
  `chunks.metadata` JSONB column via `extra_metadata`. Existing rows are unaffected (they simply
  lack `chunk_id` until re-ingested — acceptable, it's citation-only).
- **FAISS `metadata.pkl` / index format unchanged** — still `{"content", "metadata": dict}` on disk;
  only the in-memory add/search signatures change.
- Backward tolerance: `ChunkMetadataMapper` keeps popping legacy `ocr`/`llm_extracted`/`lang` keys so
  previously-ingested JSONB and any external callers stay clean.

## Verification

1. `ruff check src/` (non-test code only, per house rules — do **not** lint `tests/`).
2. `pytest` — full suite green (respect the DB-wipe note; use `--force-rebuild` if schema drift).
3. Manual end-to-end: run ingestion against a real doc dir, then hit the API / `dev.py` query path
   and confirm `search()` returns `RetrievedChunk` with correct `source`/`page`/`chunk_id`, and
   `rag_service.answer_question` still returns deduped `sources`.
4. Re-ingest real corpus after tests (local DB was wiped).

## Follow-ups (notify, do not implement)

- **Per-chunk upsert** keyed on `chunk_id` to skip re-embedding unchanged chunks (the real cost win
  this change unblocks).
- `document_metadata` derives doc-level fields from `chunks[0]` only — fine today, but fragile if a
  source ever mixes heterogeneous doc-level metadata.
