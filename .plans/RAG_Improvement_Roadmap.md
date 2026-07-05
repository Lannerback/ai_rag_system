# RAG Roadmap

This roadmap lists only the improvements **not yet implemented**. Items already shipped in
this codebase have been removed (see "Already Accomplished" below for the audit trail).

---

## Already Accomplished (removed from the backlog)

These were on the original roadmap and are now done — do not re-plan them:

- **PostgreSQL + pgvector backend** — `embedding_collections → documents → chunks` schema,
  Alembic migration, cosine-distance retrieval, JSONB metadata + promoted columns.
- **B-tree indexing** — `ix_documents_collection_id`, `ix_chunks_collection_id`,
  `ix_chunks_document_id`, the `uq_documents_collection_source` unique constraint, and a GIN
  index on `chunks.metadata` all exist in `models.py`.
- **Reranking** — BGE cross-encoder (`BAAI/bge-reranker-v2-m3`) with a FlashRank fallback,
  behind a provider registry.
- **Similarity thresholds** — `retrieval.min_similarity = 0.5` cosine floor with abstention
  (`no_docs_found`) when nothing clears it.
- **Incremental indexing** — SHA-256 per-document checksums, skip-unchanged, replace-changed,
  prune-removed (`DocumentChangeDetector`).
- **Ingestion separated from API startup** — explicit `python -m src.ingestion` command.
- **Retrieval evaluation framework** — `tests/eval/` gold set, `Recall@k`, `MRR`, regression
  floors in `tests/test_retrieval_ranking.py`.
- **Structure-aware chunking** — Unstructured `chunk_by_title` + `ElementCleaner` +
  `LegalSectionTagger`, replacing the fixed-size character splitter.

---

# Part 1 — High-Priority Improvements

---

## Step 1 — Conditional Metadata Filtering (query-driven)

**Problem.** Retrieval ignores metadata entirely: every `/ask` does a full-corpus cosine
scan. Today's stored metadata (`source`, `page`, `language`, `loader`) is *citation-only* —
`page`/`source` are **discovered** attributes (the whole point of retrieval is to find which
page/document is relevant), so they cannot be used as a pre-filter. This step adds
filtering on attributes that are **knowable from the question itself** (e.g. `country`,
`document_type`, `language`), and applies the filter **conditionally**: only when the query
actually names one.

### Pattern: Self-Querying / Query Construction

This is the established "self-query retriever" / "auto-retriever" pattern. One extra LLM call
in front of the existing pipeline turns a natural-language question into
`(structured filter, cleaned semantic query)`, then hybrid search combines an exact SQL
`WHERE` with the existing vector `ORDER BY`.

```text
user question
  -> QueryAnalyzer (LLM, structured output)
       -> filter: {"country": "France"} | {}   (empty when none detected)
       -> cleaned_query: filter phrasing stripped, topic preserved
  -> validate filter value against closed vocabulary (distinct DB values / config enum)
  -> embed cleaned_query (RETRIEVAL_QUERY)
  -> pgvector search with conditional WHERE metadata @> filter
  -> min_similarity floor -> BGE rerank -> (MMR) -> final_k
  -> existing answer_question generation call (unchanged)
```

Note this is **2 LLM calls total**, not 3: analysis + the existing generation call. The
"go back to the LLM with candidates" step is the current `RagService.answer_question`
generation — unchanged.

### How the LLM builds the filter and refines the query

The analyzer LLM receives the raw question plus the list of **filterable fields and their
allowed values**, and must return a schema-constrained JSON object. It does two things:

1. **Extract filters** — map any phrase that names a filterable attribute to a field/value.
2. **Rewrite the query** — remove the filter phrasing so the embedding represents only the
   topic. This matters: if you embed `"France's obligations in the French market"`, the
   embedder has no idea "French market" is a filter instruction and drags the vector toward
   country semantics, diluting the real topic and making the filter and the embedding fight
   over the same signal. The `WHERE` clause carries the country deterministically; the
   embedding should carry only "obligations under Article 6".

Example prompt shape (analyzer):

```text
System: You convert a user question into a retrieval filter and a cleaned search query.
Filterable fields:
  - country: one of [France, Germany, Italy, Spain]   (null if the question names none)
Rules:
  - Only set a field if the question explicitly refers to it.
  - Remove the words that express the filter from cleaned_query.
  - Never invent a value outside the allowed list; use null if unsure.
Return JSON matching the schema.

User: "What are France's obligations in the French market under Article 6?"

-> {"country": "France",
    "cleaned_query": "obligations under Article 6"}
```

If the question names no country:

```text
User: "What are the transparency obligations for high-risk systems?"
-> {"country": null,
    "cleaned_query": "transparency obligations for high-risk systems"}
```

### Pydantic schema (structured output, not free-text JSON)

Use Gemini structured/function-calling output bound to a Pydantic model — never regex-parse
free text.

```python
from typing import Optional
from pydantic import BaseModel, Field

class QueryFilter(BaseModel):
    country: Optional[str] = Field(
        None, description="One of the known countries, or null if unspecified."
    )

class AnalyzedQuery(BaseModel):
    filter: QueryFilter
    cleaned_query: str
```

`filter.model_dump(exclude_none=True)` yields the `{}`-or-`{"country": ...}` dict passed to
the store as a JSONB containment predicate (`metadata @> :filter`), served by the existing
GIN index. Promote `country` to a real column + b-tree index if it becomes high-cardinality
or hot.

### Component design (mirrors existing Strategy/registry patterns)

- New `QueryAnalyzer` component (`src/ai/query_analysis/`), swappable behind a base class:
  - `LlmQueryAnalyzer` — the structured-output call above (default).
  - `NoopQueryAnalyzer` — returns empty filter + original query (feature-flag off).
- Use a **cheaper/faster model** for analysis (e.g. `gemini-flash`), not the generation model.
  Add a second lightweight LLM role in `ServiceFactory` (see Step 4, provider split).
- Thread an optional `filters: dict | None` through
  `VectorStoreFacade.search → PgVectorStore.search → ChunkRepository.search`, applied as an
  extra `.where(Chunk.extra_metadata.op("@>")(filters))` **only when non-empty**. Everything
  downstream (`min_similarity`, rerank, MMR) is untouched.

### Safety rules

- **Validate against a closed vocabulary** before filtering. If the LLM returns `"French"`
  but stored values are `"France"`, a strict-equality filter silently returns zero rows —
  check the extracted value against known distinct values; if no match, treat as no filter.
- **Fail open.** On extraction error/timeout/malformed output, fall back to unfiltered
  full-corpus search. A missed filter degrades to today's behavior; a bad filter returns
  zero rows and looks like "no relevant documentation".
- **Don't over-strip** `cleaned_query` — filter phrasing out, semantic content (e.g.
  "Article 6") in. Needs its own eval (extraction accuracy + rewrite quality), reusing the
  `tests/eval` harness.
- Only worth it where the filterable field has **real coverage** in the corpus.

### Config sketch

```yaml
query_analysis:
  enabled: true
  provider: "gemini"
  model: "gemini-flash"          # cheaper than the generation model
  filterable_fields:
    country: ["France", "Germany", "Italy", "Spain"]
```

---

## Step 2 — Unify Chunk + Metadata Into One Object (anti-misalignment) + Pydantic Conversion

**Problem.** Document loaders return two index-aligned lists —
`texts: List[str]` and `metadatas: List[dict]` (`text_document_loader.py`,
`ocr_document_loader.py`, `llm_extractor_document_loader.py`). The invariant "`texts[i]`
belongs to `metadatas[i]`" is enforced only by convention. Any future transform that maps
over one list but not the other (dedup, filter, reorder) will silently attach the **wrong
metadata to a chunk**, which then gets embedded and permanently stored with the wrong
page/source — corrupting citations for every future query that retrieves it. This is why
loaders currently must build both lists in lockstep in a single loop.

**Fix.** Introduce a single Pydantic `DocumentChunk` carrying content + metadata together,
making misalignment structurally impossible.

```python
from typing import Optional
from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    source: str
    filename: Optional[str] = None
    page: Optional[int] = None
    language: Optional[str] = None
    loader: Optional[str] = None
    # open extension bag mapped to chunks.metadata JSONB
    extra: dict = {}

class DocumentChunk(BaseModel):
    content: str
    metadata: ChunkMetadata
```

### Scope of the refactor

- `BaseDocumentLoader.load_documents()` returns `List[DocumentChunk]` instead of
  `Tuple[List[str], List[dict]]`. Update all three loaders + `DocumentLoaderFacade`.
- `PgVectorStore.add_documents` / `CollectionWriter` / `source_grouping` consume
  `List[DocumentChunk]` (group by `chunk.metadata.source`).
- `ChunkMetadataMapper.to_columns` accepts `ChunkMetadata`; `to_metadata` returns one on read
  so the retrieval path is also typed end-to-end.
- Retrieval results (`PgVectorStore.search`) return typed objects (e.g. a
  `RetrievedChunk(content, metadata, score, embedding?)`) instead of loose dicts, so
  rerankers/MMR/`RagService` stop indexing `doc["content"]` / `doc["metadata"]`.

**Do this before or alongside Step 1** — the metadata-filtering plumbing is cleaner on a
typed metadata object than on loose dicts, and the `filterable_fields` map directly onto
`ChunkMetadata`/`extra` keys.

### Broader Pydantic conversion

Extend typed models to the API and config edges already in flight:
- API `Question`/`Answer` already use Pydantic — extend `Answer.sources` to a typed
  `Source` model (source, page, score) instead of `list[Dict]`.
- Consider a typed settings model over `config.yaml` (replacing raw `CONFIG[...]` dict
  access) for validation-at-load and autocomplete.

---

## Step 3 — Hybrid Search (semantic + BM25)

Current retrieval is semantic-only. Add lexical recall and fuse:

```text
semantic (pgvector cosine) + BM25 / ts_rank keyword search
  -> Reciprocal Rank Fusion
  -> BGE rerank -> final_k
```

Postgres can do the lexical side natively (`tsvector` + GIN), avoiding a second engine.
Learn: BM25, lexical vs semantic recall, RRF, when hybrid beats pure semantic (exact terms,
codes, names — common in legal text).

---

## Step 4 — Separate Provider Abstraction (embedding / generation / analysis / reranker)

Today one `llm.provider` selects both chat and embeddings. Split into independent roles:

- embedding provider
- generation provider
- query-analysis provider (Step 1 — should be a cheap model)
- reranker provider (already separate)

Learn modular AI architecture; unblocks using a small model for analysis and a strong one for
generation.

---

## Step 5 — Surface Retrieval Scores End-to-End

The vector `score` (`1 - cosine_distance`) is computed but not returned by the API, and the
BGE reranker score is discarded after sorting. Attach to each returned chunk:

- vector similarity score
- reranker score
- final rank position

Surface in the `/ask` response and logs. Useful for debugging, threshold tuning, eval.

---

## Step 6 — Stable, Deterministic Chunk IDs

Chunks have random UUIDs + a `chunk_index`. Add a deterministic human-readable ID:

```text
regulation_eu_2024_1689.pdf::page3::chunk5
```

Learn: traceability, citation systems, stable references across re-ingestion.

---

## Step 7 — Inline Source Citations

The API returns source metadata as a list; attach citations directly to answer spans:

```text
Answer ... [Source: regulation_eu_2024_1689.pdf, page 4]
```

Learn: grounded generation, explainability, answer traceability.

---

## Step 8 — Prompt Injection Defenses

Treat retrieved chunks as untrusted. Add context isolation / delimiting, instruction-hierarchy
prompting, and basic detection. Learn: prompt injection, secure prompting, retrieval security.

---

## Step 9 — Observability

Extend the existing `[TIMING]` instrumentation to structured logs of: retrieved chunk IDs,
similarity + reranker scores, token counts, provider, latency per stage, failures. Feeds
debugging and eval.

---

# Suggested Order

1. Unify chunk+metadata into `DocumentChunk` + Pydantic conversion (Step 2 — foundation)
2. Conditional metadata filtering (Step 1 — headline feature; rides on the typed metadata)
3. Surface retrieval scores (Step 5)
4. Separate provider abstraction (Step 4)
5. Hybrid search (Step 3)
6. Stable chunk IDs (Step 6)
7. Inline source citations (Step 7)
8. Prompt injection defenses (Step 8)
9. Observability (Step 9)

> Steps 1 and 2 are the user-prioritized block. Do Step 2 (or at least the loader-side
> `DocumentChunk`) first so Step 1's filter plumbing is built on typed metadata.

---

# Part 2 — Nice to Have

Valuable but not essential.

- **Conversation history** — memory, conversational retrieval, follow-up query rewriting.
- **Query rewriting** — expand vague questions before retrieval (complements Step 1's
  cleaned-query rewrite; here it's about vagueness, not filters).
- **Streaming responses** — progressive token output for UX.
- **Multi-query retrieval** — generate several search queries per question, merge results.
- **Context compression / contextual-compression retriever** — drop irrelevant sentences
  inside chunks before generation.
- **Parent-child retrieval** — retrieve small chunks, feed larger parent sections to the LLM.
- **Multiple retrieval strategies** — switch semantic / hybrid / keyword / reranked, compare
  on the eval set.
- **Multiple embedding models** — compare Gemini / OpenAI / Voyage / BGE / Nomic / Jina.
- **Approximate search** — pgvector HNSW instead of exact scan (no ANN index today); learn
  scalability trade-offs.
- **Advanced citation verification** — verify each generated claim is supported by a chunk.
- **Dashboard** — visualize retrieved chunks, scores, reranker output, final prompt/answer.

---

# Final Goal

Implementing Part 1 covers nearly every major modern-RAG component: ingestion, structure-aware
chunking, embeddings, vector databases, **query-driven metadata filtering**, hybrid retrieval,
reranking, grounded generation, evaluation, observability, and production-oriented
architecture.
