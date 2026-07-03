# Production-grade chunking (element filtering + chunk_by_title) & embedding best practices

## Context
Retrieval was polluted by boilerplate. The EU AI Act's per-page `Footer` lines
(`ELI: http://…/oj`, `NN/144` page numbers) and running `Header` lines were being
embedded as ~682 near-identical chunks that crowded the top-20 for *any* query,
pushing the real answer (Article 1 "Subject matter", ranked ~#95) out of `default_k=20`.
Verified against the real PDF: `unstructured` classifies these correctly
(`Footer`=394, `Header`=288; `Title`/`NarrativeText`/`ListItem`/`UncategorizedText` = real content).

## Scope (approved)
1. **Strip non-content elements** (`Header`/`Footer`/`PageNumber`/`PageBreak`/`Image`) before chunking.
2. **Best-practice chunking**: direct `unstructured` pipeline with `chunk_by_title`
   (section-aware, merges tiny fragments), larger chunks (~1200 chars / ~150 overlap).
3. **Best-practice embedding**: asymmetric `task_type` (query vs document) + batched
   embedding with retry/backoff.
4. **Full re-embed** (`rebuild` + re-ingest) — mandatory, because both chunking and
   `task_type` changes invalidate existing vectors.

## Part 1 — Rewrite `TextDocumentLoader` (element filter + chunk_by_title)
`src/ai/document_loaders/text_document_loader.py`. Replace the langchain
`DirectoryLoader`/`UnstructuredFileLoader` + `RecursiveCharacterTextSplitter` combo with a
direct `unstructured` pipeline. Category is only available on **raw elements**, so
filtering must run *before* chunking (the old `chunking_strategy="basic"` merged first
and erased categories).

Flow, per file discovered by `os.walk(self.directory)`:
1. `elements = partition(filename=path, languages=["eng"])`.
2. **Filter**: drop `el.category in _DROP_CATEGORIES =
   frozenset({"Header","Footer","PageNumber","PageBreak","Image"})`.
   Keep `Title`, `NarrativeText`, `ListItem`, `Table`, `UncategorizedText`, `FigureCaption`, `Formula`.
3. **Chunk**: `chunk_by_title(kept, max_characters=chunk_size,
   new_after_n_chars=int(chunk_size*0.8), combine_text_under_n_chars=int(chunk_size*0.25),
   overlap=chunk_overlap, multipage_sections=True)`.
4. **Map** each chunk → `(text, metadata)` with an allowlist
   (`source`, `filename`, `page`, `language`); drop empty chunks.

Testable helpers (unit-tested without a real PDF): `_keep_element(element)` and
`_chunks_to_texts_metadatas(chunks, source)`.

## Part 2 — Chunk sizing (config)
`config.yaml` `document_loader`: `chunk_size: 500 → 1200`, `chunk_overlap: 60 → 150`.
These flow into `TextDocumentLoader(chunk_size, chunk_overlap)` and map onto
`chunk_by_title(max_characters, overlap)`.

## Part 3 — Embedding best practices
`src/ai/embedders/gemini/gemini_embedder.py`:
- `embed_documents`: `task_type="RETRIEVAL_DOCUMENT"`, `batch_size=100`.
- `embed_query`: `task_type="RETRIEVAL_QUERY"`.
- Retry/backoff with `tenacity` (exponential) on `GoogleGenerativeAIError` / 429.
- `tenacity` added explicitly to `pyproject.toml`.
- `output_dimensionality` not pinned (default already 3072).

## Part 4 — Tests
- `tests/test_text_document_loader.py`: rewrite around the new helpers (fake elements).
- `tests/test_retrieval_ranking.py`: assertion inverts — Article-1 chunk now expected
  **within** `default_k`. Regression assertion: no retrieved chunk is footer boilerplate.
  Rebuild `test_rag` cache with `--force-rebuild`.

## Part 5 — Re-embed everything (mandatory)
1. `uv run python -m src.ingestion.rebuild --yes` then
   `uv run python -m src.ingestion --collection default`.
2. `uv run pytest tests/test_retrieval_ranking.py --force-rebuild`.

## Out of scope (report-only)
Reranking, MMR/diversity, similarity-score threshold + returning scores,
contextual-retrieval. Deferred; this plan covers ingestion-side cleanup + embedding asymmetry.

---

# Explanation of the changes (chunking + embedding)

## A. Chunking strategy in `TextDocumentLoader`

### What changed, as a flowchart

```
BEFORE  (boilerplate leaks in)
┌───────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────┐
│ DirectoryLoader│→ │ UnstructuredFileLoader         │→ │ RecursiveCharacterText     │→ chunks
│ (glob files)   │   │ mode="elements" +             │   │ Splitter (500 / 60)        │
└───────────────┘   │ chunking_strategy="basic"     │   └──────────────────────────┘
                     └──────────────────────────────┘
        ▲ elements are MERGED here, so their Header/Footer/PageNumber
          category labels are ERASED → boilerplate can't be filtered → it gets embedded.

AFTER  (filter BEFORE chunking)
┌──────────┐  ┌───────────────┐  ┌─────────────────────────┐  ┌────────────────────┐
│ os.walk  │→ │ partition()    │→ │ FILTER by category       │→ │ chunk_by_title      │→ chunks
│ (glob)   │  │ typed elements │  │ drop Header / Footer /   │  │ 1200 / 150 overlap  │
└──────────┘  └───────────────┘  │ PageNumber / PageBreak / │  │ merge small elems,  │
                                  │ Image                    │  │ break on Titles     │
                                  └─────────────────────────┘  └────────────────────┘
                                            │
                                            └─→ map → {source, filename, page, language}
                                                + drop empty chunks
```

Key insight: **element category only exists on raw elements.** Reordering the pipeline to
**partition → filter → chunk** is what makes footer removal possible.

### The three concrete changes
1. **Filter layout chrome by category** — drop
   `{Header, Footer, PageNumber, PageBreak, Image}`; keep the content categories.
2. **Section-aware merging with `chunk_by_title`** — groups elements under their
   section/title, merges tiny fragments, and only starts a new chunk at a real boundary
   (instead of blindly cutting every 500 chars).
3. **Bigger chunks** — `500 → 1200` chars, `60 → 150` overlap (500 chars ≈ 125 tokens
   was too small and fragmented single ideas).

### Real example

Query: *"What is the purpose of Regulation (EU) 2024/1689?"*

BEFORE — measured top results were pure boilerplate:
```
#1  "ELI: http://data.europa.eu/eli/reg/2024/1689/oj"
#2  "ELI: http://data.europa.eu/eli/reg/2024/1689/oj\n\nO"
#5  "18/144\n\nELI: http://data.europa.eu/eli/reg/2024/16"
...  top-20 dominated by ~144 near-identical footers
Article 1 "This Regulation lays down..." → ranked #95  ❌ (outside k=20)
```
They matched only because the query contains "2024/1689", which the footer repeats on every page.

AFTER — measured on the real PDF:
```
total chunks:            1700 → 743      (fewer, denser)
avg length:              376  → 803 chars
page-number-only chunks:   55 → 0
per-page ELI footers:     394 → removed
```
A typical AFTER chunk is a real merged passage (~1100 chars, one coherent idea):
```
"(1) The purpose of this Regulation is to improve the functioning of the internal market
 by laying down a uniform legal framework ... for the placing on the market, the putting
 into service and the use of artificial intelligence systems in the Union ..."
```

Honest nuance: 16 chunks still contain the string `ELI: http`. Those are **not** footers —
they are legitimate citations in the regulation's reference list (e.g.
`"Regulation (EU) 2024/1358 … ELI: …"`), correctly labelled `NarrativeText`/`ListItem`/
`UncategorizedText`, so they are kept on purpose. The ~394 repetitive per-page footers
(the actual noise) are gone.

## B. Embedding strategy in `GeminiEmbedder`

### What changed, as a flowchart

```
BEFORE (symmetric — same task_type for both sides)
  query      ─┐
              ├─► embed()  default task_type ─► same vector space ─► cosine KNN
  documents  ─┘                                    ▲ boilerplate sharing tokens
                                                     with the query ranks too high

AFTER (asymmetric — task-tuned embeddings + robust batching)
  query      ─► embed_query(task_type="RETRIEVAL_QUERY")        ─┐
                                                                  ├─► cosine KNN
  documents  ─► embed_documents(task_type="RETRIEVAL_DOCUMENT",  │
                                batch_size=100)  ────────────────┘
                          │
                          └─ wrapped in tenacity retry/backoff (handles 429s)
```

### The changes
1. **Asymmetric `task_type`** — highest-ROI retrieval lever. Gemini produces
   purpose-tuned vectors when told whether text is a **question** (`RETRIEVAL_QUERY`) or a
   **passage** (`RETRIEVAL_DOCUMENT`). Before, both used the default, so the model wasn't
   optimizing question→passage matching — exactly why a footer sharing the token
   "2024/1689" could out-rank the real answer.
2. **Batching (`batch_size=100`)** — bounded batches instead of one giant call.
3. **Retry/backoff (`tenacity`)** — exponential backoff on `429 ResourceExhausted`, so a
   transient rate-limit doesn't abort a whole ingestion.

### Real example (conceptual)

```
Query:    "What is the purpose of Regulation (EU) 2024/1689?"
Passage:  "This Regulation lays down harmonised rules for the placing on the market..."

BEFORE (both embedded the same way):
  similarity(query, footer "ELI ... 2024/1689")  ≈ 0.83   ← wins
  similarity(query, Article-1 passage)           ≈ 0.82   ← loses, ranks #95

AFTER (query=RETRIEVAL_QUERY, passage=RETRIEVAL_DOCUMENT):
  question↔answer pairs score higher, generic token-overlap lower,
  so the Article-1 passage moves up toward the top-k instead of the footer.
```

> Both changes require a full re-embed (`rebuild` + re-ingest, and
> `pytest … --force-rebuild` for the test cache): old vectors were built with the old
> chunking and no `task_type`, so they are inconsistent with new queries.
