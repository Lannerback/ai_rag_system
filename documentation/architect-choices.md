# Architecture Choices

This document records **why** each technology and strategy was selected over its alternatives.
It is the rationale companion to `overview.md` (what the system is) and
`rag-system-architecture.md` (how it works). Every choice below is reflected in
`config.yaml` and the code paths named inline.

---

## 1. Retrieval Pipeline: three stages, not one

The `/ask` pipeline (`src/ai/rag_service.py:_retrieve`) is a funnel:

```text
wide vector recall (candidate_k=120)
  -> cosine-similarity floor (min_similarity=0.5)
  -> cross-encoder rerank (BGE, rerank_k=30)
  -> diversity selection (MMR, final_k=12)  [currently DISABLED -> top final_k]
  -> 12 chunks into the LLM prompt
```

**Why a funnel instead of a single top-k vector search?**
A bi-encoder vector search (stage 1) is cheap and high-recall but low-precision: it
embeds query and chunk *independently*, so it ranks by rough directional similarity and
lets off-topic near-neighbors leak in. Recovering precision cheaply means casting a
**wide net first** (120 candidates), then spending expensive, high-accuracy compute
(a cross-encoder) only on that small pool. This is the standard
retrieve-then-rerank architecture and it measurably outperformed flat vector search on
our gold set.

Config: `config.yaml` → `retrieval:` block.

| Knob | Value | Meaning |
|------|-------|---------|
| `candidate_k` | 120 | vector-recall pool size (stage 1) |
| `min_similarity` | 0.5 | cosine floor applied to the pool (stage 2) |
| `rerank_k` | 30 | kept after cross-encoder reranking (stage 3) |
| `final_k` | 12 | fed into the LLM context (stage 4) |

---

## 2. Similarity Metric: cosine

**Chosen: cosine similarity** (pgvector `cosine_distance`, `metric: "cosine"`).

- Semantic search cares about **direction** (meaning), not vector **magnitude**. Two
  passages that mean the same thing but differ in length/scale still point the same way.
- It is the metric the Gemini embedding model is trained/tuned against, so distances are
  calibrated.
- pgvector computes `cosine_distance = 1 - cosine_similarity`; the store converts it back
  to a similarity `score = 1.0 - distance` in `pgvector_store.py:_to_result` so downstream
  stages reason in `[-1, 1]` similarity space.

Euclidean (L2) was rejected: it is magnitude-sensitive and not the space these embeddings
are optimized for.

### The `min_similarity` threshold — why 0.5

`min_similarity: 0.5` drops vector-recall candidates whose cosine similarity to the query
is below 0.5, trimming off-topic tail noise before it reaches the reranker.

The value is **empirically bounded**, not arbitrary. On the EU AI Act gold set, first-hit
similarity scores span **0.705–0.822** (top-1 max 0.822). Therefore:

- `0.85` would drop **every** correct result — too aggressive.
- `0.5` sits safely below every gold hit while still cutting the irrelevant tail.
- Raising toward `~0.68` tightens filtering; re-run `tests/eval/run_eval` before changing it.

`0.0` (or omitting the key) disables the floor entirely.

---

## 3. Reranker: BGE, not FlashRank

Both are **local cross-encoders** (no external API). We shipped FlashRank first, measured
it, and replaced it with BGE.

### What went wrong with FlashRank on legal text

FlashRank (`ms-marco-MiniLM-L-12-v2`) **degraded** ranking versus plain vector search on
our corpus: **MRR 0.672 → 0.513** on the gold set. Reasons:

- **Training-domain mismatch.** MS MARCO MiniLM cross-encoders are trained on web
  search — short, informal, general-domain English question/passage pairs. The EU AI Act
  is the opposite distribution: long, dense, formal, cross-referential legal prose with
  domain-specific terminology. The model scores relevance confidently but *wrongly* on
  text far outside its training distribution.
- **Model capacity.** MiniLM-L-12 is a small, ONNX-quantized model optimized for CPU
  speed. That capacity is fine for short factual snippets but too shallow to resolve the
  fine-grained semantic distinctions between adjacent legal clauses.
- **Passage length.** MS MARCO passages are short; our chunks are up to 1200 chars of
  packed legal text, which the MiniLM context handles less gracefully.

Net effect: a reranker that actively reordered correct chunks *downward*.

### Why BGE (`BAAI/bge-reranker-v2-m3`) is better here

- **Larger, deeper model** (~568M params, XLM-RoBERTa backbone) → enough capacity to
  distinguish closely related legal clauses.
- **Multilingual** (100+ languages) and trained on diverse, longer-passage retrieval data,
  so formal/long/legal text is in-distribution rather than out-of-distribution.
- Result on the same gold set: recall and MRR **above** flat vector search, clearing the
  regression floors in `tests/test_retrieval_ranking.py`.

### FlashRank vs BGE at a glance

| Dimension | FlashRank (`ms-marco-MiniLM-L-12-v2`) | BGE (`BAAI/bge-reranker-v2-m3`) |
|-----------|----------------------------------------|----------------------------------|
| Runtime | `flashrank` (ONNX, CPU) | `sentence-transformers` `CrossEncoder` (torch, MPS/CUDA/CPU) |
| Size / depth | Small, quantized | ~568M params, XLM-RoBERTa |
| Language | English, web domain | Multilingual, formal-domain capable |
| Speed | Very fast | Slower (~9s warm inference over 120 pairs on MPS) |
| Accuracy on legal corpus | Degraded (MRR 0.513) | Best measured (above vector baseline) |
| Cold start | Fast model load | ~6s model load + HF cache-revalidation, once per process |

Both remain wired via the registry (`ServiceFactory.RERANKER_PROVIDERS`); switching is a
one-line `reranking.provider` change. `reranking.enabled: false` bypasses reranking
entirely.

**Mechanics** (why a cross-encoder is more accurate than the vector search that fed it):
a bi-encoder embeds query and chunk separately, then compares vectors. A cross-encoder
feeds `(query, chunk)` **together** through one transformer forward pass with full
cross-attention, producing a single relevance logit. Far more accurate, far more
expensive — which is exactly why it runs only on the 120-candidate pool, not the whole
corpus. Implementation: `src/ai/rerankers/bge/bge_reranker.py`.

---

## 4. Diversity: MMR implemented but disabled

Maximal Marginal Relevance is implemented (`src/ai/diversity/mmr_selector.py`) but the
`diversity:` block in `config.yaml` is **commented out**, so the pipeline currently returns
the top `final_k` reranked chunks directly.

**What MMR does when enabled.** It greedily builds the final set, at each step maximizing:

```text
score = lambda * relevance(doc) - (1 - lambda) * max_cosine_sim(doc, already_selected)
```

- `relevance` is derived from the reranker's ordering (linear rank → `[0,1]`).
- the redundancy term penalizes a candidate for being too similar to already-picked chunks.
- `lambda_mult = 0.6` (relevance-leaning: `1.0` = pure relevance, `0.0` = pure diversity).
- It is the **only** stage that consumes chunk embeddings, so `_retrieve` fetches
  embeddings from pgvector (`with_embeddings=True`) *only* when MMR is active.

**Why disabled now.** After the chunking-quality fixes (element cleaning + legal section
tagging), the reranked top-12 were already diverse and on-topic; adding MMR's diversity
penalty risked demoting genuinely relevant, closely-related legal clauses (which *should*
co-occur in an answer about one Article) in favor of spurious variety. It provided no recall
gain on the gold set and added embedding-fetch cost, so it was switched off pending a corpus
where redundancy is an actual problem. Re-enable by uncommenting the `diversity:` block.

---

## 5. PDF Reader Strategy: `fast`, not `hi_res`

Config: `document_loader.pdf_strategy: "fast"`. PDFs are partitioned by
`unstructured.partition_pdf` (`src/ai/document_loaders/text_document_loader.py`).

| Strategy | How | Cost | Outcome |
|----------|-----|------|---------|
| `hi_res` | layout model + OCR; reconstructs two-column reading order | needs poppler + tesseract; **~8 min/doc** | correct column order, but **no measurable retrieval gain** |
| `fast` | pdfminer text layer | seconds/doc | chosen |

`hi_res` was trialed for correct two-column reading-order reconstruction on legal PDFs, but
it was **too slow (~8 min/doc) and did not improve retrieval quality** enough to justify the
cost. We reverted to `fast` and instead solved the layout-noise problem in cleanup (§7)
rather than at parse time.

---

## 6. Chunking: section-aware (`chunk_by_title`), not fixed-size

**Old:** `RecursiveCharacterTextSplitter`, `chunk_size=500`, `overlap=60` — blind
character-count splitting that cut mid-clause.

**New:** Unstructured `chunk_by_title` (`text_document_loader.py`), `chunk_size=1200`,
`overlap=150`, which starts a new chunk at each detected title/heading and merges small
fragments:

- `max_characters = 1200` — hard cap per chunk.
- `new_after_n_chars = 0.8 * chunk_size` (960) — soft cap; prefer breaking here.
- `combine_text_under_n_chars = 0.35 * chunk_size` (420) — merge tiny sections upward.
- `overlap = 150`, `multipage_sections = True`.

**Why:** legal documents are structured by Article/section. Splitting on titles keeps a
clause with its heading and neighbors, producing self-contained retrievable units instead
of arbitrary character windows. The larger 1200-char size keeps whole clauses intact.

---

## 7. Ingestion Cleanup: two legal-aware passes

Because we chose `fast` parsing (§5), layout noise must be removed in code. Two dedicated,
single-responsibility passes run **before** and **after** chunking:

- **`ElementCleaner`** (`element_cleaner.py`) — runs on raw partitioned elements (where
  category labels still exist). Drops `Header`/`Footer`/`PageNumber`/`PageBreak`/`Image`
  elements, standalone recital/footnote number markers like `(9)`, running-header leaks
  (`(Artificial Intelligence Act)`), ELI permalink footers, and fragments under 15 chars.
  Embedding this chrome would create meaningless near-duplicate vectors that dilute
  precision.
- **`LegalSectionTagger`** (`legal_section_tagger.py`) — runs after chunking on PDFs.
  `chunk_by_title` only keeps the `Article N` heading on the *first* chunk of a long
  article; continuation chunks read as orphaned clauses (`"1. interface tools..."`) that
  embed poorly. The tagger tracks the current Article and prepends it to continuation
  chunks so **every** chunk is a self-contained, anchored legal unit.

Guarded by `tests/test_element_cleaner.py`, `tests/test_legal_section_tagger.py`, and the
footer-leak assertion in `tests/test_retrieval_ranking.py`.

---

## 8. Embedding Strategy: asymmetric task types

Embedder: Gemini `models/gemini-embedding-001`, dimension **3072**
(`src/ai/embedders/gemini/gemini_embedder.py`).

**Key choice — asymmetric embedding:** documents and queries are embedded with *different*
`task_type` values:

- documents → `RETRIEVAL_DOCUMENT`
- queries → `RETRIEVAL_QUERY`

Gemini maps each role into aligned but role-specific subspaces, which materially improves
retrieval quality over embedding both sides identically. Requests are batched (100/req) and
wrapped in exponential-backoff retry (`tenacity`) against transient Gemini errors.

A pgvector collection is permanently bound to `provider + model + dimension`
(`CollectionValidator`); mismatches fail fast at startup and ingestion, preventing corrupt
retrieval from mixing incompatible vector spaces.

---

## 9. Web Framework: FastAPI

- Pydantic request validation out of the box (`Question`/`Answer` models in `src/api.py`).
- Native async on ASGI; runs under uvicorn with no adapter.
- Lightweight for a pure JSON API (no templating overhead), unlike Django.

---

## 10. Design Patterns

The codebase enforces layer isolation and swap-by-config via:

- **Abstract Factory + Registry** — `ServiceFactory` maps config strings to provider
  classes for LLM, embedder, vector store, reranker, and diversity selector. Adding a
  backend is one registry entry; selection is a dict lookup, never a branch.
- **Facade** — `RagFacade`, `VectorStoreFacade` present a thin stable surface over the
  services.
- **Singleton** — `ServiceFactory._instances` caches each service for the process lifetime
  (the BGE model is additionally lazy-loaded on first rerank, then cached).
- **Strategy** — interchangeable rerankers, diversity selectors, vector backends, and
  document loaders behind shared base classes.

---

## Decision Summary

| Concern | Chosen | Rejected alternative | Why |
|---------|--------|----------------------|-----|
| Retrieval shape | 3-stage funnel | flat top-k | precision recovery on wide recall |
| Distance metric | cosine | Euclidean/L2 | direction = meaning; embedder-calibrated |
| Similarity floor | 0.5 | 0.0 / 0.85 | below all gold hits, trims tail noise |
| Reranker | BGE v2-m3 | FlashRank MiniLM | legal-domain accuracy (MRR 0.672 vs 0.513) |
| Diversity | MMR (disabled) | MMR always-on | no recall gain post-cleanup; extra cost |
| PDF reader | `fast` | `hi_res` | 8 min/doc for no quality gain |
| Chunking | `chunk_by_title` 1200/150 | fixed 500/60 splitter | section-aware, whole clauses |
| Cleanup | ElementCleaner + LegalSectionTagger | none | remove layout noise, anchor clauses |
| Embedding | Gemini 3072, asymmetric task_type | symmetric embedding | role-specific subspaces boost recall |
| Framework | FastAPI | Flask/Django | async, pydantic, lightweight |
