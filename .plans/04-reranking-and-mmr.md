# Reranking + MMR Diversity (pgvector-first)

## Goal
Increase RAG accuracy/recall by adding a two-stage post-retrieval pipeline:
vector search (wide recall) -> cross-encoder reranking -> MMR diversity selection.
Fully abstracted so future rerankers (Cohere/BGE), diversity strategies, and vector
stores (FAISS) plug in via registries with no pipeline changes.

## Locked Decisions
- Reranker: **FlashRank** (local, CPU, no external API key).
- Diversity: **MMR** applied **after** reranking.
- Backend focus: **pgvector only** (FAISS stub raises NotImplementedError).
- MMR vectors: **extend `search()` contract** with `with_embeddings` flag to return the
  ORIGINAL ingestion vectors (correct, cheap; no re-embed, preserves Gemini task_type asymmetry).
- k-cascade: **candidate_k=120 -> rerank_k=30 -> final_k=12**.
- No feature flags: reranking + MMR are **always-on / mandatory**.

## Core Constraint
`BaseVectorStore.search()` returns `List[Dict]` of `{"content", "metadata"}` — no vectors/scores.
- Reranking needs only text (content) -> works as-is.
- MMR needs vectors -> requires the contract extension (`with_embeddings=True`).

## Pipeline (RagService.answer_question) — no branches
```
search(question, k=120, with_embeddings=True)
  -> reranker.rerank(question, docs, top_k=30)     # preserves "embedding" key
  -> mmr.select(docs, top_k=12, lambda_mult=0.6)
  -> context(12 docs) -> LLM
```

## config.yaml additions
```yaml
retrieval:
  candidate_k: 120
  rerank_k: 30
  final_k: 12
reranking:
  provider: "flashrank"
  flashrank:
    model: "ms-marco-MiniLM-L-12-v2"
diversity:
  strategy: "mmr"
  lambda_mult: 0.6
```

## NEW files
- src/ai/rerankers/base_reranker.py            # ABC rerank(query, docs, top_k) -> List[Dict]
- src/ai/rerankers/reranker_provider.py        # Abstract Factory + registry
- src/ai/rerankers/flashrank/__init__.py
- src/ai/rerankers/flashrank/flashrank_reranker.py  # lazy-imports flashrank, preserves embedding
- src/ai/diversity/base_diversity_selector.py  # ABC select(docs, top_k, lambda_mult) -> List[Dict]
- src/ai/diversity/__init__.py
- src/ai/diversity/mmr_selector.py             # greedy MMR, numpy cosine
- tests/test_flashrank_reranker.py             # monkeypatched scores, no model download
- tests/test_mmr_selector.py                   # deterministic vectors, lambda extremes

## MODIFIED files
- src/ai/rag_service.py                         # inject reranker+mmr; 3-stage pipeline
- src/ai/service_factory.py                     # get_reranker + get_diversity_selector + registry
- src/ai/vector_store_service/base_vector_store.py   # search(..., with_embeddings=False)
- src/ai/vector_store_service/vector_store_facade.py # pass-through
- src/ai/vector_store_service/pgvector/pgvector_store.py       # return embedding when requested
- src/ai/vector_store_service/pgvector/chunk_metadata_mapper.py# (embedding handled in store)
- src/ai/vector_store_service/faiss/faiss_vector_store.py      # accept param, NotImplementedError
- config.yaml
- tests/test_retrieval_ranking.py              # assert answer chunk within final_k=12
- pyproject.toml / requirements.txt            # add flashrank

## Abstraction guarantees
- New reranker: 1 file + 1 registry entry.
- New diversity strategy: 1 file + 1 registry entry.
- FAISS later: implement with_embeddings=True (NotImplementedError is the explicit marker).
- Reranker/MMR consume+emit neutral List[Dict] -> zero coupling to store/LLM.

## Quality gate
- ruff lint on src/ only (exclude tests).
- full pytest suite green before done.
