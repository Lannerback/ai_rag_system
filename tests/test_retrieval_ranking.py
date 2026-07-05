"""Retrieval-ranking regression test, backed by a cached `test_rag` schema.

How to run (needs the local pgvector DB up and a live GOOGLE_API_KEY):

    # Normal run: builds chunks+embeddings once into the `test_rag` schema on the
    # first run (~44s), then reuses the cache on every later run (~6s). Never
    # touches the real `public.default` collection.
    uv run pytest tests/test_retrieval_ranking.py

    # Force rebuild: drops and recreates the `test_rag` schema, re-parses the PDF
    # and re-embeds all chunks from scratch (~44s). Use after changing the loader,
    # chunking config, or embedding model so the cache reflects the new pipeline.
    uv run pytest tests/test_retrieval_ranking.py --force-rebuild

    # Verbose variants (add -v / -s to either command above for per-test output):
    uv run pytest tests/test_retrieval_ranking.py -v
    uv run pytest tests/test_retrieval_ranking.py --force-rebuild -sv

Without GOOGLE_API_KEY the test is skipped; without the DB it is skipped via the
`db_available` fixture. See `test_schema_db` / `--force-rebuild` in tests/conftest.py.
"""
import os
import re

import pytest

from src.ai.document_loaders.text_document_loader import TextDocumentLoader
from src.ai.service_factory import ServiceFactory
from src.ai.vector_store_service.pgvector.database import get_session
from src.ai.vector_store_service.pgvector.pgvector_store import PgVectorStore
from src.ai.vector_store_service.pgvector.repositories.chunk_repository import ChunkRepository
from src.ai.vector_store_service.pgvector.repositories.collection_repository import (
    CollectionRepository,
)
from src.common.config import CONFIG

ARTICLE_1_SNIPPET = "harmonised rules for the placing on the market"
QUESTION = "What is the purpose of Regulation (EU) 2024/1689?"
TEST_COLLECTION = "ranking_fixture"

pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="requires GOOGLE_API_KEY for real Gemini embeddings",
)


def _has_cached_chunks(collection_name: str) -> bool:
    with get_session() as session:
        collection = CollectionRepository(session).get_by_name(collection_name)
        if collection is None:
            return False
        return ChunkRepository(session).count(collection.id) > 0


@pytest.fixture
def ranking_store(test_schema_db):
    """Retrieval store backed by the cached `ranking_fixture` collection in `test_rag`.

    Builds the chunks + embeddings from the real EU AI Act PDF only on a cache miss
    (first run, or after `--force-rebuild` drops the schema). On a cache hit it reuses
    the persisted rows, skipping the PDF parse and document embedding entirely.
    """
    embedder = ServiceFactory.get_embedder()
    store = PgVectorStore(embedder, TEST_COLLECTION)

    if not _has_cached_chunks(TEST_COLLECTION):
        loader_config = CONFIG["document_loader"]
        loader = TextDocumentLoader(
            directory=loader_config["docs_directory"],
            chunk_size=loader_config["chunk_size"],
            chunk_overlap=loader_config["chunk_overlap"],
        )
        texts, metadatas = loader.load_documents()
        store.add_documents(texts, metadatas)

    return store


# The Article-1 "Subject matter" chunk must remain retrievable, and no page-footer
# boilerplate may leak into the top results after ingestion cleanup + fast parsing.
_TOP_N_FOR_FOOTER_CHECK = 20


def _is_pure_footer(content: str) -> bool:
    stripped = content.strip()
    return (
        bool(re.fullmatch(r"\d+/144", stripped))
        or stripped.lower().startswith("eli: http")
    )


def test_answer_chunk_retrievable_and_footers_absent(ranking_store):
    """Guards ingestion quality: the Article-1 answer chunk is retrievable and page
    footers (ELI permalinks, 'N/144' page markers) never leak into the top results."""
    results = ranking_store.search(QUESTION, k=2000)

    matches = [i for i, doc in enumerate(results) if ARTICLE_1_SNIPPET in doc["content"]]
    assert matches, "Article 1 chunk not found in the collection at all"

    top = results[:_TOP_N_FOR_FOOTER_CHECK]
    footer_hits = [doc["content"][:40] for doc in top if _is_pure_footer(doc["content"])]
    assert not footer_hits, (
        f"Page-footer boilerplate leaked into top-{_TOP_N_FOR_FOOTER_CHECK}: {footer_hits}"
    )


# Quality floors for the gold set on the current pipeline (measured: R@10=1.0, R@12=1.0,
# MRR~0.585 with fast parsing + BGE reranker + min_similarity=0.5, MMR disabled).
# Guards against chunking/retrieval regressions.
GOLD_RECALL_AT_10_FLOOR = 0.9
GOLD_MRR_FLOOR = 0.55


def test_gold_set_recall_meets_floor(ranking_store):
    """Regression guard on retrieval quality using the verified EU AI Act gold set.

    Runs the same retrieval the /ask pipeline uses (vector recall -> optional rerank
    -> MMR) and asserts recall@10 and MRR stay above the floors established after the
    chunking-quality fixes. A drop here means ingestion/chunking regressed.
    """
    from tests.eval.metrics import evaluate, load_gold

    retrieval = CONFIG["retrieval"]
    reranker = ServiceFactory.get_reranker()
    mmr = ServiceFactory.get_diversity_selector()
    min_similarity = retrieval.get("min_similarity", 0.0)

    # Mirrors RagService._retrieve: vector recall -> similarity floor ->
    # (optional rerank) -> (optional MMR, else top final_k).
    def retrieve(question: str):
        candidates = ranking_store.search(
            question, k=retrieval["candidate_k"], with_embeddings=mmr is not None
        )
        if min_similarity > 0.0:
            candidates = [c for c in candidates if c.get("score", 1.0) >= min_similarity]
        if reranker is not None:
            candidates = reranker.rerank(question, candidates, top_k=retrieval["rerank_k"])
        if mmr is None:
            return candidates[:retrieval["final_k"]]
        return mmr.select(
            candidates, top_k=retrieval["final_k"], lambda_mult=CONFIG["diversity"]["lambda_mult"]
        )

    report = evaluate(load_gold(), retrieve)
    print("\n" + report.format("GOLD SET (pipeline)"))

    assert report.recall_at(retrieval["final_k"]) >= GOLD_RECALL_AT_10_FLOOR, (
        f"gold recall@{retrieval['final_k']} = {report.recall_at(retrieval['final_k']):.2f} "
        f"below floor {GOLD_RECALL_AT_10_FLOOR}; chunking/retrieval regressed."
    )
    assert report.mrr() >= GOLD_MRR_FLOOR, (
        f"gold MRR = {report.mrr():.3f} below floor {GOLD_MRR_FLOOR}; retrieval regressed."
    )
