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


# The Article-1 "Subject matter" chunk ranked ~#95 before footer removal + task_type
# embedding asymmetry; afterwards it ranks ~#22. This test guards that improved state:
# the answer chunk must rank well (< IMPROVED_RANK_CEILING) and the top results must be
# real content, not the page-footer boilerplate that used to dominate. Getting it fully
# inside default_k=20 is left to a future reranking step.
IMPROVED_RANK_CEILING = 40


def _is_pure_footer(content: str) -> bool:
    stripped = content.strip()
    return bool(re.fullmatch(r"\d+/144", stripped)) or stripped in (
        "ELI: http://data.europa.eu/eli/reg/2024/1689/oj",
    )


def test_article_1_chunk_ranks_well_and_footers_are_gone(ranking_store):
    """Guards the retrieval-quality improvement from ingestion cleanup + task_type.

    After dropping Header/Footer/PageNumber elements and embedding with asymmetric
    task_type, the Article-1 "Subject matter" chunk climbs from ~#95 to ~#22, and the
    former top-of-list page-footer boilerplate no longer appears. If this regresses
    (footers back on top, or the answer chunk ranks poorly again) the test fails.
    """
    default_k = CONFIG["llm"]["default_k"]

    results = ranking_store.search(QUESTION, k=2000)

    matches = [i for i, doc in enumerate(results) if ARTICLE_1_SNIPPET in doc["content"]]

    # Manual-inspection output (visible when run with `-s`).
    print(f"\n=== Retrieved {len(results)} chunks | Article-1 rank(s): {matches[:5]} "
          f"(0-indexed, default_k={default_k}) ===")
    for i, doc in enumerate(results[:20]):
        mark = "  <-- ARTICLE 1" if ARTICLE_1_SNIPPET in doc["content"] else ""
        print(f"[{i:>2}] p{doc['metadata'].get('page', '?')}: {doc['content'][:90]!r}{mark}")
    if matches:  # show the answer chunk even if it ranks past #20
        r = matches[0]
        print(f"\n--- Article-1 chunk (rank {r}) ---\n{results[r]['content'][:400]}")

    assert matches, "Article 1 chunk not found in the collection at all"

    article_1_rank = matches[0]
    assert article_1_rank < IMPROVED_RANK_CEILING, (
        f"Article 1 chunk ranks #{article_1_rank}, worse than the expected ~#22 "
        f"(ceiling {IMPROVED_RANK_CEILING}); retrieval quality regressed."
    )

    top = results[: default_k]
    footer_hits = [doc["content"][:40] for doc in top if _is_pure_footer(doc["content"])]
    assert not footer_hits, f"Page-footer boilerplate leaked into top-{default_k}: {footer_hits}"
