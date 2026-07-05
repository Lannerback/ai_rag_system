"""Measure retrieval recall/precision on the EU AI Act gold set.

Usage (needs local pgvector DB + GOOGLE_API_KEY):

    uv run python -m tests.eval.run_eval [collection_name]

Defaults to the `ranking_fixture` collection in the active database/schema.
Reports two pipelines side by side:
  1. RAW      - plain vector search (recall ceiling of the index)
  2. PIPELINE - vector recall -> FlashRank rerank -> MMR (what /ask serves)
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://admin:admin@localhost:5433/rag")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from src.ai.service_factory import ServiceFactory  # noqa: E402
from src.ai.vector_store_service.pgvector.database import Database  # noqa: E402
from src.ai.vector_store_service.pgvector.pgvector_store import PgVectorStore  # noqa: E402
from src.common.config import CONFIG  # noqa: E402
from tests.eval.metrics import evaluate, load_gold  # noqa: E402


def _route_schema(schema: str) -> None:
    """Point the pgvector Database singleton at a non-default schema (e.g. test_rag)."""
    if not schema:
        return
    engine = create_engine(
        os.environ["DATABASE_URL"], pool_pre_ping=True, future=True
    ).execution_options(schema_translate_map={None: schema})
    Database._engine = engine
    Database._session_factory = sessionmaker(bind=engine, expire_on_commit=False)


def main() -> None:
    collection = sys.argv[1] if len(sys.argv) > 1 else "ranking_fixture"
    schema = sys.argv[2] if len(sys.argv) > 2 else "test_rag"
    _route_schema(schema)
    gold = load_gold()
    embedder = ServiceFactory.get_embedder()
    store = PgVectorStore(embedder, collection)
    reranker = ServiceFactory.get_reranker()
    mmr = ServiceFactory.get_diversity_selector()

    retrieval = CONFIG["retrieval"]
    lambda_mult = CONFIG["diversity"]["lambda_mult"]

    def raw_retrieve(question: str):
        return store.search(question, k=retrieval["candidate_k"])

    def pipeline_retrieve(question: str):
        cands = store.search(question, k=retrieval["candidate_k"], with_embeddings=True)
        if reranker is not None:
            cands = reranker.rerank(question, cands, top_k=retrieval["rerank_k"])
        return mmr.select(cands, top_k=retrieval["final_k"], lambda_mult=lambda_mult)

    print(f"Collection: {collection} | schema: {schema or 'public'} | gold queries: {len(gold)}\n")
    raw = evaluate(gold, raw_retrieve)
    print(raw.format(f"RAW vector search (k={retrieval['candidate_k']})"))
    print()
    pipe = evaluate(gold, pipeline_retrieve)
    print(pipe.format(f"PIPELINE rerank+MMR (final_k={retrieval['final_k']})"))


if __name__ == "__main__":
    main()
