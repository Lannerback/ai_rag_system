from sqlalchemy import text

from src.ai.vector_store_service.pgvector.database import Database


def _scalar_set(sql: str) -> set[str]:
    with Database.engine().connect() as connection:
        return {row[0] for row in connection.execute(text(sql))}


def test_vector_extension_present(db_available):
    assert _scalar_set("SELECT extname FROM pg_extension WHERE extname = 'vector'") == {"vector"}


def test_tables_present(db_available):
    tables = _scalar_set("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    assert {"embedding_collections", "documents", "chunks"} <= tables


def test_constraints_present(db_available):
    constraints = _scalar_set("SELECT conname FROM pg_constraint")
    assert "ck_chunks_embedding_dims" in constraints
    assert "uq_chunks_document_index" in constraints
    assert "uq_documents_collection_source" in constraints


def test_indexes_present(db_available):
    indexes = _scalar_set("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")
    assert "ix_chunks_metadata_gin" in indexes
    assert "ix_chunks_collection_id" in indexes
    assert "ix_chunks_document_id" in indexes
