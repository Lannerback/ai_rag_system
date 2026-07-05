import pytest
from sqlalchemy import text

from src.ai.vector_store_service.pgvector.database import Database
from src.ingestion.rebuild import rebuild


def _scalar(sql: str):
    with Database.engine().connect() as connection:
        return connection.execute(text(sql)).scalar()


def test_rebuild_wipes_and_reinitializes_schema(db_available):
    # Seed a collection so we can prove the wipe actually removed data.
    with Database.engine().begin() as connection:
        connection.execute(
            text(
                "INSERT INTO embedding_collections (id, name, provider, model, dimensions) "
                "VALUES (gen_random_uuid(), 'rebuild_probe', 'fake', 'fake', 3)"
            )
        )
    assert _scalar("SELECT count(*) FROM embedding_collections WHERE name='rebuild_probe'") == 1

    rebuild()

    # Schema recreated, vector extension present, all tables empty.
    assert _scalar("SELECT extname FROM pg_extension WHERE extname='vector'") == "vector"
    assert _scalar("SELECT count(*) FROM embedding_collections") == 0
    assert _scalar("SELECT count(*) FROM documents") == 0
    assert _scalar("SELECT count(*) FROM chunks") == 0
    assert _scalar("SELECT version_num FROM alembic_version") == "0001"


def test_rebuild_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        rebuild()
