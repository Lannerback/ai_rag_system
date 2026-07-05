# tests/conftest.py
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables from the project-root .env file
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

# Default to the local pgvector instance when DATABASE_URL is not otherwise set.
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://admin:admin@localhost:5433/rag")

# Isolated Postgres schema holding cached test fixtures (chunks + embeddings), kept
# separate from `public` so real data and test data can never clobber each other.
TEST_SCHEMA = "test_rag"


def pytest_addoption(parser):
    parser.addoption(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Wipe and regenerate cached test embeddings in the test_rag schema.",
    )


def _database_reachable() -> bool:
    try:
        from sqlalchemy import text
        from src.ai.vector_store_service.pgvector.database import Database

        with Database.engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def db_available():
    """Skip pgvector-dependent tests when the database is unreachable."""
    if not _database_reachable():
        pytest.skip("pgvector database not available at DATABASE_URL")


@pytest.fixture
def clean_db(db_available):
    """Truncate all pgvector tables before each test for isolation."""
    from sqlalchemy import text
    from src.ai.vector_store_service.pgvector.database import Database

    with Database.engine().begin() as connection:
        connection.execute(
            text("TRUNCATE chunks, documents, embedding_collections RESTART IDENTITY CASCADE")
        )
    yield


@pytest.fixture
def test_schema_db(db_available, pytestconfig):
    """Route the pgvector Database singleton at an isolated `test_rag` schema.

    Uses SQLAlchemy `schema_translate_map` so all table refs (DDL + queries) resolve
    to `test_rag.*`, while the pgvector `vector` type keeps resolving from `public`.
    The schema and its rows persist across runs so embeddings are cached; passing
    `--force-rebuild` drops the schema first to regenerate from scratch. The singleton
    is restored on teardown so other tests keep hitting `public`.
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    from src.ai.vector_store_service.pgvector.database import Database
    from src.ai.vector_store_service.pgvector.models import Base

    engine = create_engine(
        os.environ["DATABASE_URL"], pool_pre_ping=True, future=True
    ).execution_options(schema_translate_map={None: TEST_SCHEMA})

    with engine.begin() as connection:
        if pytestconfig.getoption("--force-rebuild"):
            connection.execute(text(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE"))
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {TEST_SCHEMA}"))
    Base.metadata.create_all(engine)

    original_engine = Database._engine
    original_factory = Database._session_factory
    Database._engine = engine
    Database._session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield engine
    finally:
        Database._engine = original_engine
        Database._session_factory = original_factory
        engine.dispose()
