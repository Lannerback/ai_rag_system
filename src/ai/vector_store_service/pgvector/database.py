import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


class Database:
    """Lazy singleton owning the SQLAlchemy engine and session factory.

    The connection string is read exclusively from the DATABASE_URL environment
    variable; no credentials live in application configuration.
    """

    _engine: Engine | None = None
    _session_factory: sessionmaker | None = None

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._engine is not None:
            return
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set; it is required for the pgvector backend."
            )
        cls._engine = create_engine(url, pool_pre_ping=True, future=True)
        cls._session_factory = sessionmaker(bind=cls._engine, expire_on_commit=False)

    @classmethod
    def engine(cls) -> Engine:
        cls._ensure_initialized()
        return cls._engine

    @classmethod
    def create_session(cls) -> Session:
        cls._ensure_initialized()
        return cls._session_factory()


@contextmanager
def get_session() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on error, always closes."""
    session = Database.create_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
