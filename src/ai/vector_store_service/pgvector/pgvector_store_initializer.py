import logging

from sqlalchemy import text

from src.ai.vector_store_service.base_vector_store_initializer import BaseVectorStoreInitializer
from src.ai.vector_store_service.pgvector.collection_validator import CollectionValidator
from src.ai.vector_store_service.pgvector.database import get_session
from src.ai.vector_store_service.pgvector.pgvector_store import PgVectorStore
from src.ai.vector_store_service.pgvector.repositories.chunk_repository import ChunkRepository
from src.ai.vector_store_service.pgvector.repositories.collection_repository import (
    CollectionRepository,
)

logger = logging.getLogger(__name__)


class PgVectorStoreInitializer(BaseVectorStoreInitializer):
    """Startup readiness for the pgvector backend.

    Verifies the database is reachable and, if the collection exists, that it matches
    the active embedder. Ingestion is an explicit command, so this never builds data;
    it only fails fast on misconfiguration and logs guidance when the store is empty.
    """

    def __init__(self, store: PgVectorStore):
        self._store = store
        self._validator = CollectionValidator()

    def initialize(self) -> None:
        embedder = self._store.embedder
        collection_name = self._store.collection_name

        with get_session() as session:
            session.execute(text("SELECT 1"))

            collection = CollectionRepository(session).get_by_name(collection_name)
            if collection is None:
                logger.warning(
                    "Collection '%s' not found. Run ingestion: "
                    "python -m src.ingestion --collection %s",
                    collection_name,
                    collection_name,
                )
                return

            self._validator.validate(collection, embedder)

            chunk_count = ChunkRepository(session).count(collection.id)
            if chunk_count == 0:
                logger.warning(
                    "Collection '%s' has no chunks. Run ingestion to populate it.",
                    collection_name,
                )
            else:
                logger.info(
                    "pgvector ready: collection '%s' with %d chunks.",
                    collection_name,
                    chunk_count,
                )
