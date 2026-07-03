import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.ai.vector_store_service.pgvector.models import EmbeddingCollection


class CollectionRepository:
    """Persistence for embedding collections. Raises domain-neutral errors only."""

    def __init__(self, session: Session):
        self._session = session

    def get_by_name(self, name: str) -> EmbeddingCollection | None:
        stmt = select(EmbeddingCollection).where(EmbeddingCollection.name == name)
        return self._session.scalar(stmt)

    def require_by_name(self, name: str) -> EmbeddingCollection:
        collection = self.get_by_name(name)
        if collection is None:
            raise LookupError(f"Collection '{name}' does not exist.")
        return collection

    def get_or_create(
        self, name: str, provider: str, model: str, dimensions: int
    ) -> EmbeddingCollection:
        collection = self.get_by_name(name)
        if collection is not None:
            return collection
        collection = EmbeddingCollection(
            id=uuid.uuid4(),
            name=name,
            provider=provider,
            model=model,
            dimensions=dimensions,
        )
        self._session.add(collection)
        self._session.flush()
        return collection
