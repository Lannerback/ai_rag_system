import uuid
from typing import List, Tuple

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, joinedload

from src.ai.vector_store_service.pgvector.models import Chunk


class ChunkRepository:
    """Persistence and similarity search for chunk embeddings."""

    def __init__(self, session: Session):
        self._session = session

    def add_many(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        self._session.add_all(chunks)
        self._session.flush()

    def delete_by_document(self, document_id: uuid.UUID) -> None:
        self._session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        self._session.flush()

    def count(self, collection_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(Chunk).where(Chunk.collection_id == collection_id)
        return self._session.scalar(stmt) or 0

    def search(
        self, collection_id: uuid.UUID, query_embedding: List[float], k: int
    ) -> List[Tuple[Chunk, str]]:
        """Return the k nearest chunks (by cosine distance) with their document source."""
        stmt = (
            select(Chunk)
            .options(joinedload(Chunk.document))
            .where(Chunk.collection_id == collection_id)
            .order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(k)
        )
        chunks = self._session.scalars(stmt).all()
        return [(chunk, chunk.document.source) for chunk in chunks]
