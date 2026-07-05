import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.ai.vector_store_service.pgvector.models import Document


class DocumentRepository:
    """Persistence for source documents within a collection."""

    def __init__(self, session: Session):
        self._session = session

    def get_by_source(self, collection_id: uuid.UUID, source: str) -> Document | None:
        stmt = select(Document).where(
            Document.collection_id == collection_id, Document.source == source
        )
        return self._session.scalar(stmt)

    def upsert(
        self, collection_id: uuid.UUID, source: str, checksum: str, metadata: dict
    ) -> Document:
        document = self.get_by_source(collection_id, source)
        if document is None:
            document = Document(
                id=uuid.uuid4(),
                collection_id=collection_id,
                source=source,
                checksum=checksum,
                doc_metadata=metadata,
            )
            self._session.add(document)
        else:
            document.checksum = checksum
            document.doc_metadata = metadata
        self._session.flush()
        return document

    def delete_by_source(self, collection_id: uuid.UUID, source: str) -> None:
        document = self.get_by_source(collection_id, source)
        if document is not None:
            self._session.delete(document)
            self._session.flush()

    def list_sources(self, collection_id: uuid.UUID) -> set[str]:
        stmt = select(Document.source).where(Document.collection_id == collection_id)
        return set(self._session.scalars(stmt).all())
