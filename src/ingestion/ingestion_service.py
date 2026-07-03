import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.ai.embedders.base_embedder import BaseEmbedder
from src.ai.document_loaders.document_loader_facade import DocumentLoaderFacade
from src.ai.vector_store_service.pgvector.collection_validator import CollectionValidator
from src.ai.vector_store_service.pgvector.collection_writer import CollectionWriter
from src.ai.vector_store_service.pgvector.database import get_session
from src.ai.vector_store_service.pgvector.document_change_detector import DocumentChangeDetector
from src.ai.vector_store_service.pgvector.repositories.chunk_repository import ChunkRepository
from src.ai.vector_store_service.pgvector.repositories.collection_repository import (
    CollectionRepository,
)
from src.ai.vector_store_service.pgvector.repositories.document_repository import DocumentRepository
from src.ai.vector_store_service.pgvector.source_grouping import document_metadata, group_by_source

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    created: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    failed: List[str] = field(default_factory=list)


class IngestionService:
    """Idempotent ingestion into a pgvector collection.

    Documents are grouped by source and compared by checksum: unchanged documents
    are skipped, changed ones re-embedded and replaced, and sources no longer present
    are pruned. Each document is replaced in its own transaction so a failure leaves
    the previous version intact.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        collection_name: str,
        document_loader_facade: DocumentLoaderFacade | None = None,
    ):
        self._embedder = embedder
        self._collection_name = collection_name
        self._loader = document_loader_facade or DocumentLoaderFacade()
        self._validator = CollectionValidator()
        self._detector = DocumentChangeDetector()
        self._writer = CollectionWriter(embedder)

    def ingest(self, rebuild: bool = False) -> IngestionReport:
        texts, metadatas = self._loader.load_all_documents()
        grouped = group_by_source(texts, metadatas)

        collection_id = self._ensure_collection()
        existing_sources = self._list_sources(collection_id)

        report = IngestionReport()
        for source, items in grouped.items():
            self._ingest_one(collection_id, source, items, rebuild, report)

        self._prune(collection_id, set(grouped.keys()), existing_sources, report)
        return report

    def _ensure_collection(self) -> uuid.UUID:
        with get_session() as session:
            collection = CollectionRepository(session).get_or_create(
                self._collection_name,
                self._embedder.provider,
                self._embedder.model,
                self._embedder.dimension,
            )
            self._validator.validate(collection, self._embedder)
            return collection.id

    def _list_sources(self, collection_id: uuid.UUID) -> set[str]:
        with get_session() as session:
            return DocumentRepository(session).list_sources(collection_id)

    def _ingest_one(
        self,
        collection_id: uuid.UUID,
        source: str,
        items: List[Tuple[str, Dict]],
        rebuild: bool,
        report: IngestionReport,
    ) -> None:
        texts = [text for text, _ in items]
        metadatas = [metadata for _, metadata in items]
        checksum = self._detector.checksum_many(texts)

        try:
            with get_session() as session:
                document_repo = DocumentRepository(session)
                chunk_repo = ChunkRepository(session)

                existing = document_repo.get_by_source(collection_id, source)
                if (
                    existing is not None
                    and not rebuild
                    and not self._detector.has_changed(existing.checksum, checksum)
                ):
                    report.skipped += 1
                    return

                document = document_repo.upsert(
                    collection_id, source, checksum, document_metadata(metadatas)
                )
                chunk_repo.delete_by_document(document.id)
                chunk_repo.add_many(
                    self._writer.build_chunks(collection_id, document.id, texts, metadatas)
                )

            if existing is None:
                report.created += 1
            else:
                report.updated += 1
        except Exception:
            logger.exception("Failed to ingest source '%s'; previous version preserved.", source)
            report.failed.append(source)

    def _prune(
        self,
        collection_id: uuid.UUID,
        present_sources: set[str],
        existing_sources: set[str],
        report: IngestionReport,
    ) -> None:
        if not present_sources:
            logger.warning("No source documents found; skipping prune to avoid data loss.")
            return

        removed_sources = existing_sources - present_sources
        if not removed_sources:
            return

        with get_session() as session:
            document_repo = DocumentRepository(session)
            for source in removed_sources:
                document_repo.delete_by_source(collection_id, source)
        report.removed += len(removed_sources)
