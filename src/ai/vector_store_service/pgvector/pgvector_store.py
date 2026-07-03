import logging
from typing import Dict, List, Tuple

from src.ai.embedders.base_embedder import BaseEmbedder
from src.ai.vector_store_service.base_vector_store import BaseVectorStore
from src.ai.vector_store_service.pgvector.chunk_metadata_mapper import ChunkMetadataMapper
from src.ai.vector_store_service.pgvector.collection_validator import CollectionValidator
from src.ai.vector_store_service.pgvector.collection_writer import CollectionWriter
from src.ai.vector_store_service.pgvector.database import get_session
from src.ai.vector_store_service.pgvector.document_change_detector import DocumentChangeDetector
from src.ai.vector_store_service.pgvector.models import EmbeddingCollection
from src.ai.vector_store_service.pgvector.repositories.chunk_repository import ChunkRepository
from src.ai.vector_store_service.pgvector.repositories.collection_repository import (
    CollectionRepository,
)
from src.ai.vector_store_service.pgvector.repositories.document_repository import DocumentRepository
from src.ai.vector_store_service.pgvector.source_grouping import document_metadata, group_by_source

logger = logging.getLogger(__name__)


class PgVectorStore(BaseVectorStore):
    """Vector store backed by PostgreSQL + pgvector, bound to one collection."""

    def __init__(self, embedder: BaseEmbedder, collection_name: str):
        self._embedder = embedder
        self._collection_name = collection_name
        self._mapper = ChunkMetadataMapper()
        self._validator = CollectionValidator()
        self._detector = DocumentChangeDetector()

    @property
    def embedder(self) -> BaseEmbedder:
        return self._embedder

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        if not texts:
            return
        metadatas = metadatas or [{} for _ in texts]

        with get_session() as session:
            collection = CollectionRepository(session).get_or_create(
                self._collection_name,
                self._embedder.provider,
                self._embedder.model,
                self._embedder.dimension,
            )
            self._validator.validate(collection, self._embedder)

            document_repo = DocumentRepository(session)
            chunk_repo = ChunkRepository(session)
            writer = CollectionWriter(self._embedder)

            for source, items in group_by_source(texts, metadatas).items():
                self._replace_document(
                    collection, source, items, document_repo, chunk_repo, writer
                )

    def search(self, query: str, k: int = 3) -> List[Dict]:
        with get_session() as session:
            collection = CollectionRepository(session).get_by_name(self._collection_name)
            if collection is None:
                logger.warning("Collection '%s' not found; returning no results.", self._collection_name)
                return []
            self._validator.validate(collection, self._embedder)

            query_embedding = self._embedder.embed_query(query)
            self._assert_query_dimension(query_embedding, collection)

            rows = ChunkRepository(session).search(collection.id, query_embedding, k)
            return [
                {"content": chunk.content, "metadata": self._mapper.to_metadata(chunk, source)}
                for chunk, source in rows
            ]

    def _replace_document(
        self,
        collection: EmbeddingCollection,
        source: str,
        items: List[Tuple[str, Dict]],
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        writer: CollectionWriter,
    ) -> None:
        texts = [text for text, _ in items]
        metadatas = [metadata for _, metadata in items]
        checksum = self._detector.checksum_many(texts)

        document = document_repo.upsert(
            collection.id, source, checksum, document_metadata(metadatas)
        )
        chunk_repo.delete_by_document(document.id)
        chunk_repo.add_many(writer.build_chunks(collection.id, document.id, texts, metadatas))

    @staticmethod
    def _assert_query_dimension(query_embedding: List[float], collection: EmbeddingCollection) -> None:
        if len(query_embedding) != collection.dimensions:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} does not match "
                f"collection '{collection.name}' dimension {collection.dimensions}."
            )
