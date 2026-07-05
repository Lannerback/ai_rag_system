import uuid
from typing import Dict, List

from src.ai.embedders.base_embedder import BaseEmbedder
from src.ai.vector_store_service.pgvector.chunk_metadata_mapper import ChunkMetadataMapper
from src.ai.vector_store_service.pgvector.document_change_detector import DocumentChangeDetector
from src.ai.vector_store_service.pgvector.models import Chunk


class CollectionWriter:
    """Builds chunk rows (with embeddings) for a single document.

    Reusable write path shared by the runtime store and the ingestion command so
    embedding and row construction logic is never duplicated.
    """

    def __init__(self, embedder: BaseEmbedder):
        self._embedder = embedder
        self._mapper = ChunkMetadataMapper()
        self._detector = DocumentChangeDetector()

    def build_chunks(
        self,
        collection_id: uuid.UUID,
        document_id: uuid.UUID,
        texts: List[str],
        metadatas: List[Dict],
    ) -> List[Chunk]:
        embeddings = self._embedder.embed_documents(texts)
        dimension = self._embedder.dimension

        chunks: List[Chunk] = []
        for index, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            columns = self._mapper.to_columns(metadata)
            chunks.append(
                Chunk(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    collection_id=collection_id,
                    chunk_index=index,
                    content=text,
                    checksum=self._detector.checksum(text),
                    embedding=embedding,
                    embedding_dimensions=dimension,
                    page=columns["page"],
                    language=columns["language"],
                    loader=columns["loader"],
                    extra_metadata=columns["extra_metadata"],
                )
            )
        return chunks
