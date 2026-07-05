import pytest

from src.ai.vector_store_service.pgvector.collection_validator import CollectionMismatchError
from src.ai.vector_store_service.pgvector.database import get_session
from src.ai.vector_store_service.pgvector.repositories.chunk_repository import ChunkRepository
from src.ai.vector_store_service.pgvector.repositories.collection_repository import (
    CollectionRepository,
)
from src.ai.vector_store_service.pgvector.repositories.document_repository import DocumentRepository
from src.ingestion.ingestion_service import IngestionService
from tests.fakes import FailingEmbedder, FakeDocumentLoaderFacade, FakeEmbedder

COLLECTION = "ingest_test"


def _service(texts, metadatas, embedder=None) -> IngestionService:
    embedder = embedder or FakeEmbedder(provider="fake", model="fake-model", dimension=3)
    loader = FakeDocumentLoaderFacade(texts, metadatas)
    return IngestionService(embedder, COLLECTION, document_loader_facade=loader)


def _two_docs():
    texts = ["doc one chunk", "doc two chunk"]
    metadatas = [{"source": "one.txt", "page": 1}, {"source": "two.txt", "page": 1}]
    return texts, metadatas


def _chunk_count() -> int:
    with get_session() as session:
        collection = CollectionRepository(session).require_by_name(COLLECTION)
        return ChunkRepository(session).count(collection.id)


def test_first_ingest_creates_documents_and_chunks(clean_db):
    report = _service(*_two_docs()).ingest()
    assert report.created == 2
    assert report.updated == 0
    assert _chunk_count() == 2


def test_rerun_is_idempotent(clean_db):
    texts, metadatas = _two_docs()
    _service(texts, metadatas).ingest()

    report = _service(texts, metadatas).ingest()

    assert report.created == 0
    assert report.skipped == 2
    assert _chunk_count() == 2


def test_changed_source_is_replaced(clean_db):
    texts, metadatas = _two_docs()
    _service(texts, metadatas).ingest()

    texts[0] = "doc one chunk CHANGED"
    report = _service(texts, metadatas).ingest()

    assert report.updated == 1
    assert report.skipped == 1


def test_removed_source_is_pruned(clean_db):
    texts, metadatas = _two_docs()
    _service(texts, metadatas).ingest()

    # Second run only contains the first source.
    report = _service([texts[0]], [metadatas[0]]).ingest()

    assert report.removed == 1
    assert _chunk_count() == 1


def test_model_mismatch_is_rejected(clean_db):
    texts, metadatas = _two_docs()
    _service(texts, metadatas).ingest()

    other = FakeEmbedder(provider="fake", model="DIFFERENT-model", dimension=3)
    with pytest.raises(CollectionMismatchError):
        _service(texts, metadatas, embedder=other).ingest()


def test_failed_document_preserves_previous_version(clean_db):
    texts, metadatas = _two_docs()
    _service(texts, metadatas).ingest()

    # Change one doc, but the embedder blows up on the new content.
    texts[0] = "boom content"
    failing = FailingEmbedder("boom content", provider="fake", model="fake-model", dimension=3)
    report = _service(texts, metadatas, embedder=failing).ingest()

    assert "one.txt" in report.failed
    # Prior version of the failed document is intact.
    with get_session() as session:
        collection = CollectionRepository(session).require_by_name(COLLECTION)
        document = DocumentRepository(session).get_by_source(collection.id, "one.txt")
        assert document.chunks[0].content == "doc one chunk"
