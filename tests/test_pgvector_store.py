import pytest

from src.ai.vector_store_service.pgvector.collection_validator import CollectionMismatchError
from src.ai.vector_store_service.pgvector.pgvector_store import PgVectorStore
from tests.fakes import FakeEmbedder

# Orthonormal vectors so cosine ordering is deterministic.
MAPPING = {
    "alpha document": [1.0, 0.0, 0.0],
    "beta document": [0.0, 1.0, 0.0],
    "gamma document": [0.0, 0.0, 1.0],
    # query closest to alpha, then beta, then gamma
    "query": [0.8, 0.6, 0.0],
}


def _embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=3, provider="fake", model="fake-model", mapping=MAPPING)


def _store(collection: str) -> PgVectorStore:
    return PgVectorStore(_embedder(), collection)


def test_search_returns_chunks_in_cosine_order(clean_db):
    store = _store("ordering")
    texts = ["alpha document", "beta document", "gamma document"]
    metadatas = [{"source": "s.txt", "page": i} for i in range(3)]
    store.add_documents(texts, metadatas)

    results = store.search("query", k=3)

    assert [r["content"] for r in results] == [
        "alpha document",
        "beta document",
        "gamma document",
    ]


def test_collection_isolation(clean_db):
    store_a = _store("col_a")
    store_b = _store("col_b")
    store_a.add_documents(["alpha document"], [{"source": "a.txt", "page": 1}])
    store_b.add_documents(["beta document"], [{"source": "b.txt", "page": 1}])

    results = store_a.search("query", k=5)

    assert len(results) == 1
    assert results[0]["content"] == "alpha document"
    assert results[0]["metadata"]["source"] == "a.txt"


def test_metadata_round_trip_with_promoted_fields(clean_db):
    store = _store("meta")
    metadata = {
        "source": "doc.pdf",
        "page": 7,
        "lang": "ara",
        "ocr": True,
        "custom_key": "custom_value",
    }
    store.add_documents(["alpha document"], [metadata])

    result = store.search("query", k=1)[0]["metadata"]

    assert result["source"] == "doc.pdf"
    assert result["page"] == 7
    assert result["language"] == "ara"  # promoted from 'lang'
    assert result["loader"] == "ocr"  # derived from ocr flag
    assert result["custom_key"] == "custom_value"  # remainder preserved in JSONB


def test_dimension_mismatch_is_rejected(clean_db):
    _store("dims").add_documents(["alpha document"], [{"source": "s.txt", "page": 1}])

    # Same collection name + provider/model, but a different dimension.
    mismatched = PgVectorStore(
        FakeEmbedder(dimension=4, provider="fake", model="fake-model"), "dims"
    )
    with pytest.raises(CollectionMismatchError):
        mismatched.search("query", k=1)


def test_search_on_missing_collection_returns_empty(clean_db):
    assert _store("never_created").search("query", k=3) == []
