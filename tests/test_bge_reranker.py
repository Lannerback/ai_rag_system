from src.ai.rerankers.bge.bge_reranker import BgeReranker


class _FakeCrossEncoder:
    """Scores [query, passage] pairs by a fixed score map keyed on passage text."""

    def __init__(self, score_by_text):
        self._score_by_text = score_by_text

    def predict(self, pairs):
        return [self._score_by_text[text] for _, text in pairs]


def _doc(text, embedding):
    return {"content": text, "metadata": {"source": text}, "embedding": embedding}


def test_empty_documents_returns_empty():
    reranker = BgeReranker(model_name="test-model")
    assert reranker.rerank("q", [], top_k=5) == []


def test_reorders_by_score_and_truncates():
    docs = [_doc("low", [0.1]), _doc("high", [0.2]), _doc("mid", [0.3])]
    reranker = BgeReranker(model_name="test-model")
    reranker._reranker = _FakeCrossEncoder({"low": 0.1, "high": 0.9, "mid": 0.5})

    result = reranker.rerank("q", docs, top_k=2)

    assert [d["content"] for d in result] == ["high", "mid"]


def test_preserves_embedding_and_metadata_keys():
    docs = [_doc("a", [1.0, 2.0]), _doc("b", [3.0, 4.0])]
    reranker = BgeReranker(model_name="test-model")
    reranker._reranker = _FakeCrossEncoder({"a": 0.9, "b": 0.1})

    result = reranker.rerank("q", docs, top_k=2)

    assert result[0]["embedding"] == [1.0, 2.0]
    assert result[0]["metadata"] == {"source": "a"}
