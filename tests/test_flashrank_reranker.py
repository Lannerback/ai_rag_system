import sys
import types

import pytest

from src.ai.rerankers.flashrank.flashrank_reranker import FlashRankReranker


class _FakeRanker:
    """Scores passages by a fixed score map keyed on passage text, desc order."""

    def __init__(self, score_by_text):
        self._score_by_text = score_by_text

    def rerank(self, request):
        scored = [
            {"id": p["id"], "score": self._score_by_text[p["text"]]}
            for p in request.passages
        ]
        return sorted(scored, key=lambda r: r["score"], reverse=True)


@pytest.fixture
def fake_flashrank(monkeypatch):
    """Install a fake `flashrank` module exposing RerankRequest only."""

    class RerankRequest:
        def __init__(self, query, passages):
            self.query = query
            self.passages = passages

    module = types.ModuleType("flashrank")
    module.RerankRequest = RerankRequest
    monkeypatch.setitem(sys.modules, "flashrank", module)
    return module


def _doc(text, embedding):
    return {"content": text, "metadata": {"source": text}, "embedding": embedding}


def test_empty_documents_returns_empty(fake_flashrank):
    reranker = FlashRankReranker(model_name="test-model")
    assert reranker.rerank("q", [], top_k=5) == []


def test_reorders_by_score_and_truncates(fake_flashrank):
    docs = [
        _doc("low", [0.1]),
        _doc("high", [0.2]),
        _doc("mid", [0.3]),
    ]
    reranker = FlashRankReranker(model_name="test-model")
    reranker._ranker = _FakeRanker({"low": 0.1, "high": 0.9, "mid": 0.5})

    result = reranker.rerank("q", docs, top_k=2)

    assert [d["content"] for d in result] == ["high", "mid"]


def test_preserves_embedding_and_metadata_keys(fake_flashrank):
    docs = [_doc("a", [1.0, 2.0]), _doc("b", [3.0, 4.0])]
    reranker = FlashRankReranker(model_name="test-model")
    reranker._ranker = _FakeRanker({"a": 0.9, "b": 0.1})

    result = reranker.rerank("q", docs, top_k=2)

    assert result[0]["embedding"] == [1.0, 2.0]
    assert result[0]["metadata"] == {"source": "a"}
