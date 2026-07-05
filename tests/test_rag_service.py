import pytest

from src.ai.base_llm import BaseLLM
from src.ai.rag_service import RagService
from src.common.APIException import APIException
from src.common.config import CONFIG


class FakeLLM(BaseLLM):
    class _Response:
        def __init__(self, content: str):
            self.content = content

    def invoke(self, messages: list[dict]):
        return self._Response("fake answer")


class FakeVectorStoreFacade:
    def __init__(self, docs):
        self._docs = docs

    def search(self, question: str, k: int, with_embeddings: bool = False):
        return self._docs


class PassthroughReranker:
    def rerank(self, query: str, documents, top_k: int):
        return documents[:top_k]


class PassthroughDiversitySelector:
    def select(self, documents, top_k: int, lambda_mult: float):
        return documents[:top_k]


_DEFAULT = object()


def _build_service(docs, diversity_selector=_DEFAULT, reranker=_DEFAULT):
    if diversity_selector is _DEFAULT:
        diversity_selector = PassthroughDiversitySelector()
    if reranker is _DEFAULT:
        reranker = PassthroughReranker()
    return RagService(
        llm=FakeLLM(),
        vector_store=FakeVectorStoreFacade(docs),
        diversity_selector=diversity_selector,
        reranker=reranker,
    )


@pytest.fixture(autouse=True)
def _pipeline_config():
    originals = {key: CONFIG.get(key) for key in ("llm", "retrieval", "diversity")}
    CONFIG["llm"] = {"system_prompt": "You are a helpful assistant."}
    CONFIG["retrieval"] = {"candidate_k": 120, "rerank_k": 30, "final_k": 12}
    CONFIG["diversity"] = {"strategy": "mmr", "lambda_mult": 0.6}
    try:
        yield
    finally:
        for key, value in originals.items():
            if value is None:
                CONFIG.pop(key, None)
            else:
                CONFIG[key] = value


def test_answer_question_raises_when_no_docs_found():
    service = _build_service([])

    with pytest.raises(APIException) as exc_info:
        service.answer_question("What is the purpose of Regulation (EU) 2024/1689?")

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "no_docs_found"


def test_answer_question_dedupes_sources_with_nested_metadata_values():
    docs = [
        {
            "content": "chunk 1",
            "metadata": {"source": "reg.pdf", "page": 1, "languages": ["eng"]},
        },
        {
            "content": "chunk 2 (duplicate source)",
            "metadata": {"source": "reg.pdf", "page": 1, "languages": ["eng"]},
        },
        {
            "content": "chunk 3",
            "metadata": {"source": "reg.pdf", "page": 2, "languages": ["eng"]},
        },
    ]
    service = _build_service(docs)

    result = service.answer_question("What is the purpose of Regulation (EU) 2024/1689?")

    assert result["answer"] == "fake answer"
    assert len(result["sources"]) == 2
    assert {"source": "reg.pdf", "page": 1, "languages": ["eng"]} in result["sources"]
    assert {"source": "reg.pdf", "page": 2, "languages": ["eng"]} in result["sources"]


def test_diversity_disabled_returns_top_final_k():
    CONFIG["retrieval"]["final_k"] = 2
    docs = [{"content": f"c{i}", "metadata": {"source": f"s{i}"}} for i in range(5)]
    service = _build_service(docs, diversity_selector=None, reranker=None)

    result = service.answer_question("q")

    assert len(result["sources"]) == 2
    assert {"source": "s0"} in result["sources"]
    assert {"source": "s1"} in result["sources"]


def test_min_similarity_filters_low_score_candidates():
    CONFIG["retrieval"]["min_similarity"] = 0.85
    docs = [
        {"content": "keep", "metadata": {"source": "a.pdf", "page": 1}, "score": 0.9},
        {"content": "drop", "metadata": {"source": "b.pdf", "page": 2}, "score": 0.4},
    ]
    service = _build_service(docs)

    result = service.answer_question("q")

    assert result["sources"] == [{"source": "a.pdf", "page": 1}]


def test_min_similarity_all_below_floor_raises_no_docs_found():
    CONFIG["retrieval"]["min_similarity"] = 0.85
    docs = [{"content": "x", "metadata": {"source": "a.pdf"}, "score": 0.5}]
    service = _build_service(docs)

    with pytest.raises(APIException) as exc_info:
        service.answer_question("q")

    assert exc_info.value.code == "no_docs_found"
