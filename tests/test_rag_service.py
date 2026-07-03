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

    def search(self, question: str, k: int):
        return self._docs


@pytest.fixture(autouse=True)
def _llm_config():
    original = CONFIG.get("llm")
    CONFIG["llm"] = {"system_prompt": "You are a helpful assistant.", "default_k": 5}
    try:
        yield
    finally:
        if original is None:
            del CONFIG["llm"]
        else:
            CONFIG["llm"] = original


def test_answer_question_raises_when_no_docs_found():
    service = RagService(llm=FakeLLM(), vector_store=FakeVectorStoreFacade([]))

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
    service = RagService(llm=FakeLLM(), vector_store=FakeVectorStoreFacade(docs))

    result = service.answer_question("What is the purpose of Regulation (EU) 2024/1689?")

    assert result["answer"] == "fake answer"
    assert len(result["sources"]) == 2
    assert {"source": "reg.pdf", "page": 1, "languages": ["eng"]} in result["sources"]
    assert {"source": "reg.pdf", "page": 2, "languages": ["eng"]} in result["sources"]
