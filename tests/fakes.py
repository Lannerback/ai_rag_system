import hashlib
from typing import Dict, List, Tuple

from src.ai.embedders.base_embedder import BaseEmbedder


class FakeEmbedder(BaseEmbedder):
    """Deterministic, offline embedder for tests.

    Texts present in `mapping` get their explicit vector (useful for asserting KNN
    order); everything else falls back to a stable hash-derived vector.
    """

    def __init__(
        self,
        dimension: int = 3,
        provider: str = "fake",
        model: str = "fake-model",
        mapping: Dict[str, List[float]] | None = None,
    ):
        self._dimension = dimension
        self._provider = provider
        self._model = model
        self._mapping = mapping or {}

    def _vector(self, text: str) -> List[float]:
        if text in self._mapping:
            return list(self._mapping[text])
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._vector(query)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model


class FailingEmbedder(FakeEmbedder):
    """Embedder that raises when a batch contains the configured marker text."""

    def __init__(self, boom_text: str, **kwargs):
        super().__init__(**kwargs)
        self._boom_text = boom_text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._boom_text in texts:
            raise RuntimeError("embedding provider failed")
        return super().embed_documents(texts)


class FakeDocumentLoaderFacade:
    """Stands in for DocumentLoaderFacade, returning preset texts + metadata."""

    def __init__(self, texts: List[str], metadatas: List[Dict]):
        self._texts = texts
        self._metadatas = metadatas

    def load_all_documents(self) -> Tuple[List[str], List[Dict]]:
        return list(self._texts), [dict(m) for m in self._metadatas]
