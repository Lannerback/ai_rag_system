"""BGE cross-encoder reranker (local, torch) via sentence-transformers CrossEncoder."""
import logging
from typing import Dict, List

from src.ai.rerankers.base_reranker import BaseReranker
from src.common.timing import timed_step

logger = logging.getLogger(__name__)


class BgeReranker(BaseReranker):
    """Reorders candidates with a BGE cross-encoder (e.g. bge-reranker-v2-m3).

    The CrossEncoder model is loaded lazily on first use to keep import cost and
    model download out of the construction path. Original document dicts are
    returned untouched (any extra keys such as ``embedding`` are preserved) so
    downstream diversity selection can consume them.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # lazy: avoids torch import at load

            logger.info("Loading BGE reranker '%s'", self._model_name)
            with timed_step(logger, "bge_model_load"):
                self._reranker = CrossEncoder(self._model_name)
        return self._reranker

    def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        if not documents:
            return []

        reranker = self._get_reranker()
        pairs = [[query, doc["content"]] for doc in documents]
        with timed_step(logger, "bge_inference"):
            scores = reranker.predict(pairs)

        ranked = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
