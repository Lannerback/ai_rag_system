"""FlashRank cross-encoder reranker (local, CPU, no external API)."""
import logging
from typing import Dict, List

from src.ai.rerankers.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class FlashRankReranker(BaseReranker):
    """Reorders candidates with a local cross-encoder via FlashRank.

    The FlashRank model is loaded lazily on first use to keep import cost and
    model download out of the construction path. Original document dicts are
    returned untouched (any extra keys such as ``embedding`` are preserved) so
    downstream diversity selection can consume them.
    """

    def __init__(self, model_name: str, cache_dir: str | None = None):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._ranker = None

    def _get_ranker(self):
        if self._ranker is None:
            from flashrank import Ranker  # lazy: avoids hard import at module load

            logger.info("Loading FlashRank model '%s'", self._model_name)
            # FlashRank's Ranker builds Path(cache_dir); passing None raises. Only
            # forward cache_dir when explicitly configured, else use its default.
            kwargs = {"model_name": self._model_name}
            if self._cache_dir is not None:
                kwargs["cache_dir"] = self._cache_dir
            self._ranker = Ranker(**kwargs)
        return self._ranker

    def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        if not documents:
            return []

        from flashrank import RerankRequest

        passages = [{"id": i, "text": doc["content"]} for i, doc in enumerate(documents)]
        results = self._get_ranker().rerank(RerankRequest(query=query, passages=passages))

        ordered = [documents[result["id"]] for result in results]
        return ordered[:top_k]
