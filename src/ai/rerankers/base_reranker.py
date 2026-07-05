"""Interface for cross-encoder rerankers.

A reranker consumes the neutral search-result shape (``List[Dict]`` of
``{"content", "metadata", ...}``) and returns the same shape, re-ordered by
semantic relevance to the query and truncated to ``top_k``. Any extra keys on
the input dicts (e.g. ``embedding``) MUST be preserved so downstream stages such
as MMR can consume them.
"""
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """Return the ``top_k`` documents most relevant to ``query``, best first."""
        pass
