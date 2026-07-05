"""Interface for diversity-aware result selectors.

A selector consumes the neutral search-result shape (``List[Dict]`` carrying an
``embedding`` key) and returns a re-ordered, truncated subset that balances
relevance against redundancy. Input order is treated as the relevance ranking.
"""
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseDiversitySelector(ABC):
    @abstractmethod
    def select(self, documents: List[Dict], top_k: int, lambda_mult: float) -> List[Dict]:
        """Return ``top_k`` documents balancing relevance and diversity.

        ``lambda_mult`` in [0, 1]: 1.0 = pure relevance, 0.0 = pure diversity.
        Each document must carry an ``embedding`` (List[float]).
        """
        pass
