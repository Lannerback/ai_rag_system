"""Greedy Maximal Marginal Relevance (MMR) selector.

MMR iteratively picks the document maximizing:

    score = lambda * relevance(doc) - (1 - lambda) * max_sim(doc, already_selected)

Relevance is taken from the input ordering (the upstream reranker's ranking),
normalized to [0, 1] so it is comparable to the cosine redundancy term.
Similarity between documents is cosine over their stored embeddings.
"""
from typing import Dict, List

import numpy as np

from src.ai.diversity.base_diversity_selector import BaseDiversitySelector


class MMRSelector(BaseDiversitySelector):
    def select(self, documents: List[Dict], top_k: int, lambda_mult: float) -> List[Dict]:
        if not documents:
            return []
        if top_k >= len(documents) and lambda_mult >= 1.0:
            return documents[:top_k]

        matrix = self._normalized_matrix(documents)
        relevance = self._relevance_scores(len(documents))

        selected: List[int] = []
        candidates = list(range(len(documents)))

        while candidates and len(selected) < top_k:
            best_index = self._best_candidate(
                candidates, selected, matrix, relevance, lambda_mult
            )
            selected.append(best_index)
            candidates.remove(best_index)

        return [documents[i] for i in selected]

    @staticmethod
    def _best_candidate(
        candidates: List[int],
        selected: List[int],
        matrix: np.ndarray,
        relevance: np.ndarray,
        lambda_mult: float,
    ) -> int:
        best_index = candidates[0]
        best_score = -np.inf
        for index in candidates:
            redundancy = (
                max(float(matrix[index] @ matrix[j]) for j in selected) if selected else 0.0
            )
            score = lambda_mult * relevance[index] - (1.0 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    @staticmethod
    def _normalized_matrix(documents: List[Dict]) -> np.ndarray:
        matrix = np.array([doc["embedding"] for doc in documents], dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @staticmethod
    def _relevance_scores(count: int) -> np.ndarray:
        """Linear relevance from input rank: first doc -> 1.0, last -> ~0."""
        if count == 1:
            return np.array([1.0], dtype="float32")
        return np.linspace(1.0, 0.0, count, dtype="float32")
