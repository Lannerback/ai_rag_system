"""Abstract Factory for reranker backends.

Concrete providers lazy-import their backend so selecting one reranker never
requires another's dependencies. Adding a backend (Cohere, BGE, ...) is a single
new provider plus one registry entry in ServiceFactory — the pipeline is untouched.
"""
from abc import ABC, abstractmethod

from src.common.config import CONFIG
from src.ai.rerankers.base_reranker import BaseReranker


class RerankerProvider(ABC):
    @abstractmethod
    def create_reranker(self) -> BaseReranker:
        pass


class FlashRankRerankerProvider(RerankerProvider):
    def create_reranker(self) -> BaseReranker:
        from src.ai.rerankers.flashrank.flashrank_reranker import FlashRankReranker

        flashrank_config = CONFIG["reranking"]["flashrank"]
        return FlashRankReranker(
            model_name=flashrank_config["model"],
            cache_dir=flashrank_config.get("cache_dir"),
        )


class BgeRerankerProvider(RerankerProvider):
    def create_reranker(self) -> BaseReranker:
        from src.ai.rerankers.bge.bge_reranker import BgeReranker

        bge_config = CONFIG["reranking"]["bge"]
        return BgeReranker(model_name=bge_config["model"])
