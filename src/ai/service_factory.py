# src/ai/service_factory.py
from src.common.config import CONFIG
from src.ai.embedders.base_embedder import BaseEmbedder
from src.ai.base_llm import BaseLLM

from src.ai.embedders.azure.azure_openai_llm import AzureLLM
from src.ai.embedders.azure.azure_embedder import AzureEmbedder
from src.ai.embedders.gemini.gemini_embedder import GeminiEmbedder
from src.ai.embedders.gemini.gemini_llm import GeminiLLM

from src.ai.rag_service import RagService
from src.ai.rag_facade import RagFacade
from src.ai.vector_store_service.vector_store_facade import VectorStoreFacade
from src.ai.vector_store_service.base_vector_store import BaseVectorStore
from src.ai.vector_store_service.base_vector_store_initializer import BaseVectorStoreInitializer
from src.ai.vector_store_service.vector_store_provider import (
    VectorStoreProvider,
    FaissVectorStoreProvider,
    PgVectorStoreProvider,
)
from src.ai.rerankers.base_reranker import BaseReranker
from src.ai.rerankers.reranker_provider import (
    RerankerProvider,
    FlashRankRerankerProvider,
    BgeRerankerProvider,
)
from src.ai.diversity.base_diversity_selector import BaseDiversitySelector
from src.ai.diversity.mmr_selector import MMRSelector


class ServiceFactory:
    """Centralized factory for creating and caching singleton service instances."""
    _instances = {}

    PROVIDERS = {
        "gemini": {"llm": GeminiLLM, "embedder": GeminiEmbedder},
        "azure": {"llm": AzureLLM, "embedder": AzureEmbedder},
    }

    # Registry: maps the configured backend to its Abstract Factory. Adding a
    # backend is a single entry here — selection stays a dict lookup, never a branch.
    VECTOR_STORE_PROVIDERS = {
        "faiss": FaissVectorStoreProvider,
        "pgvector": PgVectorStoreProvider,
    }

    # Registry: configured reranker backend -> its provider. Add a backend here.
    RERANKER_PROVIDERS = {
        "flashrank": FlashRankRerankerProvider,
        "bge": BgeRerankerProvider,
    }

    # Registry: configured diversity strategy -> its selector class.
    DIVERSITY_SELECTORS = {
        "mmr": MMRSelector,
    }

    @classmethod
    def get_llm(cls) -> BaseLLM:
        if "llm" not in cls._instances:
            provider = CONFIG["llm"]["provider"]
            llm_class = cls.PROVIDERS[provider]["llm"]
            cls._instances["llm"] = llm_class()
        return cls._instances["llm"]

    @classmethod
    def get_embedder(cls) -> BaseEmbedder:
        if "embedder" not in cls._instances:
            provider = CONFIG["llm"]["provider"]
            embedder_class = cls.PROVIDERS[provider]["embedder"]
            cls._instances["embedder"] = embedder_class()
        return cls._instances["embedder"]

    @classmethod
    def _get_vector_store_provider(cls) -> VectorStoreProvider:
        if "vector_store_provider" not in cls._instances:
            backend = CONFIG["vector_store"]["backend"]
            provider_class = cls.VECTOR_STORE_PROVIDERS[backend]
            cls._instances["vector_store_provider"] = provider_class()
        return cls._instances["vector_store_provider"]

    @classmethod
    def get_vector_store(cls) -> BaseVectorStore:
        if "vector_store" not in cls._instances:
            provider = cls._get_vector_store_provider()
            cls._instances["vector_store"] = provider.create_store(cls.get_embedder())
        return cls._instances["vector_store"]

    @classmethod
    def get_vector_store_initializer(cls) -> BaseVectorStoreInitializer:
        if "vector_store_initializer" not in cls._instances:
            provider = cls._get_vector_store_provider()
            cls._instances["vector_store_initializer"] = provider.create_initializer(
                cls.get_vector_store()
            )
        return cls._instances["vector_store_initializer"]

    @classmethod
    def get_reranker(cls) -> BaseReranker | None:
        """Return the configured reranker, or None when reranking is disabled."""
        if not CONFIG["reranking"].get("enabled", False):
            return None
        if "reranker" not in cls._instances:
            strategy = CONFIG["reranking"]["provider"]
            provider: RerankerProvider = cls.RERANKER_PROVIDERS[strategy]()
            cls._instances["reranker"] = provider.create_reranker()
        return cls._instances["reranker"]

    @classmethod
    def get_diversity_selector(cls) -> BaseDiversitySelector | None:
        """Return the configured diversity selector, or None when the block is absent."""
        if not CONFIG.get("diversity"):
            return None
        if "diversity_selector" not in cls._instances:
            strategy = CONFIG["diversity"]["strategy"]
            cls._instances["diversity_selector"] = cls.DIVERSITY_SELECTORS[strategy]()
        return cls._instances["diversity_selector"]

    @classmethod
    def get_faiss_vector_store(cls) -> BaseVectorStore:
        """Back-compat: explicit FAISS store regardless of the configured backend."""
        if "faiss_vector_store" not in cls._instances:
            cls._instances["faiss_vector_store"] = FaissVectorStoreProvider().create_store(
                cls.get_embedder()
            )
        return cls._instances["faiss_vector_store"]

    @classmethod
    def get_rag_facade(cls) -> RagFacade:
        """Return initialized RAG facade."""
        if "rag_facade" not in cls._instances:
            llm = cls.get_llm()
            vector_store = cls.get_vector_store()

            rag_service = RagService(
                llm,
                VectorStoreFacade(vector_store),
                cls.get_diversity_selector(),
                reranker=cls.get_reranker(),
            )
            cls._instances["rag_facade"] = RagFacade(rag_service)
        return cls._instances["rag_facade"]
