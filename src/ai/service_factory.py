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

            rag_service = RagService(llm, VectorStoreFacade(vector_store))
            cls._instances["rag_facade"] = RagFacade(rag_service)
        return cls._instances["rag_facade"]
