from abc import ABC, abstractmethod

from src.common.config import CONFIG
from src.ai.embedders.base_embedder import BaseEmbedder
from src.ai.vector_store_service.base_vector_store import BaseVectorStore
from src.ai.vector_store_service.base_vector_store_initializer import BaseVectorStoreInitializer


class VectorStoreProvider(ABC):
    """Abstract Factory: produces a backend family (store + matching initializer).

    Concrete providers lazy-import their backend modules so selecting one backend
    never requires the other's dependencies to be installed.
    """

    @abstractmethod
    def create_store(self, embedder: BaseEmbedder) -> BaseVectorStore:
        pass

    @abstractmethod
    def create_initializer(self, store: BaseVectorStore) -> BaseVectorStoreInitializer:
        pass


class FaissVectorStoreProvider(VectorStoreProvider):
    def create_store(self, embedder: BaseEmbedder) -> BaseVectorStore:
        from src.ai.vector_store_service.faiss.faiss_vector_store import FaissVectorStore
        return FaissVectorStore(embedder)

    def create_initializer(self, store: BaseVectorStore) -> BaseVectorStoreInitializer:
        from src.ai.vector_store_service.faiss.faiss_vector_store_initializer import (
            FaissVectorStoreInitializer,
        )
        from src.ai.document_loaders.document_loader_facade import DocumentLoaderFacade
        return FaissVectorStoreInitializer(store, DocumentLoaderFacade())


class PgVectorStoreProvider(VectorStoreProvider):
    def create_store(self, embedder: BaseEmbedder) -> BaseVectorStore:
        from src.ai.vector_store_service.pgvector.pgvector_store import PgVectorStore
        return PgVectorStore(embedder, CONFIG["vector_store"]["collection"])

    def create_initializer(self, store: BaseVectorStore) -> BaseVectorStoreInitializer:
        from src.ai.vector_store_service.pgvector.pgvector_store_initializer import (
            PgVectorStoreInitializer,
        )
        return PgVectorStoreInitializer(store)
