# Class that defines the interface for the embedder in different providers
# In order to add a new provider, you need to implement this class
# It offers methods to embed documents and queries and to get the dimension of the embeddings
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings for this embedder."""
        pass