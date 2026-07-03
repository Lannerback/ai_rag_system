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

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider identifier (e.g. 'azure', 'gemini'); binds a collection to its source."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Concrete embedding model/deployment name used to produce vectors."""
        pass