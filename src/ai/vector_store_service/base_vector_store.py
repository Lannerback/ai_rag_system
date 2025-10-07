from abc import ABC, abstractmethod
from typing import List, Dict

class BaseVectorStore(ABC):
    """Abstract base class for all vector store implementations."""
    @abstractmethod
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add a batch of documents to the vector store."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search the most relevant documents given a query."""
        pass