from abc import ABC, abstractmethod
from typing import List, Dict

class BaseVectorStore(ABC):
    """Abstract base class for all vector store implementations."""
    @abstractmethod
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add a batch of documents to the vector store."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 3, with_embeddings: bool = False) -> List[Dict]:
        """Search the most relevant documents given a query.

        Each result dict carries a ``score`` key holding the cosine similarity
        (range [-1, 1], higher is closer) between the query and the stored vector.

        When ``with_embeddings`` is True each result dict also carries an
        ``embedding`` key (List[float]) holding the stored ingestion vector,
        required by diversity selectors such as MMR.
        """
        pass