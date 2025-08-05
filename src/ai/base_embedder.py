from abc import ABC, abstractmethod
import os
import faiss
from typing import List, Dict
from src.common.config import CONFIG

EMBEDDINGS_INDEX_PATH = CONFIG["vector_store"]["index_path"]
EMBEDDINGS_METADATA_PATH = CONFIG["vector_store"]["metadata_path"]

class BaseEmbedder(ABC):
    def __init__(self, dimension: int):
        self._dimension = dimension
        vector_store_dir = os.path.dirname(EMBEDDINGS_INDEX_PATH)
        os.makedirs(vector_store_dir, exist_ok=True)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict] = []

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        pass

    @property
    def dimension(self) -> int:
        return self._dimension
    