# src/embeddings/base_embedder.py
from abc import ABC, abstractmethod
from src.common.config import CONFIG

EMBEDDINGS_INDEX_PATH = CONFIG["vector_store"]["index_path"]
EMBEDDINGS_METADATA_PATH = CONFIG["vector_store"]["metadata_path"]

class BaseEmbedder(ABC):
    def __init__(self):
        self.index = None
        self.documents = []

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
    