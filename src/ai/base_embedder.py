# src/embeddings/base_embedder.py
from abc import ABC, abstractmethod
import os
import pickle
import faiss
from typing import List, Dict
import numpy as np

EMBEDDINGS_INDEX_PATH = "vector_store/faiss.index"
EMBEDDINGS_METADATA_PATH = "vector_store/metadata.pkl"

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

    def load_from_disk(self, index_path=EMBEDDINGS_INDEX_PATH, metadata_path=EMBEDDINGS_METADATA_PATH) -> bool:
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.documents = pickle.load(f)
            return True
        return False

    def save_to_disk(self, index_path=EMBEDDINGS_INDEX_PATH, metadata_path=EMBEDDINGS_METADATA_PATH):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.documents, f)
            
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to the vector store and persist them."""
        if not texts:
            return
        embeddings = self.embed_documents(texts)
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for text, metadata in zip(texts, metadatas):
            self.documents.append({
                "content": text,
                "metadata": metadata
            })
        self.save_to_disk()
        
    def search(self, query: str, k: int = 3) -> List[Dict]:
        query_embedding = self.embed_query(query)
        query_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_np)
        D, I = self.index.search(query_np, k)
        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results
