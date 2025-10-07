import os
import pickle
import time
import logging
from typing import List, Dict
import numpy as np
import faiss

from src.ai.embedders.base_embedder import BaseEmbedder
from src.common.config import CONFIG
from src.ai.vector_store_service.base_vector_store import BaseVectorStore

class FaissVectorStore(BaseVectorStore):
    def __init__(self, embedder: BaseEmbedder):
        self.embedder: BaseEmbedder = embedder
        self.index = faiss.IndexFlatIP(self.embedder.dimension)
        self.documents: List[Dict] = []
        self._index_path = CONFIG["vector_store"]["index_path"]
        self._metadata_path = CONFIG["vector_store"]["metadata_path"]

    def load_from_disk(self) -> bool:
        if os.path.exists(self._index_path) and os.path.exists(self._metadata_path):
            self.index = faiss.read_index(self._index_path)
            with open(self._metadata_path, "rb") as f:
                self.documents = pickle.load(f)
            return True
        return False

    def save_to_disk(self):
        os.makedirs(os.path.dirname(self._index_path), exist_ok=True)
        faiss.write_index(self.index, self._index_path)
        with open(self._metadata_path, "wb") as f:
            pickle.dump(self.documents, f)

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        if not texts:
            return
        logging.debug(f"🔹 Adding {len(texts)} docs to FAISS...")
        start = time.time()
        embeddings = self.embedder.embed_documents(texts)
        logging.debug(f"embed_documents took {time.time() - start:.2f}s")

        embeddings_np = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)

        if metadatas is None:
            metadatas = [{} for _ in texts]

        for text, metadata in zip(texts, metadatas):
            self.documents.append({"content": text, "metadata": metadata})

        self.save_to_disk()

    def search(self, query: str, k: int = 3) -> List[Dict]:
        query_embedding = self.embedder.embed_query(query)
        query_np = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_np)
        _, I = self.index.search(query_np, k)
        return [self.documents[idx] for idx in I[0] if idx < len(self.documents)]