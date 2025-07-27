"""Service class that manages the load, save and search operations for embeddings using FAISS.
"""
import os
import time
import logging
from typing import List, Dict
import pickle
import numpy as np
import faiss


from src.ai.base_embedder import BaseEmbedder

# TODO: Consider using a configuration file for paths
EMBEDDINGS_INDEX_PATH = "vector_store/faiss.index"
EMBEDDINGS_METADATA_PATH = "vector_store/metadata.pkl"

class EmbedderService():
    def __init__(self,base_embedder: BaseEmbedder):
        self.__base_embedder = base_embedder

    def load_from_disk(self, index_path=EMBEDDINGS_INDEX_PATH, metadata_path=EMBEDDINGS_METADATA_PATH) -> bool:
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.__base_embedder.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.__base_embedder.documents = pickle.load(f)
            return True
        return False

    def save_to_disk(self, index_path=EMBEDDINGS_INDEX_PATH, metadata_path=EMBEDDINGS_METADATA_PATH):
        faiss.write_index(self.__base_embedder.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.__base_embedder.documents, f)
            
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to the vector store and persist them."""
        if not texts:
            return
        logging.debug(f"🔹 Adding {len(texts)} documents to the vector store...")
        start_time = time.time()  # ⏱ Start timer
        embeddings = self.__base_embedder.embed_documents(texts)
        duration = time.time() - start_time  # ⏱ End timer
        logging.debug(f"🔹 embed_documents() took {duration:.2f} seconds for {len(texts)} texts")

        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        self.__base_embedder.index.add(embeddings_np)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for text, metadata in zip(texts, metadatas):
            self.__base_embedder.documents.append({
                "content": text,
                "metadata": metadata
            })
        self.save_to_disk()
        
    def search(self, query: str, k: int = 3) -> List[Dict]:
        query_embedding = self.__base_embedder.embed_query(query)
        query_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_np)
        D, I = self.__base_embedder.index.search(query_np, k)
        results = []
        for idx in I[0]:
            if idx < len(self.__base_embedder.documents):
                results.append(self.__base_embedder.documents[idx])
        return results
