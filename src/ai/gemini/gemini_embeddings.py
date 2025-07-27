"""Module for handling document embeddings and vector storage using Gemini."""
import os
import pickle
from typing import List, Dict
import numpy as np
import faiss
import logging
from google import generativeai as genai

from src.ai.base_embedder import BaseEmbedder

EMBEDDINGS_INDEX_PATH = "vector_store/faiss.index"
EMBEDDINGS_METADATA_PATH = "vector_store/metadata.pkl"

class GeminiEmbeddingStore(BaseEmbedder):
    def __init__(self):
        super().__init__()
        self._dimension = 768  # Gemini embedding models usually return 768 dimensions

        os.makedirs("vector_store", exist_ok=True)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict] = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return list of embeddings for documents (Gemini, one by one)."""
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model=os.getenv("GEMINI_EMBEDDING_MODEL"),
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response["embedding"])
        return embeddings
 

    def embed_query(self, query: str) -> List[float]:
        """Return embedding for a single query string — same interface as Azure."""
        embedding_result = genai.embed_content(
            model=os.getenv("GEMINI_EMBEDDING_MODEL"),
            content=query
        )["embedding"]
        
        actual_dimension = len(embedding_result)        
        if actual_dimension != self.dimension:
            logging.warning("WARNING: Dimension mismatch detected for query embedding!")
            
        return embedding_result


    @property
    def dimension(self) -> int:
        return self._dimension
