"""Module for handling document embeddings and vector storage using Gemini."""
import os
from typing import List, Dict
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.ai.vector_store.base_embedder import BaseEmbedder
from src.common.config import CONFIG

class GeminiEmbeddingStore(BaseEmbedder):
    def __init__(self):
        super().__init__(CONFIG["embeddings"]["gemini"]["dimension"])
        self._dimension = CONFIG["embeddings"]["gemini"]["dimension"]

        self.embeddings: GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(
            model=CONFIG["gemini"]["embedding_model"],
            api_key=os.getenv("GOOGLE_API_KEY"),
            chunk_size=CONFIG["embeddings"]["gemini"]["chunk_size"] if "chunk_size" in CONFIG["embeddings"]["gemini"] else None
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return list of embeddings for documents (Gemini, one by one)."""
        return self.embeddings.embed_documents(texts)
 

    def embed_query(self, query: str) -> List[float]:
        """Return embedding for a single query string — same interface as Azure."""
        return self.embeddings.embed_query(query)

    @property
    def dimension(self) -> int:
        return self._dimension
