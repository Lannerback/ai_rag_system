"""Module for handling document embeddings and vector storage."""
import os
import pickle
from typing import List, Dict
import numpy as np
import faiss
from langchain_openai import AzureOpenAIEmbeddings

from src.ai.base_embedder import BaseEmbedder

EMBEDDINGS_INDEX_PATH = "vector_store/faiss.index"
EMBEDDINGS_METADATA_PATH = "vector_store/metadata.pkl"

class AzureEmbeddingStore(BaseEmbedder):
    def __init__(self):
        super().__init__()
        self._dimension = 1536
        os.makedirs("vector_store", exist_ok=True)
      
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict] = []
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            chunk_size=1000
        )   

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)    
    
    @property
    def dimension(self) -> int:
        return self._dimension
