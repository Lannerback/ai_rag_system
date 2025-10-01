"""Module for handling document embeddings and vector storage."""
import os
from typing import List, Dict
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings

from src.ai.vector_store.base_embedder import BaseEmbedder
from src.common.config import CONFIG

class AzureEmbeddingStore(BaseEmbedder):
    def __init__(self):
        super().__init__(CONFIG["embeddings"]["azure"]["dimension"])
        self._dimension = CONFIG["embeddings"]["azure"]["dimension"]
        
        self.embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
            azure_deployment=CONFIG["azure"]["embedding_deployment"],
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=CONFIG["azure"]["api_version"],
            chunk_size=CONFIG["embeddings"]["azure"]["chunk_size"]
        )   

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)    
    
    @property
    def dimension(self) -> int:
        return self._dimension
