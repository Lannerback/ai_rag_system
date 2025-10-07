"""Module for handling document embeddings and vector storage."""
import os
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings

from src.ai.embedders.base_embedder import BaseEmbedder
from src.common.config import CONFIG

class AzureEmbedder(BaseEmbedder):
    def __init__(self):
        self._dimension = CONFIG["llm"]["azure"]["embeddings_dimension"]
    
        self.embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
            azure_deployment=CONFIG["azure"]["embedding_deployment"],
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=CONFIG["azure"]["api_version"]
        )   

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)    
    
    @property
    def dimension(self) -> int:
        return self._dimension
