import os
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.ai.embedders.base_embedder import BaseEmbedder
from src.common.config import CONFIG

class GeminiEmbedder(BaseEmbedder):
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=CONFIG["gemini"]["embedding_model"],
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self._dimension = CONFIG["llm"]["gemini"]["embeddings_dimension"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)

    @property
    def dimension(self) -> int:
        return self._dimension