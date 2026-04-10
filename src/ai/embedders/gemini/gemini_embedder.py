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
        vectors = self.embeddings.embed_documents(
            texts,
            output_dimensionality=self._dimension,
        )
        for index, vector in enumerate(vectors):
            if len(vector) != self._dimension:
                raise ValueError(
                    f"Gemini returned doc embedding dim {len(vector)} for chunk {index}, "
                    f"expected {self._dimension}. Check `llm.gemini.embeddings_dimension`."
                )
        return vectors

    def embed_query(self, query: str) -> List[float]:
        vector = self.embeddings.embed_query(
            query,
            output_dimensionality=self._dimension,
        )
        if len(vector) != self._dimension:
            raise ValueError(
                f"Gemini returned query embedding dim {len(vector)}, expected {self._dimension}. "
                f"Check `llm.gemini.embeddings_dimension`."
            )
        return vector

    @property
    def dimension(self) -> int:
        return self._dimension
