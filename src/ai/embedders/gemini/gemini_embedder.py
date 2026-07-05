import os
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.ai.embedders.base_embedder import BaseEmbedder
from src.common.config import CONFIG

# Gemini caps a single embed request; batch documents to stay under the limit.
_BATCH_SIZE = 100


class GeminiEmbedder(BaseEmbedder):
    def __init__(self):
        self._model = CONFIG["gemini"]["embedding_model"]
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self._model,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )
        self._dimension = CONFIG["llm"]["gemini"]["embeddings_dimension"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Asymmetric task_type: documents and queries are embedded into aligned but
        # role-specific subspaces, which materially improves retrieval quality.
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> List[float]:
        return self._embed([query], task_type="RETRIEVAL_QUERY")[0]

    @retry(
        retry=retry_if_exception_type(GoogleGenerativeAIError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        return self.embeddings.embed_documents(texts, batch_size=_BATCH_SIZE, task_type=task_type)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model
