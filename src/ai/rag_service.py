"""Service class that manages the AI document loading, embedding, and retrieval logic."""
from src.common.APIException import APIException
from src.ai.base_llm import BaseLLM
from src.ai.vector_store_service.vector_store_facade import VectorStoreFacade
from src.common.config import CONFIG
import logging
from src.common.app_context import get_app_context

class RagService:    

    def __init__(self, vector_store: VectorStoreFacade):
        self.__llm: BaseLLM = get_app_context().state.llm
        self.__vector_store: VectorStoreFacade = vector_store
        
        # Load AI service configuration from config.yaml
        self._system_prompt = CONFIG["llm"]["system_prompt"]
        self._default_k = CONFIG["llm"]["default_k"]


    def answer_question(self, question: str) -> dict:
        """Answer a question using the LLM and relevant docs."""
        k: int = self._default_k
        relevant_docs = self._get_relevant_docs(question, k=k)
        if not relevant_docs:
            raise APIException(detail = "No relevant documentation found for the question.", status_code=400, code = "no_docs_found")

        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        prompt = f"""Based on the following documentation excerpts, please answer the question.
                        If you cannot answer the question based on the provided context, say so.

                        Documentation:
                        {context}

                        Question: {question}

                        Answer:"""
        response = self.__llm.invoke([
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt}
        ])
        return {
            "answer": response.content,
            "sources": list({frozenset(doc["metadata"].items()): doc["metadata"] for doc in relevant_docs}.values())
        }
        
    def _get_relevant_docs(self, query, k=None):
        """Retrieve relevant documents for a query."""
        k = k or self._default_k
        return self.__vector_store.search(query, k=k)