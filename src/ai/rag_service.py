"""Service class that manages the AI document loading, embedding, and retrieval logic."""
from src.common.APIException import APIException
from src.ai.base_llm import BaseLLM
from src.ai.diversity.base_diversity_selector import BaseDiversitySelector
from src.ai.rerankers.base_reranker import BaseReranker
from src.ai.vector_store_service.vector_store_facade import VectorStoreFacade
from src.common.config import CONFIG
from src.common.timing import timed_step
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class RagService:

    def __init__(
        self,
        llm: BaseLLM,
        vector_store: VectorStoreFacade,
        diversity_selector: BaseDiversitySelector | None = None,
        reranker: BaseReranker | None = None,
    ):
        self.__llm: BaseLLM = llm
        self.__vector_store: VectorStoreFacade = vector_store
        self.__diversity_selector: BaseDiversitySelector | None = diversity_selector
        self.__reranker: BaseReranker | None = reranker

        # Load AI service configuration from config.yaml
        self._system_prompt = CONFIG["llm"]["system_prompt"]
        self._candidate_k = CONFIG["retrieval"]["candidate_k"]
        self._rerank_k = CONFIG["retrieval"]["rerank_k"]
        self._final_k = CONFIG["retrieval"]["final_k"]
        # Cosine-similarity floor on vector-recall candidates; 0.0 (or absent) disables it.
        self._min_similarity = CONFIG["retrieval"].get("min_similarity", 0.0)
        self._mmr_lambda = CONFIG.get("diversity", {}).get("lambda_mult", 0.0)


    def answer_question(self, question: str) -> dict:
        """Answer a question using the LLM and relevant docs."""
        relevant_docs = self._retrieve(question)
        if not relevant_docs:
            raise APIException(detail = "No relevant documentation found for the question.", status_code=400, code = "no_docs_found")

        context = "\n\n".join(
                    [f"Document {i+1} (source: {doc['metadata'].get('source', 'unknown')}, page: {doc['metadata'].get('page','?')}):\n{doc['content']}"
                    for i, doc in enumerate(relevant_docs)]
                )
        prompt = f"""You are a helpful assistant that answers questions using only the information provided below.
                    If the answer is not contained within the documentation, clearly say:
                    "I don't have enough information to answer this question."

                    Use the language of the question in your response.

                    --- DOCUMENTATION EXCERPTS START ---
                    {context}
                    --- DOCUMENTATION EXCERPTS END ---

                    Question: {question}

                    Answer:"""
        with timed_step(logger, "llm_generate"):
            response = self.__llm.invoke([
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt}
            ])
        seen_sources: Dict[str, Dict] = {}
        for doc in relevant_docs:
            key = json.dumps(doc["metadata"], sort_keys=True, default=str)
            seen_sources.setdefault(key, doc["metadata"])

        return {
            "answer": response.content,
            "sources": list(seen_sources.values())
        }
        
    def _retrieve(self, query: str) -> List[Dict]:
        """Retrieval: wide vector recall -> similarity floor -> (optional rerank) -> (optional MMR).

        Reranking and MMR are both optional and config-driven: when a cross-encoder
        degrades ranking it is disabled, and when the diversity block is absent MMR is
        skipped and the top final_k vector hits are returned directly.
        """
        # MMR is the only consumer of candidate embeddings; skip fetching them when off.
        with timed_step(logger, "vector_search"):
            candidates = self.__vector_store.search(
                query, k=self._candidate_k, with_embeddings=self.__diversity_selector is not None
            )
        if not candidates:
            return []

        if self._min_similarity > 0.0:
            with timed_step(logger, "similarity_filter"):
                candidates = [c for c in candidates if c.get("score", 1.0) >= self._min_similarity]
            if not candidates:
                return []

        if self.__reranker is not None:
            with timed_step(logger, "rerank"):
                candidates = self.__reranker.rerank(query, candidates, top_k=self._rerank_k)

        if self.__diversity_selector is None:
            return candidates[:self._final_k]

        with timed_step(logger, "mmr_select"):
            return self.__diversity_selector.select(
                candidates, top_k=self._final_k, lambda_mult=self._mmr_lambda
            )