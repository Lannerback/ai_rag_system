"""Service class that manages the AI document loading, embedding, and retrieval logic."""
from src.common.APIException import APIException
from src.ai.base_llm import BaseLLM
from src.ai.vector_store_service.vector_store_facade import VectorStoreFacade
from src.common.config import CONFIG
import logging
from typing import List, Dict

class RagService:    

    def __init__(self, llm: BaseLLM, vector_store: VectorStoreFacade):
        self.__llm: BaseLLM = llm
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

        # TODO: apply top_p_filter to the relevant docs
        #relevant_docs = top_p_filter(relevant_docs, p=0.9)
        
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
        response = self.__llm.invoke([
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt}
        ])

        chunks = []
        for doc in relevant_docs:
            metadata = doc.get("metadata") or {}
            chunks.append({
                "chunk_id": metadata.get("chunk_id"),
                "chunk_index": metadata.get("chunk_index"),
                "faiss_index": metadata.get("faiss_index"),
                "rank": metadata.get("retrieval_rank"),
                "score": metadata.get("retrieval_score"),
                "source": metadata.get("source", metadata.get("file_path", "unknown")),
                "page": metadata.get("page", "?"),
            })

        source_metadata_keys_to_exclude = {
            "chunk_id",
            "chunk_index",
            "faiss_index",
            "retrieval_rank",
            "retrieval_score",
        }

        unique_sources = []
        seen = set()
        for doc in relevant_docs:
            metadata = dict(doc.get("metadata") or {})
            for key in source_metadata_keys_to_exclude:
                metadata.pop(key, None)
            frozen = frozenset(metadata.items())
            if frozen in seen:
                continue
            seen.add(frozen)
            unique_sources.append(metadata)

        return {
            "answer": response.content,
            "sources": unique_sources,
            "chunks": chunks,
        }
        
    def _get_relevant_docs(self, query, k) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        return self.__vector_store.search(query, k=k)
