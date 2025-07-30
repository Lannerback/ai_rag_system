"""Facade for AI document loading, embedding, and retrieval logic."""
from dotenv import load_dotenv
from src.common.APIException import APIException
from src.ai.base_llm import BaseLLM
from src.ai.base_embedder import BaseEmbedder

from src.ai.embedder_service import EmbedderService
from src.common.config import CONFIG

from .document_loader import DocumentLoader

load_dotenv()

class AiService:    

    def __init__(self,llm: BaseLLM,loader: DocumentLoader,embedder_store: BaseEmbedder):
        self.__llm: BaseLLM = llm
        self.__loader: DocumentLoader = loader
        self.embedder_store: BaseEmbedder = embedder_store   
        self.__embedder_service: EmbedderService = EmbedderService(self.embedder_store)
        
        # Load AI service configuration
        self._system_prompt = CONFIG["ai_service"]["system_prompt"]
        self._default_k = CONFIG["ai_service"]["default_k"]
        
        # Initialize the vector store from documents            
        self._initialize_store()
               

    def _initialize_store(self):
        """Load or build the vector store from documents."""
        if not self.__embedder_service.load_from_disk():
            documents = self.__loader.load_documents()
            self.__embedder_service.add_documents(
                texts=[doc["content"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents]
            )
            self.__embedder_service.save_to_disk()

    def _get_relevant_docs(self, query, k=None):
        """Retrieve relevant documents for a query."""
        k = k or self._default_k
        return self.__embedder_service.search(query, k=k)

    def answer_question(self, question: str, k: int = None) -> dict:
        """Answer a question using the LLM and relevant docs."""
        k = k or self._default_k
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
            "answer": response,
            "sources": list({frozenset(doc["metadata"].items()): doc["metadata"] for doc in relevant_docs}.values())
        }