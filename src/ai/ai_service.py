"""Facade for AI document loading, embedding, and retrieval logic."""
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from src.common.APIException import APIException

from .document_loader import DocumentLoader
from src.ai.base_llm import BaseLLM
from src.ai.base_embedder import BaseEmbedder

from .azure.azure_openai_llm import AzureLLM
from .azure.azure_embeddings import AzureEmbeddingStore
from src.ai.embedder_service import EmbedderService

load_dotenv()

class AiService:    

    def __init__(self):
        self.__llm: BaseLLM = self._build_llm()
        self.__loader: DocumentLoader = self._build_loader()
        self.__store: BaseEmbedder = self._build_store()    
        self.__embedder_service: EmbedderService = self._build_embedder_service(self.__store)
        
            
        self._initialize_store()
        

    
    def _build_llm(self) -> AzureChatOpenAI:
        return AzureLLM()
    
    def _build_loader(self) -> DocumentLoader:
        return DocumentLoader()

    def _build_store(self) -> BaseEmbedder:        
        return AzureEmbeddingStore()
    
    def _build_embedder_service(self,store) -> EmbedderService:
        return EmbedderService(store)

    def _initialize_store(self):
        """Load or build the vector store from documents."""
        if not self.__embedder_service.load_from_disk():
            documents = self.__loader.load_documents()
            self.__embedder_service.add_documents(
                texts=[doc["content"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents]
            )
            self.__embedder_service.save_to_disk()

    def _get_relevant_docs(self, query, k=3):
        """Retrieve relevant documents for a query."""
        return self.__embedder_service.search(query, k=k)

    def answer_question(self, question: str, k: int = 3) -> dict:
        """Answer a question using the LLM and relevant docs."""
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
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documentation."},
            {"role": "user", "content": prompt}
        ])
        return {
            "answer": response.content,
            "sources": [doc["metadata"] for doc in relevant_docs]
        }