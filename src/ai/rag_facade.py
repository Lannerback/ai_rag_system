"""Facade for AI that exposes the AI service to the API."""

from src.ai.rag_service import RagService
from src.ai.builder_dispatcher import BuilderDispatcher
from src.ai.document_loaders.document_loader_facade import DocumentLoaderFacade
from src.ai.base_llm import BaseLLM
from src.common.config import CONFIG
from src.ai.vector_store_service.vestor_store_facade import VectorStoreFacade
from src.ai.vector_store_service.base_embedder import BaseEmbedder

class RagFacade:        

    def __init__(self):
        self.__builder_dispatcher = BuilderDispatcher()
        self.__llm: BaseLLM = self.__builder_dispatcher.get_llm()
        self.__embedder: BaseEmbedder = self.__builder_dispatcher.get_embedder_store()
        self.__vector_store: VectorStoreFacade = VectorStoreFacade(self.__embedder)
        
        self.__rag_service = RagService(self.__llm, self.__vector_store)
               
    def initialize_vector_store(self):
        """Initialize the vector store. Should be called at application startup."""
        self.__vector_store.initialize_vector_store(self.__llm)
   
    def answer_question(self, question: str, k: int = 3) -> dict:           
        return self.__rag_service.answer_question(question, k=k)