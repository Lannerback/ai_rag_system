"""Facade for AI that exposes the AI service to the API."""

from src.ai.rag_service import RagService
from src.ai.builder_dispatcher import BuilderDispatcher
from src.ai.document_loaders.document_loader_facade import DocumentLoaderFacade
from src.common.config import CONFIG
class RagFacade:    

    def __init__(self):
        self.__builder_dispatcher = BuilderDispatcher()
        self.__rag_service = RagService(self.__builder_dispatcher.get_llm(),
                                        self.__builder_dispatcher.get_embedder_service(),
                                        CONFIG["llm"]["system_prompt"],
                                        CONFIG["llm"]["default_k"])
        self.__document_loader_facade = DocumentLoaderFacade()
               
    def initialize_vector_store(self):
        """Initialize the vector store. Should be called at application startup."""
        self.__rag_service.initialize_vector_store(self.__document_loader_facade)
  
    def answer_question(self, question: str, k: int = 3) -> dict:           
        return self.__rag_service.answer_question(question, k=k)