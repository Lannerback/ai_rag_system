"""Facade for AI that exposes the AI service to the API."""

from src.ai.rag_service import RagService
from src.ai.vector_store_service.vestor_store_facade import VectorStoreFacade

class RagFacade:        

    def __init__(self):
        self.__vector_store: VectorStoreFacade = VectorStoreFacade()        
        self.__rag_service = RagService(self.__vector_store)
               
    def initialize_vector_store(self):
        """
        Initialize the vector store. Should be called at application startup.
        
        No LLM parameter needed - the document loaders create any needed
        dependencies internally, removing coupling.
        """
        self.__vector_store.initialize_vector_store()
   
    def answer_question(self, question: str) -> dict:           
        return self.__rag_service.answer_question(question)