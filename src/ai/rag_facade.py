"""Facade for AI that exposes the AI service to the API."""

from src.ai.rag_service import RagService
from src.ai.vector_store_service.vector_store_facade import VectorStoreFacade

class RagFacade:        

    def __init__(self):
        self.__rag_service = RagService(VectorStoreFacade())
  
    def answer_question(self, question: str) -> dict:           
        return self.__rag_service.answer_question(question)