"""Facade for AI that exposes the AI service to the API."""

from src.ai.rag_service import RagService


class RagFacade:        

    def __init__(self, rag_service: RagService):
        self.__rag_service = rag_service
  
    def answer_question(self, question: str) -> dict:           
        return self.__rag_service.answer_question(question)