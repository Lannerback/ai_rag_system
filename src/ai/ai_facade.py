"""Facade for AI document loading, embedding, and retrieval logic."""
from dotenv import load_dotenv

from src.ai.ai_service import AiService

load_dotenv()

class AiFacade:    

    def __init__(self):
        self.__ai_service = AiService()
               
  
    def answer_question(self, question: str, k: int = 3) -> dict:           
        return self.__ai_service.answer_question(question, k=k)