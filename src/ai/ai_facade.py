"""Facade for AI document loading, embedding, and retrieval logic."""
from dotenv import load_dotenv

from src.ai.ai_service import AiService
from src.ai.builder_dispatcher import BuilderDispatcher
load_dotenv()

class AiFacade:    

    def __init__(self):
        self.__builder_dispatcher = BuilderDispatcher()
        self.__ai_service = AiService(self.__builder_dispatcher.get_llm(),
                                      self.__builder_dispatcher.get_embedder_service())
               
  
    def answer_question(self, question: str, k: int = 3) -> dict:           
        return self.__ai_service.answer_question(question, k=k)