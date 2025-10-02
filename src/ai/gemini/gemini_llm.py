# src/llm/gemini_llm.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from src.ai.base_llm import BaseLLM
from src.common.config import CONFIG

class GeminiLLM(BaseLLM):
    def __init__(self):
        self.__llm = ChatGoogleGenerativeAI(
            model=CONFIG["gemini"]["model"],
            temperature=CONFIG["llm"]["gemini"]["temperature"],
            top_p=CONFIG["llm"]["gemini"]["top_p"],
            api_key=os.getenv("GOOGLE_API_KEY")
        )

    def invoke(self, messages: list[dict]) -> BaseMessage:
        response = self.__llm.invoke(messages)
        return response