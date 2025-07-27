# src/llm/azure_openai_llm.py
from langchain_openai import AzureChatOpenAI
import os

from src.ai.base_llm import BaseLLM

class AzureLLM(BaseLLM):
    def __init__(self):
        self.__llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            temperature=0.7,
            max_tokens=500
        )

    def invoke(self, messages: list[dict]):
        response = self.__llm.invoke(messages)
        return response.content
