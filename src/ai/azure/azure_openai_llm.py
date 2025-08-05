# src/llm/azure_openai_llm.py
from langchain_openai import AzureChatOpenAI
import os

from src.ai.base_llm import BaseLLM
from src.common.config import CONFIG

class AzureLLM(BaseLLM):
    def __init__(self):
        self.__llm = AzureChatOpenAI(
            azure_deployment=CONFIG["azure"]["deployment"],
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=CONFIG["azure"]["api_version"],
            temperature=CONFIG["llm"]["azure"]["temperature"],
            max_tokens=CONFIG["llm"]["azure"]["max_tokens"]
        )

    def invoke(self, messages: list[dict]):
        response = self.__llm.invoke(messages)
        return response.content
