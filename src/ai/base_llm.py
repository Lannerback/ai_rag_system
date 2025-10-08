# src/llm/base_llm.py
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, messages: list[dict]):
        """
        Invoke the LLM with a list of messages.
        Returns the response object (with .content attribute).
        """
        pass