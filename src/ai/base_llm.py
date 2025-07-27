# src/llm/base_llm.py
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, messages: list[dict]) -> any:
        pass
