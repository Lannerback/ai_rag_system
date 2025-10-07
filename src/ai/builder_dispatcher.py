""" class that builds the needed embedder and llm based on the chosen envrionment (Gemini, Azure, etc.) """
from src.common.config import CONFIG

from src.ai.base_llm import BaseLLM
from src.ai.embedders.base_embedder import BaseEmbedder
from src.ai.embedders.azure.azure_openai_llm import AzureLLM
from src.ai.embedders.azure.azure_embedder import AzureEmbedder
from src.ai.embedders.gemini.gemini_embedder import GeminiEmbedder
from src.ai.embedders.gemini.gemini_llm import GeminiLLM

class BuilderDispatcher:    
    # Map provider names to their respective LLM and EmbeddingStore classes
    LLM_PROVIDERS = {
        'azure': AzureLLM,
        'gemini': GeminiLLM
    }

    EMBEDDING_PROVIDERS = {
        'azure': AzureEmbedder,
        'gemini': GeminiEmbedder
    }

    def __init__(self):
        self.__llm: BaseLLM = self._build_llm()
        self.__embedder_store: BaseEmbedder = self._build_embedder_store()
                        

    def _build_llm(self) -> BaseLLM:
        llm_provider_name = CONFIG["llm"]["provider"]
        llm_class: BaseLLM = self.LLM_PROVIDERS.get(llm_provider_name)
        return llm_class()


    def _build_embedder_store(self) -> BaseEmbedder:
        embeddings_provider_name = CONFIG["llm"]["provider"]
        embedding_class: BaseEmbedder = self.EMBEDDING_PROVIDERS.get(embeddings_provider_name)
        return embedding_class()
    
    
    def get_llm(self) -> BaseLLM:
        """Get the LLM instance."""
        return self.__llm
    
    def get_embedder_store(self) -> BaseEmbedder:
        """Get the EmbedderStore instance."""
        return self.__embedder_store
    