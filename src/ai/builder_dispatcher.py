""" class that builds the needed embedder and llm based on the chosen envrionment (Gemini, Azure, etc.) """
from dotenv import load_dotenv
from src.common.config import CONFIG

from src.ai.base_llm import BaseLLM
from src.ai.vector_store.base_embedder import BaseEmbedder

from .azure.azure_openai_llm import AzureLLM
from .azure.azure_embeddings import AzureEmbeddingStore
from src.ai.vector_store.faiss_vector_store import FaissVectorStore
from src.ai.gemini.gemini_embeddings import GeminiEmbeddingStore
from src.ai.gemini.gemini_llm import GeminiLLM

load_dotenv()

class BuilderDispatcher:    
    # Map provider names to their respective LLM and EmbeddingStore classes
    LLM_PROVIDERS = {
        'azure': AzureLLM,
        'gemini': GeminiLLM
    }

    EMBEDDING_PROVIDERS = {
        'azure': AzureEmbeddingStore,
        'gemini': GeminiEmbeddingStore
    }

    def __init__(self):
        self.__llm: BaseLLM = self._build_llm()
        self.__embedder_store: BaseEmbedder = self._build_store()
        self.__embedder_service: FaissVectorStore = self._build_embedder_service(self.__embedder_store)
                        

    def _build_llm(self) -> BaseLLM:
        llm_provider_name = CONFIG["providers"]["llm"]
        llm_class = self.LLM_PROVIDERS.get(llm_provider_name)
        if llm_class:
            return llm_class()
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider_name}")


    def _build_store(self) -> BaseEmbedder:
        embeddings_provider_name = CONFIG["providers"]["embeddings"]
        embedding_class = self.EMBEDDING_PROVIDERS.get(embeddings_provider_name)
        if embedding_class:
            return embedding_class()
        else:
            raise ValueError(f"Unsupported embeddings provider: {embeddings_provider_name}")
    
    def _build_embedder_service(self,store) -> FaissVectorStore:
        return FaissVectorStore(store)
    
    def get_llm(self) -> BaseLLM:
        """Get the LLM instance."""
        return self.__llm
    
    
    def get_embedder_service(self) -> FaissVectorStore:
        """Get the FaissVectorStore instance."""
        return self.__embedder_service  