""" class that builds the needed embedder and llm based on the chosen envrionment (Gemini, Azure, etc.) """
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from src.common.APIException import APIException
from src.common.config import CONFIG

from .document_loader import DocumentLoader
from src.ai.base_llm import BaseLLM
from src.ai.base_embedder import BaseEmbedder

from .azure.azure_openai_llm import AzureLLM
from .azure.azure_embeddings import AzureEmbeddingStore
from src.ai.embedder_service import EmbedderService
from src.ai.gemini.gemini_embeddings import GeminiEmbeddingStore
from src.ai.gemini.gemini_llm import GeminiLLM

load_dotenv()

class BuilderDsipatcher:    

    def __init__(self):
        self.__llm: BaseLLM = self._build_llm()
        self.__loader: DocumentLoader = self._build_loader()
        self.__embedder_store: BaseEmbedder = self._build_store()    
                        

    def _build_llm(self) -> BaseLLM:
        llm_provider = CONFIG["providers"]["llm"]
        
        if llm_provider == 'azure':
            return AzureLLM()
        elif llm_provider == 'gemini':
            return GeminiLLM()
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    def _build_loader(self) -> DocumentLoader:
        return DocumentLoader()

    def _build_store(self) -> BaseEmbedder:
        embeddings_provider = CONFIG["providers"]["embeddings"]
        
        if embeddings_provider == 'azure':
            return AzureEmbeddingStore()
        elif embeddings_provider == 'gemini':
            return GeminiEmbeddingStore()
        else:
            raise ValueError(f"Unsupported embeddings provider: {embeddings_provider}")
    
    def _build_embedder_service(self,store) -> EmbedderService:
        return EmbedderService(store)
    
    def get_llm(self) -> BaseLLM:
        """Get the LLM instance."""
        return self.__llm
    
    def get_loader(self) -> DocumentLoader:
        """Get the DocumentLoader instance."""
        return self.__loader
    
    def get_embedder_store(self) -> BaseEmbedder:
        """Get the BaseEmbedder instance."""
        return self.__embedder_store  