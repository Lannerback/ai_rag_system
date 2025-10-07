from src.ai.vector_store_service.faiss.faiss_vector_store import FaissVectorStore
from src.ai.document_loaders.document_loader_facade import DocumentLoaderFacade
import logging
from typing import List, Dict
from src.common.app_context import get_app_context
from src.ai.vector_store_service.base_vector_store import BaseVectorStore

class VectorStoreFacade:
    def __init__(self):
        self.__vector_store: BaseVectorStore = FaissVectorStore(get_app_context().state.embedder)

    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        self.__vector_store.add_documents(texts, metadatas)

    def search(self, question: str, k: int) -> List[Dict]:
        return self.__vector_store.search(question, k=k)

               