from src.ai.vector_store_service.faiss_vector_store import FaissVectorStore
from src.ai.document_loaders.document_loader_facade import DocumentLoaderFacade
import logging
from typing import List, Dict

class VectorStoreFacade:
    def __init__(self):
        self.__vector_store = FaissVectorStore()

    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        self.__vector_store.add_documents(texts, metadatas)

    def search(self, question: str, k: int) -> List[Dict]:
        return self.__vector_store.search(question, k=k)
    
    def initialize_vector_store(self):
        """
        Initialize the vector store from documents.
        
        Loads from disk if available, otherwise builds from all document sources.
        The facade creates any needed dependencies (like LLM for extraction) internally.
        """
        if not self.__vector_store.load_from_disk():
            logging.info("Nothing found in vector store, building from documents")
            
            # Use facade to load all documents from all sources
            # No LLM needed - facade creates it internally if needed!
            document_loader_facade = DocumentLoaderFacade()
            texts, metadatas = document_loader_facade.load_all_documents()

            if not texts or not metadatas:
                logging.warning("No documents found to initialize vector store")
                return
            
            self.__vector_store.add_documents(texts, metadatas)
            self.__vector_store.save_to_disk()
            logging.info("Vector store initialized successfully with documents") 
        else:
            logging.info("Vector store loaded from disk")
               