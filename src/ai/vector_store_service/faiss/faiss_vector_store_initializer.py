import logging
from src.ai.vector_store_service.faiss.faiss_vector_store import FaissVectorStore
from src.ai.document_loaders.document_loader_facade import DocumentLoaderFacade
from src.common.app_context import get_app_context

class VectorStoreInitializer:
    """Handles loading or building the vector store."""

    def __init__(self):
        self.__vector_store = FaissVectorStore(get_app_context().state.embedder)

    def initialize_vector_store(self):
        """
        Initialize the vector store from documents.
        
        Loads from disk if available, otherwise builds from all document sources.
        The facade creates any needed dependencies (like LLM for extraction) internally.
        """
        if not self.__vector_store.load_from_disk():
            logging.info("Nothing found in vector store, building from documents")
            
            # Use facade to load all documents from all sources
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