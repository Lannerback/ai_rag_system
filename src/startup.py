import logging
import time

from src.ai.service_factory import ServiceFactory
from src.ai.rag_facade import RagFacade
from src.ai.vector_store_service.faiss.faiss_vector_store_initializer import VectorStoreInitializer
from src.ai.document_loaders.document_loader_facade import DocumentLoaderFacade
from src.ai.vector_store_service.base_vector_store import BaseVectorStore


def initialize_rag_facade() -> RagFacade:
    return ServiceFactory.get_rag_facade()

def initialize_vector_store():
    """Initialize the vector store."""
    start_time = time.perf_counter()
    logging.info("Initializing vector store...")
    vector_store: BaseVectorStore = ServiceFactory.get_faiss_vector_store()
    doc_loader_facade = DocumentLoaderFacade()    
    VectorStoreInitializer(vector_store, doc_loader_facade).initialize_vector_store()
    logging.info("Initialization of vector store completed")
    duration_sec = time.perf_counter() - start_time
    duration_min = duration_sec / 60
    logging.info(f"Vector store initialized in {duration_min:.2f} minutes ({duration_sec:.2f} seconds)")

