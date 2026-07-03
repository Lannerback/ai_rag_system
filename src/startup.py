import logging
import time

from src.ai.service_factory import ServiceFactory
from src.ai.rag_facade import RagFacade


def initialize_rag_facade() -> RagFacade:
    return ServiceFactory.get_rag_facade()


def initialize_vector_store():
    """Initialize the configured vector store via its backend-specific strategy."""
    start_time = time.perf_counter()
    logging.info("Initializing vector store...")
    ServiceFactory.get_vector_store_initializer().initialize()
    logging.info("Initialization of vector store completed")
    duration_sec = time.perf_counter() - start_time
    duration_min = duration_sec / 60
    logging.info(f"Vector store initialized in {duration_min:.2f} minutes ({duration_sec:.2f} seconds)")
