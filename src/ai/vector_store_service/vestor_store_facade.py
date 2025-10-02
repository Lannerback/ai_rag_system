from src.ai.vector_store_service.faiss_vector_store import FaissVectorStore
from src.ai.document_loaders.document_loader_facade import DocumentLoaderFacade
from src.ai.base_llm import BaseLLM
import logging
from src.ai.vector_store_service.base_embedder import BaseEmbedder
from typing import List, Dict

class VectorStoreFacade:
    def __init__(self, embedder: BaseEmbedder):
        self.__vector_store = FaissVectorStore(embedder)

    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        self.__vector_store.add_documents(texts, metadatas)

    def search(self, question: str, k: int = 3) -> List[Dict]:
        return self.__vector_store.search(question, k=k)
    
    def initialize_vector_store(self, base_llm: BaseLLM):
        document_loader_facade: DocumentLoaderFacade = DocumentLoaderFacade(base_llm)
        """Initialize the vector store from documents. Should be called at startup."""
        if not self.__vector_store.load_from_disk():
            logging.info("Nothing found in vector store, building from documents")
            texts, metadatas = [], []

            # Load "normal" text-based documents
            logging.info("Loading text-based documents")
            docs_texts, docs_metadatas = document_loader_facade.text_document_loader.load_documents()
            texts.extend(docs_texts)
            metadatas.extend(docs_metadatas)

            # Load OCR-based scanned PDFs
            logging.info("Loading OCR-based scanned PDFs")
            ocr_texts, ocr_metadatas = document_loader_facade.ocr_document_loader.load_documents()
            texts.extend(ocr_texts)
            metadatas.extend(ocr_metadatas)

            # Load LLM-extracted scanned PDFs (if available)
            if document_loader_facade.llm_extractor_document_loader is not None:
                logging.info("Loading LLM-extracted scanned PDFs")
                llm_texts, llm_metadatas = document_loader_facade.llm_extractor_document_loader.load_documents()
                texts.extend(llm_texts)
                metadatas.extend(llm_metadatas)

            if not texts or not metadatas:
                logging.warning("No documents found to initialize vector store")
                return
            
            self.__vector_store.add_documents(texts, metadatas)
            self.__vector_store.save_to_disk()
            logging.info("Vector store initialized successfully with documents") 
        else:
            logging.info("Vector store loaded from disk")
               