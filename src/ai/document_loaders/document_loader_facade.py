"""Facade providing a simple interface for loading documents from all sources."""
import logging
from typing import List, Tuple

from src.ai.document_loaders.base_document_loader import BaseDocumentLoader
from src.ai.document_loaders.document_loader_factory import DocumentLoaderFactory


class DocumentLoaderFacade:
    """
    Facade for document loading from text, OCR, and LLM extraction.
    
    No external dependencies needed - the factory creates everything internally.
    """
    
    def __init__(self):
        """
        Initialize document loader facade.
        """
        self.__loaders: List[BaseDocumentLoader] = DocumentLoaderFactory.create_all_loaders()
        logging.info(f"DocumentLoaderFacade initialized with {len(self.__loaders)} loaders")
    
    def load_all_documents(self) -> Tuple[List[str], List[dict]]:
        all_texts: List[str] = []
        all_metadatas: List[dict] = []
        
        for loader in self.__loaders:
            loader_name = loader.__class__.__name__
            logging.info(f"Loading documents with {loader_name}...")
            
            try:
                texts, metadatas = loader.load_documents()
                all_texts.extend(texts)
                all_metadatas.extend(metadatas)
                logging.info(f"✅ {loader_name} loaded {len(texts)} chunks")
            except Exception as e:
                logging.error(f"❌ {loader_name} failed: {e}")
        
        logging.info(f"Total documents loaded: {len(all_texts)} chunks from {len(self.__loaders)} sources")
        return all_texts, all_metadatas
    