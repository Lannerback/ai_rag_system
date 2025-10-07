"""Factory for creating document loaders based on configuration."""
import logging
from typing import List, Optional

from src.common.config import CONFIG
from src.ai.base_llm import BaseLLM
from src.ai.document_loaders.base_document_loader import BaseDocumentLoader
from src.ai.document_loaders.text_document_loader import TextDocumentLoader
from src.ai.document_loaders.ocr_document_loader import OcrDocumentLoader
from src.ai.document_loaders.llm_extractor_document_loader import LlmExtractorDocumentLoader


class DocumentLoaderFactory:
    """
    Factory for creating document loaders.
    
    Handles the complexity of creating different loader types with their
    specific dependencies and configuration.
    """
    
    @staticmethod
    def create_text_loader() -> BaseDocumentLoader:
        """Create a text document loader from config."""
        return TextDocumentLoader(
            directory=CONFIG["document_loader"]["docs_directory"],
            chunk_size=CONFIG["document_loader"]["chunk_size"],
            chunk_overlap=CONFIG["document_loader"]["chunk_overlap"]
        )
    
    @staticmethod
    def create_ocr_loader() -> BaseDocumentLoader:
        """Create an OCR document loader from config."""
        return OcrDocumentLoader(
            directory=CONFIG["document_loader"]["ocr_docs_dir"],
            chunk_size=CONFIG["document_loader"]["chunk_size"],
            chunk_overlap=CONFIG["document_loader"]["chunk_overlap"],
            lang=CONFIG["document_loader"]["scanned_docs_lang"]
        )
    
    @staticmethod
    def create_llm_extractor_loader() -> BaseDocumentLoader:
        """
        Create an LLM extractor document loader.
        """
        return LlmExtractorDocumentLoader(
                directory=CONFIG["document_loader"]["llm_extractor_docs_dir"],
                chunk_size=CONFIG["document_loader"]["chunk_size"],
                chunk_overlap=CONFIG["document_loader"]["chunk_overlap"],
                lang=CONFIG["document_loader"]["scanned_docs_lang"]
            )
  
    
    @staticmethod
    def create_all_loaders() -> List[BaseDocumentLoader]:
        """
        Create all configured document loaders. 
        Returns:
            List of all available document loaders
        """
        loaders: List[BaseDocumentLoader] = []
        
        loaders.append(DocumentLoaderFactory.create_text_loader())
        logging.debug("✅ Created text document loader")
        
        loaders.append(DocumentLoaderFactory.create_ocr_loader())
        logging.debug("✅ Created OCR document loader")
        
        logging.info(f"Created {len(loaders)} standard document loaders")        
        # Add LLM extractor loader if requested
        llm_loader = DocumentLoaderFactory.create_llm_extractor_loader()
        loaders.append(llm_loader)
        logging.debug("✅ Created LLM extractor document loader")
        
        logging.info(f"Created {len(loaders)} document loaders total")
        return loaders

