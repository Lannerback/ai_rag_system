from src.common.config import CONFIG
from src.ai.document_loaders.text_document_loader import TextDocumentLoader
from src.ai.document_loaders.ocr_document_loader import OcrDocumentLoader
from src.ai.document_loaders.llm_extractor_document_loader import LlmExtractorDocumentLoader
from src.ai.base_llm import BaseLLM
from typing import Optional, Any
import logging



class DocumentLoaderFacade:
    def __init__(self, base_llm: BaseLLM):
        """
        Initialize document loaders.
        
        Args:
            langchain_llm: Optional LangChain LLM instance for LLM-based extraction (e.g., ChatGoogleGenerativeAI)
        """
        self.text_document_loader: TextDocumentLoader = TextDocumentLoader(
            CONFIG["document_loader"]["docs_directory"], 
            CONFIG["document_loader"]["chunk_size"], 
            CONFIG["document_loader"]["chunk_overlap"]
        )
        
        # OCR document loader
        self.ocr_document_loader: OcrDocumentLoader = OcrDocumentLoader(
            CONFIG["document_loader"]["ocr_docs_dir"], 
            CONFIG["document_loader"]["chunk_size"], 
            CONFIG["document_loader"]["chunk_overlap"], 
            CONFIG["document_loader"]["scanned_docs_lang"]
        )
        
        # LLM extractor document loader
        try:
            self.llm_extractor_document_loader = LlmExtractorDocumentLoader(
                CONFIG["document_loader"]["llm_extractor_docs_dir"], 
                CONFIG["document_loader"]["chunk_size"], 
                CONFIG["document_loader"]["chunk_overlap"], 
                CONFIG["document_loader"]["scanned_docs_lang"],
                base_llm
            )
            logging.info("✅ LLM extractor document loader initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize LLM extractor: {e}")
    