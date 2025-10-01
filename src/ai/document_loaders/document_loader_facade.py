from src.ai.vector_store.faiss_vector_store import FaissVectorStore
from src.common.config import CONFIG
from src.ai.document_loaders.text_document_loader import TextDocumentLoader
from src.ai.document_loaders.ocr_document_loader import OcrDocumentLoader
import logging



class DocumentLoaderFacade:
    def __init__(self):
        self.text_document_loader: TextDocumentLoader = TextDocumentLoader(CONFIG["document_loader"]["docs_directory"], 
                                                                           CONFIG["document_loader"]["chunk_size"], 
                                                                           CONFIG["document_loader"]["chunk_overlap"])
        self.ocr_document_loader: OcrDocumentLoader = OcrDocumentLoader(CONFIG["document_loader"]["scanned_docs_dir"], 
                                                                        CONFIG["document_loader"]["chunk_size"], 
                                                                        CONFIG["document_loader"]["chunk_overlap"], 
                                                                        CONFIG["document_loader"]["scanned_docs_lang"])
        
        
    def _initialize_store(self):
        """Load or build the vector store from documents."""
        if not self.__embedder_service.load_from_disk():
            texts, metadatas = [], []

            # Load "normal" text-based documents
            docs_texts, docs_metadatas = self.text_document_loader.load_documents()
            texts.extend(docs_texts)
            metadatas.extend(docs_metadatas)

            # Load OCR-based scanned PDFs
            ocr_texts, ocr_metadatas = self.ocr_document_loader.load_documents()
            texts.extend(ocr_texts)
            metadatas.extend(ocr_metadatas)

            self.__embedder_service.add_documents(texts, metadatas)
            self.__embedder_service.save_to_disk()

    