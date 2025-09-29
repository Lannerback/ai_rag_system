"""Service class that manages the AI document loading, embedding, and retrieval logic."""
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.common.APIException import APIException
from src.ai.base_llm import BaseLLM

from src.ai.embedder_service import EmbedderService
from src.common.config import CONFIG
import logging
import os
import pytesseract
from pdf2image import convert_from_path
from typing import Optional

load_dotenv()

class AiService:    

    def __init__(self,llm: BaseLLM, embedder_service: EmbedderService):
        self.__llm: BaseLLM = llm
        self.__embedder_service: EmbedderService = embedder_service
        
        # Load AI service configuration
        self._system_prompt = CONFIG["ai_service"]["system_prompt"]
        self._default_k = CONFIG["ai_service"]["default_k"]
        
        # Initialize the vector store from documents            
        self._initialize_store()
               

    def _initialize_store(self):
        """Load or build the vector store from documents."""
        if not self.__embedder_service.load_from_disk():
            texts, metadatas = [], []

            # Load "normal" text-based documents
            docs_texts, docs_metadatas = self._load_documents()
            texts.extend(docs_texts)
            metadatas.extend(docs_metadatas)

            # Load OCR-based scanned PDFs
            ocr_texts, ocr_metadatas = self._load_documents_ocr(CONFIG["document_loader"]["scanned_docs_lang"])
            texts.extend(ocr_texts)
            metadatas.extend(ocr_metadatas)

            self.__embedder_service.add_documents(texts, metadatas)
            self.__embedder_service.save_to_disk()

    def _get_relevant_docs(self, query, k=None):
        """Retrieve relevant documents for a query."""
        k = k or self._default_k
        return self.__embedder_service.search(query, k=k)

    def answer_question(self, question: str, k: int = None) -> dict:
        """Answer a question using the LLM and relevant docs."""
        k = k or self._default_k
        relevant_docs = self._get_relevant_docs(question, k=k)
        if not relevant_docs:
            raise APIException(detail = "No relevant documentation found for the question.", status_code=400, code = "no_docs_found")

        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        prompt = f"""Based on the following documentation excerpts, please answer the question.
                        If you cannot answer the question based on the provided context, say so.

                        Documentation:
                        {context}

                        Question: {question}

                        Answer:"""
        response = self.__llm.invoke([
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt}
        ])
        return {
            "answer": response,
            "sources": list({frozenset(doc["metadata"].items()): doc["metadata"] for doc in relevant_docs}.values())
        }
        
    def _load_documents(self):
        """Load documents using langchain."""
        docs_dir: str = CONFIG["document_loader"]["docs_directory"]
        
         # Check if directory exists
        if not os.path.exists(docs_dir) or not os.path.isdir(docs_dir):
            logging.warning(f"Docs directory for not found: {docs_dir}")
            return [], []
        
        doc_loader_cfg = CONFIG.get("document_loader", {})

        # Validate existence
        if "chunk_size" not in doc_loader_cfg or doc_loader_cfg["chunk_size"] is None:
            raise ValueError("Missing required config: document_loader.chunk_size")

        if "chunk_overlap" not in doc_loader_cfg or doc_loader_cfg["chunk_overlap"] is None:
            raise ValueError("Missing required config: document_loader.chunk_overlap")


        loader = DirectoryLoader(CONFIG["document_loader"]["docs_directory"], glob="**/*", loader_cls=UnstructuredFileLoader)
        documents = loader.load()
        chunk_size = CONFIG["document_loader"]["chunk_size"]
        chunk_overlap = CONFIG["document_loader"]["chunk_overlap"]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_documents = text_splitter.split_documents(documents)

        texts = [doc.page_content for doc in split_documents]
        metadatas = [doc.metadata for doc in split_documents]
        return texts, metadatas
    
    
    def _load_documents_ocr(self, lang: str):
        """
        Load documents scanned from images to text using OCR
        """
        
        pdf_dir: str = CONFIG["document_loader"]["scanned_docs_dir"]
        
        # Check if directory exists
        if not os.path.exists(pdf_dir) or not os.path.isdir(pdf_dir):
            logging.warning(f"OCR directory for scanned_docs not found: {pdf_dir}")
            return [], []
        
        logging.info(f"🔹 Running OCR on {pdf_dir} in lang {lang}")

        ocr_texts = []
        ocr_metadatas = []

        for file in os.listdir(pdf_dir):
            if not file.lower().endswith(".pdf"):
                continue 

            pdf_path = os.path.join(pdf_dir, file)
            logging.info(f"Processing pdf: {pdf_path}")

            try:
                images = convert_from_path(pdf_path)
            except Exception as e:
                logging.error(f"Failed to read {pdf_path}: {e}")
                raise APIException(detail = f"PDF file is not valid {pdf_path}", status_code=400, code = "invalid_docs")

            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img, lang=lang)
                if text.strip():
                    ocr_texts.append(text)
                    ocr_metadatas.append({
                        "source": pdf_path,
                        "page": i + 1,
                        "lang": lang,
                        "ocr": True
                    })
                #logging.debug(f"{pdf_path} page {i+1}: {len(text)} chars extracted")

        if not ocr_texts:
            logging.warning(f"No text extracted in {pdf_dir} with OCR")

        return ocr_texts, ocr_metadatas
        