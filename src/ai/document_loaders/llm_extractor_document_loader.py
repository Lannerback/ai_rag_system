from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
from src.common.APIException import APIException

import logging
import os
import base64
import io
from pdf2image import convert_from_path
from typing import List, Tuple
from src.ai.base_llm import BaseLLM


class LlmExtractorDocumentLoader:
    
    def __init__(self, directory: str, chunk_size: int, chunk_overlap: int, lang: str, base_llm: BaseLLM):
        """
        Initialize LLM-based document extractor for scanned PDFs.
        
        Args:
            directory: Directory containing PDF files to process
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            lang: Language of the documents
            langchain_llm: Optional LangChain LLM instance with vision capabilities (e.g., ChatGoogleGenerativeAI)
        """
        self.__directory: str = directory
        self.__text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.__lang: str = lang
        self.__llm = base_llm

    
    def load_documents(self) -> Tuple[List[str], List[dict]]:
        """
        Load documents from scanned PDFs using Gemini LLM extraction.
        Returns lists of texts and their metadata.
        """
        # Check if directory exists
        if not os.path.exists(self.__directory) or not os.path.isdir(self.__directory):
            logging.warning(f"Docs directory not found: {self.__directory}")
            return [], []   

        logging.info(f"🔹 Running LLM extraction on {self.__directory} in lang {self.__lang}")

        extracted_texts, extracted_metadatas = [], []

        for file in os.listdir(self.__directory):
            if not file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(self.__directory, file)
            self._process_pdf(pdf_path, file, extracted_texts, extracted_metadatas)

        if not extracted_texts:
            logging.warning(f"No text extracted from LLM in {self.__directory}")

        return extracted_texts, extracted_metadatas
    
    def _process_pdf(self, pdf_path: str, filename: str, texts: List[str], metadatas: List[dict]) -> None:
        """Process a single PDF file and extract text from all pages."""
        logging.info(f"Processing PDF: {pdf_path}")
        
        try:
            # convert the pdf to images
            images = convert_from_path(pdf_path)
        except Exception as e:
            logging.error(f"Failed to read {pdf_path}: {e}")
            raise APIException(
                detail=f"PDF file is not valid {pdf_path}",
                status_code=400,
                code="invalid_docs",
            )
        
        for i, img in enumerate(images, start=1):
            # extract the text from the image
            self._extract_text_from_page(img, pdf_path, filename, i, texts, metadatas)
    
    
    def _extract_text_from_page(self, img, pdf_path: str, filename: str, page_num: int, 
                                  texts: List[str], metadatas: List[dict]) -> None:
        """Extract text from a single page using LLM."""
        img_b64 = self._pil_to_base64(img)
        task = (
            f"Extract all the text from this scanned PDF page in the iso3 code language: {self.__lang}. "
            "Remove headers/footers, keep paragraph structure."
        )  
        
        try:
            response = self.__llm.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": task},
                    {"type": "image_url", "image_url": img_b64}
                ])
            ])
            
            text = response.content
            logging.info(f"✅ Page {page_num} of {filename} processed")
            
            if text.strip():
                chunks = self.__text_splitter.split_text(text)
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append({
                        "source": pdf_path,
                        "page": page_num,
                        "lang": self.__lang,
                        "llm_extracted": True,
                    })
        except Exception as e:
            logging.error(f"Failed to extract text from page {page_num} of {pdf_path}: {e}")
    
    
    def _pil_to_base64(self, img, format: str = "PNG") -> str:
        """Convert a PIL Image to base64 string with proper header."""
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return f"data:image/{format.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()
    