from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.common.APIException import APIException

import logging
import os
import pytesseract
from pdf2image import convert_from_path
from typing import List, Tuple


class OcrDocumentLoader:
    
    def __init__(self, directory: str, chunk_size: int, chunk_overlap: int, lang: str):
        self.directory: str = directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.lang: str = lang

    
    def load_documents(self) -> Tuple[List[str], List[dict]]:
        """
        Load documents scanned from PDFs using OCR (pytesseract).
        Returns lists of texts and their metadata.
        """
        # Check if directory exists
        if not os.path.exists(self.directory) or not os.path.isdir(self.directory):
            logging.warning(f"Docs directory not found: {self.directory}")
            return [], []   

        logging.info(f"🔹 Running OCR on {self.directory} in lang {self.lang}")

        ocr_texts, ocr_metadatas = [], []

        for file in os.listdir(self.directory):
            if not file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(self.directory, file)
            logging.info(f"Processing PDF: {pdf_path}")

            try:
                images = convert_from_path(pdf_path)
            except Exception as e:
                logging.error(f"Failed to read {pdf_path}: {e}")
                raise APIException(
                    detail=f"PDF file is not valid {pdf_path}",
                    status_code=400,
                    code="invalid_docs",
                )

            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img, lang=self.lang)
                if text.strip():
                    # Split OCR text into smaller chunks
                    chunks = self.text_splitter.split_text(text)
                    for chunk in chunks:
                        ocr_texts.append(chunk)
                        ocr_metadatas.append({
                            "source": pdf_path,
                            "page": i + 1,
                            "lang": self.lang,
                            "ocr": True,
                        })

        if not ocr_texts:
            logging.warning(f"No text extracted from OCR in {self.directory}")

        return ocr_texts, ocr_metadatas
    
    