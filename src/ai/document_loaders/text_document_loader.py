import logging
import os
from typing import List, Tuple

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ai.document_loaders.base_document_loader import BaseDocumentLoader


class TextDocumentLoader(BaseDocumentLoader):
    """Document loader for text-based files (Markdown, PDF, DOCX)."""
    
    def __init__(self, directory: str, chunk_size: int, chunk_overlap: int):
        self.directory = directory
        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_documents(self) -> Tuple[List[str], List[dict]]:
        """Load documents from the directory, split into chunks, and return texts + metadata."""
        if not os.path.exists(self.directory) or not os.path.isdir(self.directory):
            logging.warning(f"Docs directory not found: {self.directory}")
            return [], []

        documents = self._load_directory_documents()
        split_documents = self.text_splitter.split_documents(documents)

        texts = [doc.page_content for doc in split_documents]
        metadatas = [doc.metadata for doc in split_documents]
        return texts, metadatas

    def _load_directory_documents(self):
        documents = []

        for root, _, files in os.walk(self.directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                if filename.lower().endswith(".pdf"):
                    documents.extend(self._load_pdf_with_fallback(file_path))
                else:
                    documents.extend(self._load_non_pdf(file_path))

        return documents

    def _load_pdf_with_fallback(self, file_path: str):
        try:
            return PyMuPDFLoader(file_path).load()
        except Exception as exc:
            logging.warning(
                "PyMuPDF failed for %s, falling back to UnstructuredFileLoader: %s",
                file_path,
                exc,
            )
            return self._load_non_pdf(file_path)

    def _load_non_pdf(self, file_path: str):
        try:
            return UnstructuredFileLoader(file_path).load()
        except Exception as exc:
            logging.warning("Skipping unreadable file %s: %s", file_path, exc)
            return []
