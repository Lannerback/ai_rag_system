from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common.config import CONFIG
import logging
import os
from typing import List, Tuple

class TextDocumentLoader:
    def __init__(self, directory: str, chunk_size: int, chunk_overlap: int):
        self.directory = directory
        self.loader: DirectoryLoader = DirectoryLoader(directory, glob="**/*", loader_cls=UnstructuredFileLoader)
        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)      
    
    def load_documents(self) -> Tuple[List[str], List[dict]]:
        """Load documents from the directory, split into chunks, and return texts + metadata."""
        # Check if directory exists
        if not os.path.exists(self.directory) or not os.path.isdir(self.directory):
            logging.warning(f"Docs directory not found: {self.directory}")
            return [], []

        # Load and split
        documents = self.loader.load()
        split_documents = self.text_splitter.split_documents(documents)

        texts = [doc.page_content for doc in split_documents]
        metadatas = [doc.metadata for doc in split_documents]
        return texts, metadatas
