import os
import logging
from typing import List, Dict
from pathlib import Path
import pdfplumber
from docx import Document as DocxDocument

from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_documents(self) -> List[Dict]:
        documents = []

        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                try:
                    if ext == ".md":
                        content = self._load_markdown(file_path)
                    elif ext == ".pdf":
                        content = self._load_pdf(file_path)
                    elif ext == ".docx":
                        content = self._load_docx(file_path)
                    else:
                        continue

                    chunks = self.text_splitter.split_text(content)
                    for chunk in chunks:
                        documents.append({
                            "content": chunk,
                            "metadata": {
                                "source": str(file_path),
                                "type": ext[1:]  # remove dot
                            }
                        })
                except Exception as e:
                    logging.error(f"Skipping file {file_path} due to error: {e}")

        return documents

    def _load_markdown(self, path: Path) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_pdf(self, path: Path) -> str:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    def _load_docx(self, path: Path) -> str:
        doc = DocxDocument(path)
        return "\n".join([para.text for para in doc.paragraphs])
