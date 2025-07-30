"""Module for loading and processing markdown documents."""
import os
from typing import List, Dict
from pathlib import Path
from langchain_text_splitters import MarkdownTextSplitter
from src.common.config import CONFIG

class DocumentLoader:
    def __init__(self, docs_dir: str = None):
        self.docs_dir = docs_dir or CONFIG["document_loader"]["docs_directory"]
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=CONFIG["document_loader"]["chunk_size"],
            chunk_overlap=CONFIG["document_loader"]["chunk_overlap"]
        )

    def load_documents(self) -> List[Dict]:
        """Load and chunk all markdown documents from the docs directory."""
        documents = []
        
        # Read inside the docs directory
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split the content into chunks
                    chunks = self.text_splitter.split_text(content)
                    
                    # Create document objects with metadata
                    for chunk in chunks:
                        documents.append({
                            "content": chunk,
                            "metadata": {
                                "source": str(file_path),
                                "type": "markdown"
                            }
                        })
        
        return documents
