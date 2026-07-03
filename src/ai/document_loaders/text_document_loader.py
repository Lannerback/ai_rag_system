import logging
import os
from typing import List, Tuple

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from src.ai.document_loaders.base_document_loader import BaseDocumentLoader


class TextDocumentLoader(BaseDocumentLoader):
    """Loads text-based files (PDF, DOCX, Markdown) into content chunks.

    Pipeline: partition into typed elements -> drop layout chrome (headers/footers/
    page numbers) -> `chunk_by_title` (section-aware merge of small elements). Filtering
    must precede chunking because element categories only exist on raw elements.
    """

    # Layout chrome with no semantic value; embedding these pollutes retrieval with
    # near-duplicate boilerplate (page footers, running headers, page numbers).
    _DROP_CATEGORIES = frozenset({"Header", "Footer", "PageNumber", "PageBreak", "Image"})

    # Provenance keys worth persisting; everything else Unstructured emits is dropped.
    _KEEP_METADATA_KEYS = ("source", "filename", "page", "language")

    def __init__(self, directory: str, chunk_size: int, chunk_overlap: int):
        self.directory = directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self) -> Tuple[List[str], List[dict]]:
        if not os.path.isdir(self.directory):
            logging.warning(f"Docs directory not found: {self.directory}")
            return [], []

        texts: List[str] = []
        metadatas: List[dict] = []
        for path in self._iter_files():
            elements = partition(filename=path, languages=["eng"])
            kept = [el for el in elements if self._keep_element(el)]
            chunks = chunk_by_title(
                kept,
                max_characters=self.chunk_size,
                new_after_n_chars=int(self.chunk_size * 0.8),
                combine_text_under_n_chars=int(self.chunk_size * 0.25),
                overlap=self.chunk_overlap,
                multipage_sections=True,
            )
            file_texts, file_metadatas = self._chunks_to_texts_metadatas(chunks, source=path)
            texts.extend(file_texts)
            metadatas.extend(file_metadatas)
        return texts, metadatas

    def _iter_files(self) -> List[str]:
        paths: List[str] = []
        for root, _, files in os.walk(self.directory):
            for name in files:
                if not name.startswith("."):
                    paths.append(os.path.join(root, name))
        return sorted(paths)

    @classmethod
    def _keep_element(cls, element) -> bool:
        """Drop layout-chrome elements; keep everything with real content."""
        return getattr(element, "category", None) not in cls._DROP_CATEGORIES

    @classmethod
    def _chunks_to_texts_metadatas(
        cls, chunks, source: str
    ) -> Tuple[List[str], List[dict]]:
        """Map chunks to aligned (texts, metadatas), pruning metadata and dropping blanks."""
        texts: List[str] = []
        metadatas: List[dict] = []
        for chunk in chunks:
            text = (chunk.text or "").strip()
            if not text:
                continue
            texts.append(text)
            metadatas.append(cls._chunk_metadata(chunk, source))
        return texts, metadatas

    @classmethod
    def _chunk_metadata(cls, chunk, source: str) -> dict:
        """Build the allowlisted metadata for a chunk (source/filename/page/language)."""
        meta = chunk.metadata
        raw = {
            "source": source,
            "filename": getattr(meta, "filename", None),
            "page": getattr(meta, "page_number", None),
            "language": (getattr(meta, "languages", None) or [None])[0],
        }
        return {key: value for key, value in raw.items() if key in cls._KEEP_METADATA_KEYS and value is not None}
