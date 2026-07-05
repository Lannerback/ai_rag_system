import logging
import os
from typing import List, Tuple

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

from src.ai.document_loaders.base_document_loader import BaseDocumentLoader
from src.ai.document_loaders.element_cleaner import ElementCleaner
from src.ai.document_loaders.legal_section_tagger import LegalSectionTagger


class TextDocumentLoader(BaseDocumentLoader):
    """Loads text-based files (PDF, DOCX, Markdown) into content chunks.

    Pipeline: partition into typed elements (PDFs use `partition_pdf` with the
    `fast` strategy, which reconstructs two-column reading order that the generic
    auto-partitioner scrambles) -> `ElementCleaner` drops layout chrome and margin
    noise -> `chunk_by_title` (section-aware merge). Filtering must precede chunking
    because element categories only exist on raw elements.
    """

    # Provenance keys worth persisting; everything else Unstructured emits is dropped.
    _KEEP_METADATA_KEYS = ("source", "filename", "page", "language")

    def __init__(
        self, directory: str, chunk_size: int, chunk_overlap: int, pdf_strategy: str = "fast"
    ):
        self.directory = directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._pdf_strategy = pdf_strategy
        self._cleaner = ElementCleaner()
        self._tagger = LegalSectionTagger()

    def load_documents(self) -> Tuple[List[str], List[dict]]:
        if not os.path.isdir(self.directory):
            logging.warning(f"Docs directory not found: {self.directory}")
            return [], []

        texts: List[str] = []
        metadatas: List[dict] = []
        for path in self._iter_files():
            elements = self._partition(path)
            kept = self._cleaner.clean(elements)
            chunks = chunk_by_title(
                kept,
                max_characters=self.chunk_size,
                new_after_n_chars=int(self.chunk_size * 0.8),
                combine_text_under_n_chars=int(self.chunk_size * 0.35),
                overlap=self.chunk_overlap,
                multipage_sections=True,
            )
            file_texts, file_metadatas = self._chunks_to_texts_metadatas(chunks, source=path)
            if path.lower().endswith(".pdf"):
                file_texts = self._tagger.tag(file_texts)
            texts.extend(file_texts)
            metadatas.extend(file_metadatas)
        return texts, metadatas

    def _partition(self, path: str) -> List:
        """Partition a file into typed elements, using the PDF-specific reader for PDFs."""
        if path.lower().endswith(".pdf"):
            return partition_pdf(filename=path, languages=["eng"], strategy=self._pdf_strategy)
        return partition(filename=path, languages=["eng"])

    def _iter_files(self) -> List[str]:
        paths: List[str] = []
        for root, _, files in os.walk(self.directory):
            for name in files:
                if not name.startswith("."):
                    paths.append(os.path.join(root, name))
        return sorted(paths)

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
