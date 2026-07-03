from typing import Dict

from src.ai.vector_store_service.pgvector.models import Chunk


class ChunkMetadataMapper:
    """Maps between the loader metadata dict and promoted columns + JSONB remainder.

    Canonical columns (`page`, `language`, `loader`) are promoted out of the dict;
    everything else stays in JSONB. On reconstruction, canonical columns take
    precedence over duplicate JSONB keys, and `source` is re-injected from the
    owning document so downstream consumers keep seeing the original shape.
    """

    def to_columns(self, metadata: Dict) -> Dict:
        meta = dict(metadata or {})
        meta.pop("source", None)  # stored on the Document, not the chunk

        page = meta.pop("page", None)
        language = meta.pop("language", None)
        lang = meta.pop("lang", None)
        if language is None:
            language = lang

        loader = meta.pop("loader", None)
        is_ocr = meta.pop("ocr", None)
        is_llm = meta.pop("llm_extracted", None)
        if loader is None:
            loader = "llm_extractor" if is_llm else ("ocr" if is_ocr else "text")

        return {
            "page": page,
            "language": language,
            "loader": loader,
            "extra_metadata": meta,
        }

    def to_metadata(self, chunk: Chunk, source: str) -> Dict:
        metadata = dict(chunk.extra_metadata or {})
        metadata["source"] = source
        if chunk.page is not None:
            metadata["page"] = chunk.page
        if chunk.language is not None:
            metadata["language"] = chunk.language
        if chunk.loader is not None:
            metadata["loader"] = chunk.loader
        return metadata
