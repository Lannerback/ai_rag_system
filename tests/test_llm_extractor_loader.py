import os
from pathlib import Path

import pytest

from src.ai.document_loaders.llm_extractor_document_loader import (
    LlmExtractorDocumentLoader,
)
from src.ai.service_factory import ServiceFactory


pytestmark = pytest.mark.integration


def _credentials_available() -> bool:
    """Best-effort check for provider credentials."""
    # Gemini
    if os.getenv("GOOGLE_API_KEY"):
        return True
    # Azure
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return True
    return False


@pytest.mark.skipif(not _credentials_available(), reason="Missing LLM credentials in environment")
def test_llm_extractor_real_conversion(tmp_path: Path):
    """
    Integration test: runs the real LLM extraction over a PDF page set.
    It processes a single PDF by copying it to a temp folder to limit scope.
    """
    # Arrange
    source_pdf = Path("llm_extractor_docs") / "03-dattes-WEB.pdf"
    assert source_pdf.exists(), f"Missing test PDF at {source_pdf}"

    work_dir = tmp_path / "llm_pdfs"
    work_dir.mkdir(parents=True, exist_ok=True)
    test_pdf = work_dir / source_pdf.name
    test_pdf.write_bytes(source_pdf.read_bytes())

    llm = ServiceFactory.get_llm()

    loader = LlmExtractorDocumentLoader(
        directory=str(work_dir),
        chunk_size=300,
        chunk_overlap=15,
        lang="ara",
        llm=llm,
    )

    # Act
    texts, metadatas = loader.load_documents()

    # Assert basic output
    assert isinstance(texts, list)
    assert isinstance(metadatas, list)
    assert len(texts) > 0, "Expected at least one text chunk from LLM extraction"
    assert len(texts) == len(metadatas), "Texts and metadatas should align"
    assert any(m.get("llm_extracted") for m in metadatas), "Chunks should be flagged as LLM extracted"

    # Temp file should be created for manual inspection
    out_path = Path.cwd() / "llm_extracted_text_temp" / f"{source_pdf.stem}_extracted.txt"
    assert out_path.exists(), f"Expected temp file to be created at {out_path}"
    content = out_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, "Temp file should not be empty"
