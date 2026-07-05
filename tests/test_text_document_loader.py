from types import SimpleNamespace

from src.ai.document_loaders.text_document_loader import TextDocumentLoader


def _chunk(text, *, filename="reg.pdf", page_number=1, languages=("eng",)):
    metadata = SimpleNamespace(filename=filename, page_number=page_number, languages=list(languages))
    return SimpleNamespace(text=text, metadata=metadata)


def test_chunks_mapping_prunes_metadata_and_maps_page_language():
    texts, metadatas = TextDocumentLoader._chunks_to_texts_metadatas(
        [_chunk("Body text.", filename="reg.pdf", page_number=5, languages=("eng",))],
        source="docs/reg.pdf",
    )

    assert texts == ["Body text."]
    assert metadatas[0] == {
        "source": "docs/reg.pdf",
        "filename": "reg.pdf",
        "page": 5,
        "language": "eng",
    }


def test_chunks_mapping_drops_blank_chunks_and_keeps_alignment():
    chunks = [
        _chunk("Real content.", page_number=1),
        _chunk("   \n\t ", page_number=2),
        _chunk("More content.", page_number=3),
    ]

    texts, metadatas = TextDocumentLoader._chunks_to_texts_metadatas(chunks, source="docs/reg.pdf")

    assert texts == ["Real content.", "More content."]
    assert [m["page"] for m in metadatas] == [1, 3]


def test_chunk_metadata_omits_missing_fields():
    # Markdown has no page concept; missing values must be omitted, not stored as None.
    chunk = _chunk("md body", filename="notes.md", page_number=None, languages=())

    metadata = TextDocumentLoader._chunk_metadata(chunk, source="docs/notes.md")

    assert metadata == {"source": "docs/notes.md", "filename": "notes.md"}


def test_load_documents_missing_directory_returns_empty(tmp_path):
    loader = TextDocumentLoader(str(tmp_path / "missing"), chunk_size=1200, chunk_overlap=150)

    assert loader.load_documents() == ([], [])
