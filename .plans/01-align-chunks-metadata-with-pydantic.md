# Align Chunks And Metadata With Pydantic

## Objective

Refactor the document loading and vector store ingestion flow so each chunk and its metadata are represented as one validated object instead of two parallel lists.

Current fragile model:

```python
texts: list[str]
metadatas: list[dict]
```

Target model:

```python
chunks: list[DocumentChunk]
```

Each `DocumentChunk` must contain:

```python
content: str
metadata: ChunkMetadata
```

This guarantees chunk content and metadata stay connected from creation, through embedding, to persistence and retrieval.

## Explicit Requirement

Introduce Pydantic models for chunk and metadata validation.

Do not use plain dictionaries as the primary contract between loaders and vector store ingestion.

## Proposed Model

Add a new module:

```text
src/ai/document_loaders/document_chunk.py
```

Define Pydantic models similar to:

```python
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    chunk_index: int
    loader: str
    page: int | None = None
    lang: str | None = None
    ocr: bool = False
    llm_extracted: bool = False


class DocumentChunk(BaseModel):
    content: str = Field(min_length=1)
    metadata: ChunkMetadata
```

Use stricter literals for `loader` if desired:

```python
Literal["text", "ocr", "llm_extractor"]
```

## Metadata Schema

Every chunk must have the same base metadata keys:

```python
{
    "chunk_id": "stable unique chunk id",
    "document_id": "stable document id",
    "source": "path/to/original/file",
    "page": 1,
    "chunk_index": 0,
    "lang": "ara",
    "loader": "text|ocr|llm_extractor",
    "ocr": False,
    "llm_extracted": False,
}
```

Required fields:

```text
chunk_id
document_id
source
chunk_index
loader
```

Optional fields:

```text
page
lang
ocr
llm_extracted
```

## Stable ID Helper

Add a small helper module, for example:

```text
src/ai/document_loaders/chunk_metadata_builder.py
```

Responsibilities:

```text
create stable document_id from source path or file checksum
create stable chunk_id from document_id, page, loader, chunk_index
build ChunkMetadata consistently for all loaders
```

Minimal implementation idea:

```python
import hashlib


def stable_document_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
```

Chunk ID format example:

```text
{document_id}:loader:{loader}:page:{page}:chunk:{chunk_index}
```

For text files without page numbers:

```text
{document_id}:loader:text:chunk:{chunk_index}
```

## Loader Contract Change

Update:

```text
src/ai/document_loaders/base_document_loader.py
```

From:

```python
def load_documents(self) -> tuple[list[str], list[dict]]:
```

To:

```python
def load_documents(self) -> list[DocumentChunk]:
```

## Text Loader Change

Update:

```text
src/ai/document_loaders/text_document_loader.py
```

Current behavior:

```python
texts = [doc.page_content for doc in split_documents]
metadatas = [doc.metadata for doc in split_documents]
return texts, metadatas
```

Target behavior:

```python
chunks: list[DocumentChunk] = []

for chunk_index, doc in enumerate(split_documents):
    source = doc.metadata.get("source", self.directory)
    metadata = build_chunk_metadata(
        source=source,
        chunk_index=chunk_index,
        loader="text",
        page=doc.metadata.get("page"),
    )
    chunks.append(DocumentChunk(content=doc.page_content, metadata=metadata))

return chunks
```

## OCR Loader Change

Update:

```text
src/ai/document_loaders/ocr_document_loader.py
```

Current behavior appends to separate lists:

```python
ocr_texts.append(chunk)
ocr_metadatas.append({...})
```

Target behavior:

```python
document_chunks: list[DocumentChunk] = []

for page_index, img in enumerate(images, start=1):
    text = pytesseract.image_to_string(img, lang=self.lang)
    page_chunks = self.text_splitter.split_text(text)

    for chunk_index, chunk in enumerate(page_chunks):
        metadata = build_chunk_metadata(
            source=pdf_path,
            page=page_index,
            chunk_index=chunk_index,
            lang=self.lang,
            loader="ocr",
            ocr=True,
        )
        document_chunks.append(DocumentChunk(content=chunk, metadata=metadata))
```

## LLM Extractor Loader Change

Update:

```text
src/ai/document_loaders/llm_extractor_document_loader.py
```

Current behavior appends to separate lists:

```python
texts.append(chunk)
metadatas.append({...})
```

Target behavior:

```python
for chunk_index, chunk in enumerate(chunks):
    metadata = build_chunk_metadata(
        source=pdf_path,
        page=page_num,
        chunk_index=chunk_index,
        lang=self.__lang,
        loader="llm_extractor",
        llm_extracted=True,
    )
    texts.append(DocumentChunk(content=chunk, metadata=metadata))
```

Rename local variables from `texts` to `document_chunks` where practical to avoid confusion.

## Facade Change

Update:

```text
src/ai/document_loaders/document_loader_facade.py
```

From:

```python
all_texts: list[str] = []
all_metadatas: list[dict] = []

texts, metadatas = loader.load_documents()
all_texts.extend(texts)
all_metadatas.extend(metadatas)

return all_texts, all_metadatas
```

To:

```python
all_chunks: list[DocumentChunk] = []

chunks = loader.load_documents()
all_chunks.extend(chunks)

return all_chunks
```

## Vector Store Initializer Change

Update:

```text
src/ai/vector_store_service/faiss/faiss_vector_store_initializer.py
```

From:

```python
texts, metadatas = self.__document_loader_facade.load_all_documents()
self.__vector_store.add_documents(texts, metadatas)
```

To:

```python
chunks = self.__document_loader_facade.load_all_documents()
self.__vector_store.add_documents(chunks)
```

## Vector Store Contract Change

Update:

```text
src/ai/vector_store_service/base_vector_store.py
src/ai/vector_store_service/faiss/faiss_vector_store.py
src/ai/vector_store_service/vector_store_facade.py
```

From:

```python
def add_documents(self, texts: list[str], metadatas: list[dict] | None = None):
```

To:

```python
def add_documents(self, chunks: list[DocumentChunk]) -> None:
```

FAISS implementation target:

```python
def add_documents(self, chunks: list[DocumentChunk]) -> None:
    if not chunks:
        return

    texts = [chunk.content for chunk in chunks]
    embeddings = self.embedder.embed_documents(texts)

    embeddings_np = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_np)
    self.index.add(embeddings_np)

    self.documents.extend([
        {
            "content": chunk.content,
            "metadata": chunk.metadata.model_dump(),
        }
        for chunk in chunks
    ])

    self.save_to_disk()
```

## Search Behavior

Search can remain mostly unchanged:

```python
return [self.documents[idx] for idx in I[0] if idx < len(self.documents)]
```

Returned records should keep this shape:

```python
{
    "content": "chunk text",
    "metadata": {
        "chunk_id": "...",
        "document_id": "...",
        "source": "...",
        "page": 1,
        "chunk_index": 0,
        "loader": "ocr",
    },
}
```

## Persistence

Current persistence remains valid:

```text
vector_store/faiss.index      -> vectors
vector_store/metadata.pkl     -> chunk content + metadata
```

Change only the contents of `metadata.pkl` so all records use the same metadata schema.

Existing `metadata.pkl` should be treated as incompatible after this refactor unless a migration is written.

Recommended development approach:

```text
delete/rebuild vector_store/faiss.index
delete/rebuild vector_store/metadata.pkl
```

Do not silently mix old and new metadata structures.

## Tests

Update existing tests affected by loader return types:

```text
tests/test_chunks_retrieval.py
tests/test_llm_extractor_loader.py
```

Add or update assertions:

```python
assert chunk.content
assert chunk.metadata.chunk_id
assert chunk.metadata.document_id
assert chunk.metadata.source
assert chunk.metadata.chunk_index >= 0
assert chunk.metadata.loader in {"text", "ocr", "llm_extractor"}
```

Add vector store ingestion assertions:

```python
assert stored_doc["content"]
assert stored_doc["metadata"]["chunk_id"]
assert stored_doc["metadata"]["document_id"]
assert stored_doc["metadata"]["source"]
```

## Verification

After implementation, run:

```bash
pytest
```

Run linting if configured:

```bash
ruff check .
```

If this project still relies on pylint instead of ruff, run the existing configured lint command.

## Migration Note

This is a breaking internal data-shape change for persisted vector metadata.

Before running the app after implementation, rebuild the vector store or add an explicit migration.

Preferred simple path for this project:

```text
remove old FAISS index and metadata pickle
restart app
let VectorStoreInitializer rebuild from source documents
```

## Expected Final Flow

```text
DocumentLoader
-> list[DocumentChunk]
-> VectorStoreInitializer
-> FaissVectorStore.add_documents(chunks)
-> embed chunk.content
-> save vectors to FAISS
-> save {content, metadata} to metadata.pkl
-> search returns aligned chunk + metadata
```

This removes the parallel-list alignment risk permanently.
