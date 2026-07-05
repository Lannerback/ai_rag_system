# AI Assistant - Documentation-Based RAG System
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async%20web%20framework-green)
![LangChain](https://img.shields.io/badge/langchain-enabled-brightgreen)
![Status](https://img.shields.io/badge/status-active-brightgreen)

This project is a Retrieval-Augmented Generation (RAG) system that leverages Markdown files as its knowledge base. Simply place your documentation inside the docs/ folder to get started.
The documentation extensions supported are PDF,MD,DOCX.

The system supports both Gemini and Azure OpenAI models interchangeably, thanks to a clean abstraction layer. This design allows for easy integration of additional models in the future.

## Who is this for?

- Developers who want to build a chatbot over internal docs
- Teams with Markdown/PDF/DOCX knowledge bases
- Anyone who wants a local RAG system using Gemini or Azure, or want to implement a new one really easily

## 🚀 Quickstart

1. Clone the repository:
   ```bash
   git clone https://github.com/Lannerback/ai_rag_system.git
   cd ai-rag-system
2. Create a virtual environment (or create the conda environment with the "environment.yaml" conda file):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
3. Set up your .env file:
   ```bash
   cp .env.example .env
   # Edit .env and insert your API keys
4. Select provider
   Select the AI provider you want to use in the **[config.yaml](./config.yaml)** file. A
   single `llm.provider` value drives both the chat model and the embeddings for that
   provider.
   ```yaml
   llm:
     # Set to 'azure' or 'gemini' to choose the AI provider
     provider: "gemini"   # or "azure"
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
6. Fill docs folder
   Fill the **docs** folder with documents you need to be used as knowledge for your RAG system 
7. Run the app:
   ```bash
   python main.py
   
8. The API will be available at `http://localhost:8000`

9. Use the `/ask` endpoint to ask questions:
   ```bash
   curl -X POST http://localhost:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"text": "What is the purpose of this system?"}'
   ```

The application at the startup will check if the **vector_store** folder already exists, if so, it will start loading the **vector store from local disk**, otherwise it will 
create **embeddings** from the docs and create the vector store Plug-and-Play Docs Support (Markdown, PDF, DOCX)plugfolder.

### Alternative: using `uv`

[`uv`](https://docs.astral.sh/uv/) is fully supported as a faster alternative to the conda/pip setup
above (steps 2 and 5). Both paths install the same pinned dependencies; pick whichever you prefer.

1. System prerequisites (not pip-installable — provide the `pdftoppm`/`pdftocairo` and OCR binaries
   used by the PDF/OCR loaders):
   ```bash
   brew install poppler tesseract        # macOS
   # apt-get install poppler-utils tesseract-ocr   # Debian/Ubuntu
   ```
2. Install dependencies (runtime + dev tools) and create `.venv/`:
   ```bash
   uv sync --group dev
   ```
3. Continue with steps 3–4 above (`.env`, provider selection in `config.yaml`), then:
   ```bash
   uv run alembic upgrade head                                  # if using the pgvector backend
   uv run python -m src.ingestion --collection default          # ingest docs (add --rebuild to re-embed everything)
   uv run python main.py                                        # start the app
   uv run pytest tests/                                         # run tests
   uv run ruff check src/                                       # lint
   ```

## Folders structure
```bash
├── docs/                          # Primary knowledge base (MD/PDF/DOCX)
├── ocr_docs/                      # Scanned PDFs/images to be OCR-processed
├── llm_extractor_docs/            # Docs for LLM-based text extraction (non-text PDFs)
├── llm_extracted_text_temp/       # Temporary text extracted by OCR/LLM pipelines
├── src/
│   ├── ai/
│   │   ├── embedders/
│   │   │   ├── azure/             # Azure OpenAI LLM + embeddings
│   │   │   └── gemini/            # Gemini LLM + embeddings
│   │   ├── vector_store_service/  # pgvector + FAISS backends (registry-selected)
│   │   ├── rerankers/             # BGE / FlashRank cross-encoder rerankers
│   │   ├── diversity/             # MMR diversity selector (config-disabled)
│   │   ├── document_loaders/      # partition + clean + chunk + legal tagging
│   │   ├── base_llm.py            # LLM abstraction
│   │   └── rag_service.py         # RAG orchestration
│   ├── api.py                     # FastAPI entrypoint
│   └── common/config.py           # Config loader
├── vector_store*/                  # Local FAISS stores (default + provider-specific)
├── scripts/                        # Utilities (OCR, LLM extraction, tests)
├── tests/                          # Unit tests
├── config.yaml
├── .env.example
├── requirements.txt
```

## 🔐 Environment Variables

Set API keys in your `.env` file. See [.env.example](.env.example) for required fields.

-   `GOOGLE_API_KEY`: Your API key for Google Gemini. This is required if `providers.llm` or `providers.embeddings` in `config.yaml` is set to `gemini`.
-   `AZURE_OPENAI_API_KEY`: Your API key for Azure OpenAI. This is required if `providers.llm` or `providers.embeddings` in `config.yaml` is set to `azure`.
-   `AZURE_OPENAI_ENDPOINT`: The endpoint URL for your Azure OpenAI service. Required if using Azure OpenAI.
-   `DATABASE_URL`: PostgreSQL connection string. Required only when `vector_store.backend` is `pgvector`.

Example `.env` file:
```
GOOGLE_API_KEY="your_GOOGLE_API_KEY_here"
AZURE_OPENAI_API_KEY="your_azure_openai_api_key_here"
AZURE_OPENAI_ENDPOINT="https://your-azure-openai-instance.openai.azure.com/"
DATABASE_URL="postgresql+psycopg://admin:admin@localhost:5433/rag"
```

## Configuration

The `config.yaml` controls provider selection, model parameters, vector store paths, and document loading.

- **`azure`**: Azure OpenAI settings: `api_version`, `deployment` (chat model), `embedding_deployment` (embeddings).
- **`gemini`**: Gemini settings: `model` (chat model), `embedding_model` (embeddings).
- **`vector_store`**: Backend selection and storage settings.
  - `backend`: active vector store (`faiss` or `pgvector`). Resolved at startup via a provider registry — no code changes to switch.
  - `collection`: pgvector logical collection name (bound to one embedding provider/model/dimension).
  - `index_path`, `metadata_path`: FAISS file locations (used when `backend: faiss`).
  - `pgvector.metric`: similarity metric for the pgvector backend.
- **`llm`**: Runtime behavior and provider selection.
  - `provider`: active provider name (`azure` or `gemini`).
  - `system_prompt`: system instruction used at query time.
  - `default_k`: **legacy** raw-recall knob. No longer drives `/ask` (see `retrieval:`);
    kept only for the raw vector-search regression tests.
  - `azure`: LLM parameters for Azure (`temperature`, `top_p`, `max_tokens`, `embeddings_dimension`).
  - `gemini`: LLM parameters for Gemini (`temperature`, `top_p`, `max_output_tokens`, `embeddings_dimension`).
- **`retrieval`**: The three-stage `/ask` pipeline.
  - `candidate_k` (120): wide vector-recall pool size (stage 1).
  - `min_similarity` (0.5): cosine-similarity floor on the pool (stage 2); `0.0`/absent disables it.
  - `rerank_k` (30): kept after cross-encoder reranking (stage 3).
  - `final_k` (12): chunks fed into the LLM context (stage 4).
- **`reranking`**: Cross-encoder second-pass scoring.
  - `enabled` (true): set `false` to skip reranking entirely.
  - `provider` (`bge`): `bge` (`BAAI/bge-reranker-v2-m3`) or `flashrank` (`ms-marco-MiniLM-L-12-v2`).
  - `bge.model` / `flashrank.model`: model id per backend.
- **`diversity`**: MMR redundancy-aware final selection. **Commented out by default** (MMR
  disabled → pipeline returns top `final_k`). Uncomment to enable.
  - `strategy` (`mmr`), `lambda_mult` (0.6): 1.0 = pure relevance, 0.0 = pure diversity.
- **`document_loader`**: Chunking and folders for ingestion.
  - `chunk_size` (1200), `chunk_overlap` (150): `chunk_by_title` sizing.
  - `pdf_strategy` (`fast`): Unstructured `partition_pdf` strategy (`fast` | `hi_res`).
  - `docs_directory`: primary document folder.
  - `ocr_docs_dir`: folder for scanned PDFs/images to run OCR.
  - `llm_extractor_docs_dir`: folder for non-text PDFs processed via LLM extraction.
  - `scanned_docs_lang`: ISO code for OCR language.

Adjust these values to switch providers and tune retrieval/generation without code changes.
The rationale for each retrieval/reranking/chunking choice is documented in
[documentation/architect-choices.md](./documentation/architect-choices.md).

## Vector store backends

The backend is chosen by `vector_store.backend` and resolved through a provider registry
(`ServiceFactory.VECTOR_STORE_PROVIDERS`), so switching never touches application code.

### FAISS (default file-based)
Set `vector_store.backend: faiss`. The index and metadata are loaded from / built to the paths
in `vector_store.index_path` / `metadata_path` at startup, exactly as before.

### PostgreSQL + pgvector
Set `vector_store.backend: pgvector` and provide `DATABASE_URL`. Data lives in a
`embedding_collections → documents → chunks` schema; ingestion is an explicit, idempotent command.

1. Start a pgvector-enabled PostgreSQL (or use `docker compose up -d db`).
2. Apply the schema (creates the `vector` extension and tables):
   ```bash
   alembic upgrade head
   ```
3. Ingest documents into a collection:
   ```bash
   python -m src.ingestion --collection default        # incremental (checksum-based)
   python -m src.ingestion --collection default --rebuild  # re-embed everything
   ```
   Ingestion skips unchanged sources, replaces changed ones in a per-document transaction,
   and prunes sources no longer present. A collection is permanently bound to the embedder's
   provider/model/dimension; mismatches fail fast at startup and ingestion.

The app startup does **not** build pgvector data — run ingestion explicitly.


### Key Features

- #### Plug-and-Play Docs Support (Markdown, PDF, DOCX) 
Add `.md` files to the `docs/` folder — no special formatting needed.

- #### Embedding Generation  
Converts your documentation into vector embeddings using either **Azure OpenAI** or **Gemini**, depending on your configuration.

- #### Vector Store: pgvector or FAISS
PostgreSQL + pgvector is the primary backend (idempotent ingestion, checksum-based change
detection, collection/dimension validation). A file-based FAISS backend is also available;
switch with `vector_store.backend` — no code changes.

- #### Retrieve-then-rerank pipeline
Wide cosine vector recall → `min_similarity` floor → local BGE cross-encoder reranking →
optional MMR diversity → top `final_k` into the LLM. Each stage is timed and logged.

- #### Flexible Model Switching  
Easily switch between providers (**Azure**, **Gemini**) via environment variables.

- #### FastAPI Interface  
Exposes an `/ask` endpoint for querying the documentation and receiving answers grounded in your content.

---

### ⚙️ How It Works

- #### 📄 Load Documentation  
Content is read from `docs/`, with optional intake from `ocr_docs/` (scanned PDFs/images via OCR) and `llm_extractor_docs/` (LLM-based text extraction from scanned file, like OCR but better perfomances). Files are split into semantic chunks.

- #### 📑 Clean & Chunk
PDFs are partitioned with Unstructured (`fast` strategy), stripped of layout noise
(headers/footers/page markers) by `ElementCleaner`, chunked section-aware with
`chunk_by_title`, and legal continuation chunks are re-anchored to their `Article` heading.

- #### 🔢 Generate Embeddings  
Each chunk is converted into a vector using your configured provider (Gemini uses asymmetric
`RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` task types for better retrieval).

- #### 💾 Store in pgvector  
Embeddings, chunk text, and metadata are stored in PostgreSQL + pgvector (FAISS is also
supported as a file-based backend, selectable via `vector_store.backend`).

- #### 🤖 Query via API  
Users send questions to the `/ask` endpoint. A three-stage pipeline runs — wide cosine
vector recall (`candidate_k`) → `min_similarity` floor → **BGE cross-encoder reranking**
(`rerank_k`) → optional MMR diversity → top `final_k` chunks — then passes the final chunks
to the LLM for a grounded response.


## Add new provider support
To add support for a new AI provider, follow these steps:

1. **Create provider module**: add `src/ai/embedders/<provider>/` with two files:
   - `<provider>_embedder.py` implementing `src/ai/embedders/base_embedder.py`.
   - `<provider>_llm.py` implementing `src/ai/base_llm.py`.
2. **Export classes**: update `src/ai/embedders/<provider>/__init__.py` as needed.
3. **Wire the factory**: add the provider to `ServiceFactory.PROVIDERS` in `src/ai/service_factory.py` with `{"llm": <YourLLM>, "embedder": <YourEmbedder>}`.
4. **Extend configuration**: add a top-level section in `config.yaml` for provider-specific settings (models, embeddings), and set `llm.provider` to your provider name.
5. **Test**: run ingestion and a sample query to verify embeddings and chat paths work end-to-end.

This keeps the integration consistent with existing Azure and Gemini providers.

## Tech choices

### Why use FastApi over flask (or Django and others)? 
- FastApi has pydentic type validation, automatically checks for the error in the user Request
- it supports asynchonous natively and based on ASGI server (can easily run with uvicorn without adapters)
- lightweight for just API without any support for html or css, and it's natively ashynchronous

### Distance metric choice
I've chosen **cosine similarity** as the distance metric for semantic search in this RAG system. This choice is based on several factors:

- It is the **standard metric** used for **LLM-based retrieval systems**.
- It measures the **angle between vectors**, focusing on their **direction**, not their magnitude or raw distance. This means two vectors with the same meaning but different scales are still treated as similar.
- In semantic search, we're interested more on the meaning instead of the which words you chose. What matters is the **intent of the query**, not the exact words.

For example:
- `"How do I cook potatoes?"`  
- `"What’s the best way to prepare potatoes for dinner?"`  

These sentences use different words, but their embedding points are in the same direction. Cosine similarity captures that, while metrics like Euclidean distance would be sensitive to length differences.

### Why a retrieve-then-rerank pipeline?
A bi-encoder vector search is high-recall but low-precision. Casting a wide net
(`candidate_k=120`) and then rescoring with an expensive, high-accuracy **cross-encoder**
recovers precision without running the cross-encoder over the whole corpus. A
`min_similarity=0.5` cosine floor trims off-topic tail noise first.

### Why BGE over FlashRank for the reranker?
FlashRank (`ms-marco-MiniLM`) is trained on short, general-domain English web search and
**degraded** ranking on our legal corpus (MRR 0.672 → 0.513). `BAAI/bge-reranker-v2-m3` is
larger, multilingual, and trained on longer/formal passages, so it beats flat vector search
on the legal gold set. Both run locally; switch via `reranking.provider`.

### Why is MMR disabled?
MMR diversity is implemented but switched off: after the ingestion-cleanup fixes the
reranked top-12 were already diverse, so MMR added embedding-fetch cost with no recall gain.
Re-enable by uncommenting the `diversity:` block.

Full rationale for every technology and threshold:
[documentation/architect-choices.md](./documentation/architect-choices.md).

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.