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
   Select the LLM provider you want to use in the **[config.yaml](./config.yaml)** file, in the **providers** section for both llm and embeddings 
   ```bash
   # Provider Configuration
   # Set to 'azure' or 'gemini' to choose the AI provider
   providers:
      llm: "gemini" # or "azure"
      embeddings: "gemini" # or "azure"
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
│   │   ├── vector_store_service/  # FAISS vector store
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

Example `.env` file:
```
GOOGLE_API_KEY="your_GOOGLE_API_KEY_here"
AZURE_OPENAI_API_KEY="your_azure_openai_api_key_here"
AZURE_OPENAI_ENDPOINT="https://your-azure-openai-instance.openai.azure.com/"
```

## Configuration

The `config.yaml` controls provider selection, model parameters, vector store paths, and document loading.

- **`azure`**: Azure OpenAI settings: `api_version`, `deployment` (chat model), `embedding_deployment` (embeddings).
- **`gemini`**: Gemini settings: `model` (chat model), `embedding_model` (embeddings).
- **`vector_store`**: FAISS file locations: `index_path`, `metadata_path`.
- **`llm`**: Runtime behavior and provider selection.
  - `provider`: active provider name (`azure` or `gemini`).
  - `system_prompt`: system instruction used at query time.
  - `default_k`: number of chunks to retrieve per query.
  - `azure`: LLM parameters for Azure (`temperature`, `top_p`, `max_tokens`, `embeddings_dimension`).
  - `gemini`: LLM parameters for Gemini (`temperature`, `top_p`, `max_output_tokens`, `embeddings_dimension`).
- **`document_loader`**: Chunking and folders for ingestion.
  - `chunk_size`, `chunk_overlap`, `docs_directory`.
  - `ocr_docs_dir`: folder for scanned PDFs/images to run OCR.
  - `llm_extractor_docs_dir`: folder for non-text PDFs processed via LLM extraction.
  - `scanned_docs_lang`: ISO code for OCR language.

Adjust these values to switch providers and tune retrieval/generation without code changes.


### Key Features

- #### Plug-and-Play Docs Support (Markdown, PDF, DOCX) 
Add `.md` files to the `docs/` folder — no special formatting needed.

- #### Embedding Generation  
Converts your documentation into vector embeddings using either **Azure OpenAI** or **Gemini**, depending on your configuration.

- #### Local Vector Store with FAISS  
Stores embeddings on disk and then retrieve them at startup for the semantic search. NB: This will be replaced with a vector db soon

- #### Flexible Model Switching  
Easily switch between providers (**Azure**, **Gemini**) via environment variables.

- #### FastAPI Interface  
Exposes an `/ask` endpoint for querying the documentation and receiving answers grounded in your content.

---

### ⚙️ How It Works

- #### 📄 Load Documentation  
Content is read from `docs/`, with optional intake from `ocr_docs/` (scanned PDFs/images via OCR) and `llm_extractor_docs/` (LLM-based text extraction from scanned file, like OCR but better perfomances). Files are split into semantic chunks.

- #### 🔢 Generate Embeddings  
Each chunk is converted into a vector using your configured LLM provider.

- #### 💾 Store with FAISS  
Embeddings are saved locally using FAISS for efficient similarity search.

- #### 🤖 Query via API  
Users send questions to the `/ask` endpoint. The system retrieves the top relevant chunks and passes them to the LLM to generate a grounded response.


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



## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.