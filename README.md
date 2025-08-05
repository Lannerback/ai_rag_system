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
├── docs/                  # Documentation files (md/pdf/docx)
├── src/
│   ├── ai/                # Embedding + LLM abstraction
│   ├── api.py             # FastAPI entrypoint
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

The `config.yaml` file is the central place to configure various aspects of the RAG system. It allows you to specify which AI models to use, how documents are processed, and other operational parameters.

Here's a breakdown of its key sections:

-   **`azure`**: Configuration for Azure OpenAI, including `api_version`, `deployment` for the LLM, and `embedding_deployment` for the embedding model.
-   **`gemini`**: Configuration for Google Gemini, including `model` for the LLM and `embedding_model` for the embedding model.
-   **`providers`**: Specifies which AI provider to use for the Large Language Model (`llm`) and embeddings. You can set these to `azure` or `gemini`.
-   **`vector_store`**: Defines paths for storing the FAISS index and metadata for the vector store.
-   **`embeddings`**: Contains specific configurations for embedding models, such as `dimension` for both Azure and Gemini, and `chunk_size` and `api_version` for Azure.
-   **`llm`**: Contains specific configurations for the Large Language Models, such as `temperature` and `max_tokens` for Azure, and `temperature`, `top_p`, and `max_output_tokens` for Gemini.
-   **`document_loader`**: Configures how documents are loaded and processed, including `chunk_size`, `chunk_overlap`, and the `docs_directory`.
-   **`ai_service`**: Contains general AI service settings like the `system_prompt` and `default_k` (number of relevant chunks to retrieve).

This modular configuration allows you to easily switch between different AI providers and fine-tune the system's behavior without modifying the core code.


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
Markdown files are read from the `docs/` folder and split into semantic chunks.

- #### 🔢 Generate Embeddings  
Each chunk is converted into a vector using your configured LLM provider.

- #### 💾 Store with FAISS  
Embeddings are saved locally using FAISS for efficient similarity search.

- #### 🤖 Query via API  
Users send questions to the `/ask` endpoint. The system retrieves the top relevant chunks and passes them to the LLM to generate a grounded response.


## Add new provider support
To add support for a new AI provider, follow these steps:

1.  **Create a new folder:** Inside the `src/ai/` directory, create a new folder with the name of your provider (e.g., `src/ai/my_new_provider/`).
2.  **Implement `BaseEmbedder`:** Within your new provider folder, create a class that implements the abstract class `src/ai/base_embedder.py`.
3.  **Implement `BaseLLM`:** Also within your new provider folder, create a class that implements the abstract class `src/ai/base_llm.py`.
4.  **Update `builder_dispatcher`:** Modify `src/ai/builder_dispatcher.py` to include your new provider's classes for both embeddings and LLM, allowing the system to correctly instantiate them based on the `config.yaml` settings.
5.  **Update `config.yaml`:** Add a new section for your provider in `config.yaml` and update the `providers.llm` and `providers.embeddings` fields to reference your new provider.

By following this structure, you can seamlessly integrate new AI models while maintaining a clear and modular codebase.

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