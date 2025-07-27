# AI Assistant - Documentation-Based RAG System

This project is a Retrieval-Augmented Generation (RAG) system that leverages Markdown files as its knowledge base. Simply place your documentation inside the docs/ folder to get started.

The system supports both Gemini and Azure OpenAI models interchangeably, thanks to a clean abstraction layer. This design allows for easy integration of additional models in the fut

## Key Features

### Plug-and-Play Markdown Support  
Add `.md` files to the `docs/` folder — no special formatting needed.

### Embedding Generation  
Converts your documentation into vector embeddings using either **Azure OpenAI** or **Gemini**, depending on your configuration.

### Local Vector Store with FAISS  
Stores embeddings on disk for fast, scalable semantic search.

### Flexible Model Switching  
Easily switch between providers (**Azure**, **Gemini**) via environment variables.

### FastAPI Interface  
Exposes an `/ask` endpoint for querying the documentation and receiving answers grounded in your content.

---

## ⚙️ How It Works

### 📄 Load Documentation  
Markdown files are read from the `docs/` folder and split into semantic chunks.

### 🔢 Generate Embeddings  
Each chunk is converted into a vector using your configured LLM provider.

### 💾 Store with FAISS  
Embeddings are saved locally using FAISS for efficient similarity search.

### 🤖 Query via API  
Users send questions to the `/ask` endpoint. The system retrieves the top relevant chunks and passes them to the LLM to generate a grounded response.


## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
4. Add your markdown documentation files to the `docs/` directory

The application at the startup will check if the **vectore_store** folder already exists, if so, it will start loading the **vectore store from local disk**, otherwise it will 
create **embeddings** from the docs and create the vectore store folder.

## Usage

1. Start the server:
   ```bash
   python main.py
   ```

2. The API will be available at `http://localhost:8000`

3. Use the `/ask` endpoint to ask questions:
   ```bash
   curl -X POST http://localhost:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"text": "What is the purpose of this system?"}'
   ```

## API Reference

### POST /ask

Ask a question of which the answer can be found in your documentation.

Request body:
```json
{
    "text": "Your question here"
}
```

Response body:
```json
{
    "answer": "The answer based on the documentation",
    "sources": [
        {
            "source": "path/to/document.md",
            "type": "markdown"
        }
    ]
}
```

## Environment Variables

Check the .env.example

## Tech choices

### Why use FastApi over flask (or Django and others)? 
- FastApi has pydentic hint check, automatically checks for the error in the user Request
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



