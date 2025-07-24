"""
API file that create the FastAPI application and defines the endpoints.
"""
import os
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingStore

load_dotenv()

EMBEDDINGS_INDEX_PATH = "vector_store/faiss.index"
EMBEDDINGS_METADATA_PATH = "vector_store/metadata.pkl"

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    temperature=0.7,
    max_tokens=500
)

app = FastAPI(title="AI Assistant",)

loader = DocumentLoader()
store = EmbeddingStore()  # Will be initialized in startup_event

# Load documents on startup of the application
@app.on_event("startup")
async def startup_event():
    if not store.load_from_disk():
        documents = loader.load_documents()
        store.add_documents(
            texts=[doc["content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )
        store.save_to_disk()

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: list[Dict]

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """
        Ask a question and get an answer based on the documentation.
    """
    try:
        # Retrieve relevant documents
        relevant_docs = store.search(question.text)
        
        if not relevant_docs:
            raise HTTPException(status_code=404, detail="No relevant documentation found")
        
        # Construct the prompt with context
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        prompt = f"""Based on the following documentation excerpts, please answer the question.
        If you cannot answer the question based on the provided context, say so.

        Documentation:
        {context}

        Question: {question.text}

        Answer:"""
        
        # Get completion from Azure OpenAI
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documentation."},
            {"role": "user", "content": prompt}
        ])
        
        return Answer(
            answer=response.content,
            sources=[doc["metadata"] for doc in relevant_docs]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
