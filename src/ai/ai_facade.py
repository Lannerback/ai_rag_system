"""Facade for AI document loading, embedding, and retrieval logic."""
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from src.common.APIException import APIException

from .document_loader import DocumentLoader
from src.ai.base_llm import BaseLLM
from src.ai.base_embedder import BaseEmbedder

from .azure.azure_openai_llm import AzureLLM
from .azure.azure_embeddings import AzureEmbeddingStore


load_dotenv()

class AiFacade:    

    def __init__(self):
        self.__llm: BaseLLM = AzureLLM()
        self.__loader: DocumentLoader = self._build_loader()
        self.__store: BaseEmbedder = self._build_store()    
            
        self._initialize_store()
        

    
    def _build_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            temperature=0.7,
            max_tokens=500
        )    
    
    def _build_loader(self) -> DocumentLoader:
        return DocumentLoader()


    def _build_store(self) -> BaseEmbedder:        
        return AzureEmbeddingStore()

    def _initialize_store(self):
        """Load or build the vector store from documents."""
        if not self.__store.load_from_disk():
            documents = self.__loader.load_documents()
            self.__store.add_documents(
                texts=[doc["content"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents]
            )
            self.__store.save_to_disk()

    def _get_relevant_docs(self, query, k=3):
        """Retrieve relevant documents for a query."""
        return self.__store.search(query, k=k)

    def answer_question(self, question: str, k: int = 3) -> dict:
        """Answer a question using the LLM and relevant docs."""
        relevant_docs = self._get_relevant_docs(question, k=k)
        if not relevant_docs:
            raise APIException(detail = "No relevant documentation found for the question.", status_code=400, code = "no_docs_found")

        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        prompt = f"""Based on the following documentation excerpts, please answer the question.
                        If you cannot answer the question based on the provided context, say so.

                        Documentation:
                        {context}

                        Question: {question}

                        Answer:"""
        response = self.__llm.invoke([
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documentation."},
            {"role": "user", "content": prompt}
        ])
        return {
            "answer": response.content,
            "sources": [doc["metadata"] for doc in relevant_docs]
        }