"""Service class that manages the AI document loading, embedding, and retrieval logic."""
from src.common.APIException import APIException
from src.ai.base_llm import BaseLLM
from src.ai.document_loaders.document_loader_facade import DocumentLoaderFacade
from src.ai.vector_store.faiss_vector_store import FaissVectorStore
from src.common.config import CONFIG
import logging

class RagService:    

    def __init__(self,llm: BaseLLM, embedder_service: FaissVectorStore, system_prompt: str, default_k: int):
        self.__llm: BaseLLM = llm
        self.__vector_store: FaissVectorStore = embedder_service
        
        # Load AI service configuration
        self._system_prompt = system_prompt
        self._default_k = default_k

    def initialize_vector_store(self, document_loader_facade: DocumentLoaderFacade):
        """Initialize the vector store from documents. Should be called at startup."""
        if not self.__vector_store.load_from_disk():
            logging.info("Nothing found in vector store, building from documents")
            texts, metadatas = [], []

            # Load "normal" text-based documents
            logging.info("Loading text-based documents")
            docs_texts, docs_metadatas = document_loader_facade.text_document_loader.load_documents()
            texts.extend(docs_texts)
            metadatas.extend(docs_metadatas)

            # Load OCR-based scanned PDFs
            logging.info("Loading OCR-based scanned PDFs")
            ocr_texts, ocr_metadatas = document_loader_facade.ocr_document_loader.load_documents()
            texts.extend(ocr_texts)
            metadatas.extend(ocr_metadatas)

            if not texts or not metadatas:
                logging.warning("No documents found to initialize vector store")
                return
            
            self.__vector_store.add_documents(texts, metadatas)
            self.__vector_store.save_to_disk()
            logging.info("✅ Vector store initialized successfully with documents")
        else:
            logging.info("✅ Vector store loaded from disk")
               

    def answer_question(self, question: str, k: int = None) -> dict:
        """Answer a question using the LLM and relevant docs."""
        k = k or self._default_k
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
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt}
        ])
        return {
            "answer": response,
            "sources": list({frozenset(doc["metadata"].items()): doc["metadata"] for doc in relevant_docs}.values())
        }
        
    def _get_relevant_docs(self, query, k=None):
        """Retrieve relevant documents for a query."""
        k = k or self._default_k
        return self.__vector_store.search(query, k=k)
        