"""Module for handling document embeddings and vector storage."""
import os
import pickle
from typing import List, Dict
import numpy as np
import faiss
from langchain_openai import AzureOpenAIEmbeddings

EMBEDDINGS_INDEX_PATH = "vector_store/faiss.index"
EMBEDDINGS_METADATA_PATH = "vector_store/metadata.pkl"

class EmbeddingStore:
    def __init__(self):
        os.makedirs("vector_store", exist_ok=True)
        self.dimension = 1536
        #TODO: change to cosine similarity
        # Use cosine similarity (normalize vectors, then use IndexFlatIP)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict] = []
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            chunk_size=1000
        )

    def load_from_disk(self, index_path=EMBEDDINGS_INDEX_PATH, metadata_path=EMBEDDINGS_METADATA_PATH):
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.documents = pickle.load(f)
            return True
        return False

    def save_to_disk(self, index_path=EMBEDDINGS_INDEX_PATH, metadata_path=EMBEDDINGS_METADATA_PATH):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.documents, f)

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to the vector store and persist them."""
        if not texts:
            return
        embeddings = self.embeddings.embed_documents(texts)
        # Normalize embeddings for cosine similarity
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for text, metadata in zip(texts, metadatas):
            self.documents.append({
                "content": text,
                "metadata": metadata
            })
        # Save index and metadata
        faiss.write_index(self.index, EMBEDDINGS_INDEX_PATH)
        with open(EMBEDDINGS_METADATA_PATH, "wb") as f:
            pickle.dump(self.documents, f)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for most similar documents."""
        query_embedding = self.embeddings.embed_query(query)
        query_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_np)
        D, I = self.index.search(query_np, k)
        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results
