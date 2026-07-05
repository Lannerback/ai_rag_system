from src.ai.vector_store_service.base_vector_store import BaseVectorStore
from typing import List, Dict

class VectorStoreFacade:
    def __init__(self, vector_store: BaseVectorStore):
        self.__vector_store: BaseVectorStore = vector_store

    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        self.__vector_store.add_documents(texts, metadatas)

    def search(self, question: str, k: int, with_embeddings: bool = False) -> List[Dict]:
        return self.__vector_store.search(question, k=k, with_embeddings=with_embeddings)

               