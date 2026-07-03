from abc import ABC, abstractmethod


class BaseVectorStoreInitializer(ABC):
    """Strategy interface for backend-specific vector-store startup behavior.

    Each backend decides what "initialize" means (FAISS: load-or-build from disk;
    pgvector: validate connection + collection). Callers invoke `initialize()`
    polymorphically, so the backend choice never leaks into branching logic.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Prepare the vector store so it is ready to serve queries."""
        pass
